# v2 서빙 (EXP-011 1단계 + EXP-012 2단계 + EXP-013 3단계)

v2 시세 모델(G2, EXP-008)과 회수액 계산(EXP-009)을 FastAPI API와 배치 스크립트로 호출할 수 있게 만든 것이다.
모델 설정·학습 데이터는 바꾸지 않았다(학습+검증 45,634건, num_leaves=63, min_child_samples=20, best_iteration=2,554).
`v2/price_model.py`, `v2/recovery.py`, `src/` 아래 파일은 수정하지 않고 import만 한다.

- 1단계(EXP-011): 로컬 파일(`artifact/model.txt`·`ref.parquet`·`meta.json`)로 서빙.
- 2단계(EXP-012): MLflow로 학습 기록·모델 버전을 관리하고, API·배치가 레지스트리의 champion 별칭 모델을 불러오게 한다. `MODEL_URI`가 있으면 MLflow, 없으면 1단계 방식(`MODEL_DIR`)을 쓴다(`model_loader.py`).
- 3단계(EXP-013): Airflow DAG(`npl_el_daily_rescore`)로 champion 모델 배치 채점을 매일 자동 실행하고 실행 기록을 MLflow `npl-el-batch` 실험에 남긴다. `/valuation`·배치 입력의 주소(구·umdNm·jibun)를 필수로 바꿨다.

## 1단계: 로컬 파일 기반 실행 순서

```bash
# 1) artifact 생성 (model.txt·ref.parquet·meta.json) — git에 커밋하지 않는다
uv run python v2/serving/build_artifact.py

# 2) 단위 테스트
uv run pytest v2/serving/tests/ -v

# 3) Docker Compose로 API만 기동
cd v2/serving
docker compose up -d --build api

# 4) curl 예시 (아래 참조)

# 5) 종료
docker compose down
```

## 2단계: MLflow 기반 실행 순서

```bash
cd v2/serving

# 1) mlflow 서비스 기동 (포트 5000, SQLite+파일 저장은 ./mlflow_data, git에 커밋하지 않는다)
docker compose up -d --build mlflow

# 2) 학습 + MLflow 기록·등록 (호스트에서 실행 — data/ 접근이 필요하다)
cd ..
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python v2/serving/train_register.py
# → npl-el-price 모델에 새 버전 등록, champion 별칭을 그 버전에 붙인다
# → http://localhost:5000 에서 run·등록 모델 확인 가능

# 3) API 기동 (MLFLOW_TRACKING_URI·MODEL_URI는 docker-compose.yml에 이미 설정됨)
cd v2/serving
docker compose up -d --build api

# 4) curl 예시 (아래 참조, /health에 모델 이름·버전·별칭·run_id가 담긴다)

# 5) 종료
docker compose down
```

### 별칭으로 모델 교체하기

새 버전을 champion으로 바꾸려면 `train_register.py`를 다시 돌리면 된다(매번 새 버전을 만들고
champion 별칭을 그 버전으로 옮긴다). 특정 버전을 수동으로 지정하려면:

```python
from mlflow import MlflowClient
MlflowClient().set_registered_model_alias("npl-el-price", "champion", "<version>")
```

API는 `/health`가 캐시한 모델을 쓰므로, champion을 바꾼 뒤 반영하려면 api 컨테이너를 재시작한다
(`docker compose restart api`) — 이번 2단계 범위에는 무중단 모델 교체(hot reload)는 없다.

## 3단계: Airflow 일일 배치 재채점

DAG `npl_el_daily_rescore`(태그 `npl-el`,`v2`,`EXP-013`)는 매일 02:00(Asia/Seoul)에 4개 작업을 순서대로 돈다.

1. `check_input` — 입력 CSV(기본 `v2/serving/examples/sample_input.csv`, `NPL_EL_INPUT_CSV` 환경변수로 override 가능)가 있고 필수 컬럼(`lat,lng,excluUseAr,floor,buildYear,dealYear,dealMonth,구,umdNm,jibun,deposit,ead,months,rho_scenario`)을 다 갖췄는지 확인한다. 하나라도 없으면 실패한다.
2. `score` — `/opt/npl-venv`(Airflow 패키지와 분리된 전용 가상환경, `@task.external_python`)에서 `models:/npl-el-price@champion`을 불러와 채점하고 `v2/serving/airflow/output/rescore_YYYY-MM-DD.csv`에 쓴다.
3. `validate_output` — 입력·출력 행 수 일치, LGD가 0~1 범위, 오류 표시 없는 행에 시세 결측이 없는지 확인한다.
4. `log_to_mlflow` — MLflow `npl-el-batch` 실험에 run을 남긴다(params: model_version, input_file / metrics: n_rows, n_error, avg_lgd, portfolio_lgd=1-ΣPV/ΣEAD).

처음 기동 순서:

```bash
cd v2/serving
cp .env.example .env
echo "AIRFLOW_UID=$(id -u)" > .env   # 호스트 사용자 uid로 맞춘다(중요, 아래 "권한" 참고)
docker compose up -d --build mlflow api airflow
```

- **unpause가 필요할 수 있다**: Airflow는 기본적으로 DAG를 일시정지(`is_paused`) 상태로 만든다. `npl_el_daily_rescore`는 코드에서 `is_paused_upon_creation=False`로 바꿔 두었지만, 과거에 한 번이라도 생성된 적이 있는 DAG(메타DB에 이미 레코드가 있는 경우)는 이 설정이 적용되지 않는다. 수동 트리거 후 run이 몇 분이 지나도 `queued`에 멈춰 있으면 `docker compose exec airflow airflow dags unpause npl_el_daily_rescore`로 확인한다.
- **권한(AIRFLOW_UID)**: `airflow` 서비스는 `user: "${AIRFLOW_UID:-50000}:0"`로 호스트 사용자 uid에 맞춰 돈다(컨테이너 기본 uid 50000으로 쓰면 `v2/serving/airflow/output`·`db`·`logs`에 권한 오류가 난다). 처음 실행 전에 반드시 `.env`를 만든다(위 명령 참고). `.env`는 커밋하지 않는다(`.env.example`만 커밋).
- **폴더가 root 소유로 만들어졌다면**: Docker가 호스트 경로를 먼저 만들면 root 소유가 될 수 있다. 이 경우 `sudo chown -R $(id -u):$(id -g) v2/serving/airflow`로 고친 뒤 다시 올린다.
- **실행 기록·로그 위치**: Airflow 메타DB(SQLite)는 `v2/serving/airflow/db/airflow.db`, 태스크 로그는 `v2/serving/airflow/logs/`에 volume으로 영속화된다(컨테이너를 재생성해도 run 기록이 남는다). 둘 다 `.gitignore`에서 제외하되 폴더 자체는 `.gitkeep`으로 추적한다.
- **수동 실행**: `docker compose exec airflow airflow dags trigger npl_el_daily_rescore`, 상태 확인은 `docker compose exec airflow airflow dags state npl_el_daily_rescore <run_id>`.
- **결과 위치**: 채점 결과는 `v2/serving/airflow/output/rescore_YYYY-MM-DD.csv`, 실행 기록은 MLflow UI(`http://localhost:5000`, 실험 `npl-el-batch`)에서 본다.
- **모델 교체**: `train_register.py`로 새 버전을 등록하고 champion 별칭을 옮긴 뒤(2단계 "별칭으로 모델 교체하기" 참고) `docker compose restart api`를 하면 API가 바로 반영한다. DAG의 `score` 작업은 다음 실행(수동 트리거 또는 다음날 02:00)부터 새 champion을 쓴다 — airflow 컨테이너를 따로 재시작할 필요는 없다.

## curl 예시

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/valuation -H "Content-Type: application/json" -d '{
  "lat":37.6036,"lng":126.9589,"excluUseAr":36.5,"floor":3,"buildYear":2005,
  "dealYear":2026,"dealMonth":9,"gu_name":"종로구","umdNm":"신영동","jibun":"213-**"
}'

curl -s -X POST http://127.0.0.1:8000/recovery -H "Content-Type: application/json" -d '{
  "market_price":48974,"rho":0.644620,"deposit":14700,"ead":34281.8,"months":24
}'
```

배치 채점(예시 입력 3행 → CSV):

```bash
uv run python v2/serving/batch_score.py
```

## 엔드포인트

| 경로 | 설명 |
|---|---|
| GET /health | 모델 소스(MODEL_DIR/MLFLOW), 모델 이름·레지스트리 버전·별칭·run_id(MLflow 경로일 때), test_mdape 등 메타데이터 |
| POST /valuation | 시세(만원), ㎡당 가격, knn_ppm2_5, same_bldg |
| POST /recovery | `compute_recovery_v2` 결과(S,K,P,D2,R,PV,EAD,LGD) |

rho 시나리오 값(EXP-010, 일괄매각 의심 제외 후): 비관 0.5738 / 기본 0.644620 / 낙관 0.8002.

## 재현·검증 결과

| 확인 항목 | 결과 |
|---|---|
| (EXP-011) artifact 재예측 MdAPE | 11.5635%(EXP-008 등록값과 차이 0.000000%p, 허용 0.01%p) |
| (EXP-011) docker compose up -d --build api → curl /health, /valuation, /recovery | 모두 200 응답 |
| (EXP-011) 효창동 예시 /recovery LGD | 58.6010%(기준 58.6%, ±0.1%p) |
| (EXP-011) 범위 밖 입력(excluUseAr=0) /valuation | 422 |
| (EXP-012) 재현 A: MLflow 모델(models:/npl-el-price@champion)로 테스트 7,561건 재예측 MdAPE | 11.563502%(차이 0.000000%p) |
| (EXP-012) 재현 B: sample_input.csv 3행, 1단계 vs 2단계 /valuation 시세 차이 | 0.0(3행 모두) |
| (EXP-012) pytest(`v2/serving/tests`) | 9 passed(MLFLOW_TRACKING_URI 없으면 MLflow 경로 테스트 1개는 skip) |
| (EXP-012) docker compose up -d --build(mlflow+api) → curl 3개 | 모두 200, `/health`가 npl-el-price·champion·run_id 반환 |
| (EXP-013) 주소 없는 /valuation·누락 행 batch_score.py | 422 / 오류 행으로 표시(배치는 중단되지 않음) |
| (EXP-013) DAG 1회 실행 작업 4개 | check_input·score·validate_output·log_to_mlflow 전부 success |
| (EXP-013) DAG 출력 vs batch_score.py 직접 실행 시세 차이 | 0.0(3행 모두) |
| (EXP-013) MLflow npl-el-batch run | model_version=3, n_rows=3, n_error=0, portfolio_lgd=0.578428 |
| (EXP-013) 필수 컬럼 누락 입력 DAG 실행 | check_input 실패(재시도 포함), 나머지 작업은 upstream_failed |

## 알아둘 점

- `v2/price_model.py`가 내부적으로 `src/knn_comps.py`(faiss)를 import하는 구조라, `requirements.txt`에 `faiss-cpu`를 추가했다(EXP-008 패키지 목록에는 없던 의존성).
- `python:3.12-slim` 이미지에는 lightgbm이 필요로 하는 `libgomp.so.1`이 없다. EXP-011에서는 apt 없이 faiss-cpu 휠의 libgomp 사본에 심볼릭 링크로 해결했으나, EXP-012 지시로 이미지 안 정식 설치(`apt-get install libgomp1`)로 바꿨다(apt는 Dockerfile 안에서만 쓴다).
- 모델 파일(`artifact/`)과 데이터(`data/`)는 이미지에 복사하지 않고 Compose volume(`./artifact:/app/artifact:ro`)으로 읽기 전용 연결한다. `mlflow_data/`(SQLite+MLflow 아티팩트)도 volume으로만 연결하고 git에 커밋하지 않는다.
- MLflow 모델은 `mlflow.models.set_model`("models from code") 방식으로 로그한다 — cloudpickle로 객체를 직렬화하는 기본 방식은 서빙 컨테이너의 패키지 구성이 학습 때와 조금만 달라도 깨지기 쉬워, 이 파일(`pyfunc_model.py`)을 서빙 환경에서 다시 실행하는 방식으로 바꿨다.
- MLflow 서버(3.x)는 Host 헤더 기반 DNS 리바인딩 방어가 기본 켜져 있다. docker-compose 네트워크 안에서 서비스명으로 접근하는 것과 호스트에서 `localhost`로 접근하는 것을 모두 허용하려면 `--allowed-hosts`에 포트까지 포함한 값(`mlflow:5000`, `localhost:5000` 등)을 넣어야 한다(`docker-compose.yml` 참고).
- 지오코딩은 이번 범위 밖이다 — `/valuation`은 호출자가 직접 lat/lng을 줘야 한다.
- `pyfunc_model.py`의 `NplPriceModel.predict`는 행 단위로 처리해 7,561건 같은 대량 배치에는 느리다(재현 확인 A에 몇 분 걸림). 배치 스코어링(`batch_score.py`)은 수십~수백 행 단위라 문제가 되지 않는다.
