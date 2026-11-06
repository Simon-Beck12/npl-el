# v2 서빙 (EXP-011 1단계 + EXP-012 2단계)

v2 시세 모델(G2, EXP-008)과 회수액 계산(EXP-009)을 FastAPI API와 배치 스크립트로 호출할 수 있게 만든 것이다.
모델 설정·학습 데이터는 바꾸지 않았다(학습+검증 45,634건, num_leaves=63, min_child_samples=20, best_iteration=2,554).
`v2/price_model.py`, `v2/recovery.py`, `src/` 아래 파일은 수정하지 않고 import만 한다.

- 1단계(EXP-011): 로컬 파일(`artifact/model.txt`·`ref.parquet`·`meta.json`)로 서빙.
- 2단계(EXP-012): MLflow로 학습 기록·모델 버전을 관리하고, API·배치가 레지스트리의 champion 별칭 모델을 불러오게 한다. `MODEL_URI`가 있으면 MLflow, 없으면 1단계 방식(`MODEL_DIR`)을 쓴다(`model_loader.py`).

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

## 알아둘 점

- `v2/price_model.py`가 내부적으로 `src/knn_comps.py`(faiss)를 import하는 구조라, `requirements.txt`에 `faiss-cpu`를 추가했다(EXP-008 패키지 목록에는 없던 의존성).
- `python:3.12-slim` 이미지에는 lightgbm이 필요로 하는 `libgomp.so.1`이 없다. EXP-011에서는 apt 없이 faiss-cpu 휠의 libgomp 사본에 심볼릭 링크로 해결했으나, EXP-012 지시로 이미지 안 정식 설치(`apt-get install libgomp1`)로 바꿨다(apt는 Dockerfile 안에서만 쓴다).
- 모델 파일(`artifact/`)과 데이터(`data/`)는 이미지에 복사하지 않고 Compose volume(`./artifact:/app/artifact:ro`)으로 읽기 전용 연결한다. `mlflow_data/`(SQLite+MLflow 아티팩트)도 volume으로만 연결하고 git에 커밋하지 않는다.
- MLflow 모델은 `mlflow.models.set_model`("models from code") 방식으로 로그한다 — cloudpickle로 객체를 직렬화하는 기본 방식은 서빙 컨테이너의 패키지 구성이 학습 때와 조금만 달라도 깨지기 쉬워, 이 파일(`pyfunc_model.py`)을 서빙 환경에서 다시 실행하는 방식으로 바꿨다.
- MLflow 서버(3.x)는 Host 헤더 기반 DNS 리바인딩 방어가 기본 켜져 있다. docker-compose 네트워크 안에서 서비스명으로 접근하는 것과 호스트에서 `localhost`로 접근하는 것을 모두 허용하려면 `--allowed-hosts`에 포트까지 포함한 값(`mlflow:5000`, `localhost:5000` 등)을 넣어야 한다(`docker-compose.yml` 참고).
- 지오코딩은 이번 범위 밖이다 — `/valuation`은 호출자가 직접 lat/lng을 줘야 한다.
- `pyfunc_model.py`의 `NplPriceModel.predict`는 행 단위로 처리해 7,561건 같은 대량 배치에는 느리다(재현 확인 A에 몇 분 걸림). 배치 스코어링(`batch_score.py`)은 수십~수백 행 단위라 문제가 되지 않는다.
