# v2 서빙 1단계 (EXP-011)

v2 시세 모델(G2, EXP-008)과 회수액 계산(EXP-009)을 FastAPI API와 배치 스크립트로 호출할 수 있게 만든 것이다.
모델 설정·학습 데이터는 바꾸지 않았다(학습+검증 45,634건, num_leaves=63, min_child_samples=20, best_iteration=2,554).
`v2/price_model.py`, `v2/recovery.py`는 수정하지 않고 import만 한다.

## 실행 순서

```bash
# 1) artifact 생성 (model.txt·ref.parquet·meta.json) — git에 커밋하지 않는다
uv run python v2/serving/build_artifact.py

# 2) 단위 테스트
uv run pytest v2/serving/tests/ -v

# 3) Docker Compose로 API 기동
cd v2/serving
docker compose up -d --build

# 4) curl 예시 (아래 참조)

# 5) 종료
docker compose down
```

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
| GET /health | 모델 버전·메타데이터(test_mdape 등) |
| POST /valuation | 시세(만원), ㎡당 가격, knn_ppm2_5, same_bldg |
| POST /recovery | `compute_recovery_v2` 결과(S,K,P,D2,R,PV,EAD,LGD) |

rho 시나리오 값(EXP-010, 일괄매각 의심 제외 후): 비관 0.5738 / 기본 0.644620 / 낙관 0.8002.

## 재현·검증 결과 (EXP-011)

| 확인 항목 | 결과 |
|---|---|
| artifact 재예측 MdAPE | 11.5635%(EXP-008 등록값과 차이 0.000000%p, 허용 0.01%p) |
| pytest(`v2/serving/tests`) | 8 passed |
| docker compose up -d --build → curl /health, /valuation, /recovery | 모두 200 응답 |
| 효창동 예시 /recovery LGD | 58.6010%(기준 58.6%, ±0.1%p) |
| 범위 밖 입력(excluUseAr=0) /valuation | 422 |

## 알아둘 점

- `v2/price_model.py`가 내부적으로 `src/knn_comps.py`(faiss)를 import하는 구조라, `requirements.txt`에 `faiss-cpu`를 추가했다(EXP-008 패키지 목록에는 없던 의존성).
- `python:3.12-slim` 이미지에는 lightgbm이 필요로 하는 `libgomp.so.1`이 없다. apt 설치 없이 faiss-cpu 휠이 이미 들고 있는 libgomp 사본에 심볼릭 링크를 만들어 해결했다(`Dockerfile` 참고).
- 모델 파일(`artifact/`)과 데이터(`data/`)는 이미지에 복사하지 않고 Compose volume(`./artifact:/app/artifact:ro`)으로 읽기 전용 연결한다.
- 지오코딩은 이번 범위 밖이다 — `/valuation`은 호출자가 직접 lat/lng을 줘야 한다.
