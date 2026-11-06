# npl-el v2 서빙 (FastAPI + Docker)

EXP-008 채택 모델(G2, LightGBM 8변수)과 EXP-009 회수액 6줄 계산을 HTTP API로 제공한다.
기존 v2 코드는 수정하지 않는다. 이 폴더에 새 파일만 둔다.

## 구조

```
export_artifacts.py   EXP-008 G2를 같은 설정으로 다시 학습 → artifacts/ 저장 + 재현 확인 3가지
predictor.py          artifact 로드, 피처 8개 계산, 시세·회수액 예측
recovery_core.py      v2/recovery.py compute_recovery_v2의 서빙용 사본(동일성은 테스트로 확인)
app.py                FastAPI: GET /health, POST /v1/price, POST /v1/recovery
Dockerfile            python:3.12-slim + libgomp1, artifact를 이미지에 포함
tests/                가짜 artifact로 도는 테스트 + 실제 artifact가 있으면 golden 비교
```

## 실행 순서 (저장소 루트에서)

```bash
uv run python v2/serving/export_artifacts.py      # 1. artifact 생성 + 재현 확인 (멈추면 원인 보고)
uv run pytest v2/serving/tests -v                 # 2. test_golden_real 포함 전부 통과해야 한다
cd v2/serving
docker build -t npl-el-api .                      # 3. 이미지 빌드
docker run --rm -p 8000:8000 npl-el-api           # 4. 실행 → http://localhost:8000/docs
```

Docker 없이 실행: `cd v2/serving && uv run uvicorn app:app --port 8000`
(uv 환경에 `fastapi`, `uvicorn`이 없으면 `uv pip install fastapi==0.162.0 uvicorn==0.54.0`)

## 재현 확인 3가지 (export_artifacts.py, 하나라도 실패하면 멈춤)

| 확인 | 기준 |
|---|---|
| 재학습 모델 테스트 MdAPE vs EXP-008 기록(11.56%) | 차이 0.01%p 이내 |
| 저장한 model.txt 재로드 예측 vs 메모리 모델 예측 | 상대오차 1e-9 이내 |
| 낙찰가율 시나리오 vs EXP-009 표2(0.5741/0.6446/0.8042) | 차이 5e-5 이내 |

그 뒤 `tests/test_golden.py::test_golden_real`이 API용 피처 코드(predictor.py)와
학습 코드(price_model.build_lgbm_frame) 예측이 테스트 거래 200건에서 같은지(상대오차 1e-6 이내) 확인한다.

## 요청 예시

```bash
curl -X POST localhost:8000/v1/recovery -H 'Content-Type: application/json' -d '{
  "lat": 37.556, "lng": 126.91, "area_m2": 45, "build_year": 2005, "floor": 3,
  "gu": "마포구", "dong": "망원동", "jibun": "123-4", "as_of": "2026-05",
  "deposit_won": 80000000, "ead_won": 150000000, "months": [12, 24, 36]
}'
```

응답: 예측 시세(원), 낙찰가율 3개 × 개월 수별 S·K·P·D2·R·PV·LGD, 근거 이웃 거래 5건(거리·㎡당 가격), 경고(flags).

## 경고(flags) 규칙

- 기준월이 학습 마지막 월(2026-05) 이후: LightGBM은 시간 추세를 외삽하지 않는다.
- 가장 가까운 학습 거래가 1 km 이상 떨어짐.
- 전용면적이 학습 데이터 1~99% 범위 밖.
- 같은 건물(구·동·지번) 학습 거래 없음: EXP-008 표2에서 이 그룹 MdAPE가 더 컸다.
- EAD 미입력: 시세 × 0.70으로 가정(v2와 같은 가정).

## 한계

- 학습 기간은 2024-11~2026-05로 EXP-008과 같다. 운영용으로는 최신 거래까지 다시 학습해야 한다.
- 낙찰가율은 물건별이 아니라 전역 분위수 3개다(EXP-009 단순화 가정 그대로).
- 입력 좌표는 호출하는 쪽이 지오코딩해서 넣는다. 지번 표기는 국토부 실거래가 형식과 같아야 same_bldg가 맞는다.
- 인증·요청 제한·모니터링은 없다(포트폴리오 범위).
