# EXP-014 성능 비교 (k-NN 피처 벡터화)

측정일 2026-11-07. 기존 경로 = 커밋 `87810cf`(EXP-014 변경 전, `git worktree`로 별도 폴더에 꺼냄).
새 경로 = 이 커밋(`fast_features.KnnIndex`로 cKDTree·same_bldg 집계를 1회만 생성). 같은 artifact
(`v2/serving/artifact/`: model.txt·ref.parquet·meta.json, EXP-008 G2) 사용.

## 1. API 단건 응답 시간 (정정, 실측)

`bench.py --api-only --app-dir <경로>`, TestClient, 워밍업 50회 제외 1,000회, OMP_NUM_THREADS=1.

| 경로 | p50 (ms) | p95 (ms) | 평균 (ms) | 결과 파일 |
|---|---|---|---|---|
| 기존(87810cf, worktree) | 30.337 | 33.096 | 30.472 | `bench_before_api_fixed.json` |
| 새(이 커밋) | 5.697 | 6.232 | 5.710 | `bench_after_api_fixed.json` |
| 배수(p50) | 5.32배 감소(−81.2%) | 5.31배 | 5.34배 | - |

채택 기준 4(API 단건 p50 감소) **통과**(정정값 기준). `bench_before.json`/`bench_after.json`의 `api` 항목은
무효(아래 2절 버그 참고).

## 2. bench.py 버그와 수정

기존 `bench_api()`는 `--path` 인자와 무관하게 항상 현재 작업 폴더의 `app.py`만 import해서 쟀다.
`app.py`는 EXP-014 6~9단계에서 이미 새 경로로 영구 수정된 상태였으므로, `bench_before.json`·
`bench_after.json`의 `api` 필드는 **둘 다 새 경로를 잰 같은 값**이었다(최초 보고한 "5.923→5.719ms,
1.04배"는 무효). 배치 측정(`bench_valuation_batch`)은 old/new 함수를 인자로 직접 분기해 호출하므로
이 버그의 영향이 없다.

수정: `bench_api(app_dir=None)`로 바꿔, `app_dir`을 주면 그 경로의 `v2/serving`·`v2`·`src`를
`sys.path` 맨 앞에 넣고 `app.py`를 import한다(`v2/serving/bench.py`). `git worktree add /tmp/npl-el-old
87810cf`로 변경 전 커밋을 별도 폴더에 꺼내고, `MODEL_DIR` 환경변수를 현재 저장소의 실제
`artifact/`로 지정해(`artifact/`는 gitignore 대상이라 worktree에는 없음) 같은 모델로 실측했다.
측정 뒤 `git worktree remove /tmp/npl-el-old`로 정리했다.

## 3. 배치(시세 추정) 비교

| 측정 항목 | 기존 경로 | 새 경로 | 배수 |
|---|---|---|---|
| 7,561건 전체 시간 | 190.121초 | 0.773초 | 246.0배 |
| 행당 평균 | 25.145 ms | 0.102 ms | 246.0배 |
| 초당 처리 행 수 | 39.8 행/초 | 9,782.9 행/초 | - |

결과 파일: `bench_before.json`(batch_valuation 필드만 유효), `bench_after.json`.

## 4. API 1건 처리 시간 구간별 분해 (고정 주소, p50, ms)

`model_loader.load_model`로 모델을 1번 불러온 뒤, 같은 프로세스에서 구간 a~d를 1,000회씩
독립적으로 측정(워밍업 50회 제외). old/new 함수는 둘 다 `price_model.py`에 남아있는 원본
함수(`add_knn_ppm2`/`add_same_bldg`, 수정 안 함)와 새 `KnnIndex` 메서드를 직접 호출해 비교했다.

| 구간 | 기존 (ms) | 새 (ms) |
|---|---|---|
| a. 입력 검증·DataFrame 생성 | 0.240 | 0.240 (동일 코드) |
| b. k-NN 피처(cKDTree) | 9.067 | 0.308 |
| c. same_bldg 판별 | 13.872 | 1.139 |
| d. LightGBM 예측 | 0.849 | 0.849 (동일 코드) |
| 함수 합(a+b+c+d) | 24.028 | 2.536 |
| e. FastAPI/pydantic 등 나머지(TestClient 전체 − 함수 합) | 3.834(추정) | 3.834 |
| 함수 합 + e (≈ API 전체) | 27.862(추정) | 6.370(실측 TestClient) |

다양 주소(테스트셋 랜덤 샘플 1,050건)로도 재측정: a=0.239, b_old=9.035, c_old=13.660, b_new=0.304,
c_new=1.154, d=0.875 — 고정 주소와 차이 0.2ms 이내. **주소 차이(시연용 고정 주소 vs 테스트셋 실제
주소)는 배치·API 격차의 원인이 아니다.**

### 배치-구간합(~25.1ms)이 애초 보고한 API(~5.9ms)보다 컸던 이유

1차 보고의 "API p50 5.9ms"는 2절의 bench.py 버그로 실제로는 **새 경로**를 잰 값이었다(함수 합
2.5ms + FastAPI 오버헤드와 맞물림). 올바른 비교는 기존 경로끼리다: 배치 행당(25.145ms, FastAPI
없음)과 기존 경로 함수 합(24.028ms, 역시 FastAPI 없음)이 거의 같고, 여기에 FastAPI/pydantic
오버헤드(e≈3.8ms)를 더한 추정 API 전체(27.862ms)는 실측 기존 API(30.337ms)와 8~9% 차이로
비슷한 크기다(완전히 같지 않은 차이는 별도 프로세스(worktree)·측정 시점이 달라 생기는 노이즈로
본다, 20% 기준 이내). 즉 **역설이 아니었다** — 배치든 API든 공통으로 b·c 구간(cKDTree
재생성+same_bldg 재집계, 약 23ms)이 지배적이었고, 새 경로는 이 23ms를 2.5ms로 줄였다.

### 이전 커밋 코드로 재검증 (구간 20% 기준)

`git worktree add /tmp/npl-el-old 87810cf`로 꺼낸 **리터럴 옛 코드**로 구간 a~d를 다시 쟀다
(고정 주소, 1,000회).

| 구간 | 현재 저장소에서 옛 함수 호출 (ms) | 옛 커밋 코드 그대로(worktree) (ms) | 차이 |
|---|---|---|---|
| a | 0.240 | 0.229 | 4.60% |
| b | 9.067 | 9.084 | 0.18% |
| c | 13.872 | 13.996 | 0.89% |
| d | 0.849 | 0.885 | 4.03% |

모든 구간 차이 20% 미만 — 멈춤 지점 해당 없음, 위 1차 분해 값을 그대로 쓴다.
결과 파일: `exp014_stage_breakdown_raw.txt`, `exp014_stage_breakdown_old_worktree_raw.txt`.

## 측정 조건

- CPU: AMD Ryzen 9 5950X 16-Core Processor(32논리코어), 메모리 125.7GB.
- Python 3.12.3, LightGBM 4.7.0, SciPy 1.18.1.
- `OMP_NUM_THREADS=1`로 모든 측정 실행. 적용 확인: 45,634행 `booster.predict()` 20회 반복 시간이
  1/4/32 스레드에서 각각 5452.40 / 1440.10 / 312.54 ms/회로 거의 선형으로 줄어, LightGBM이 이
  환경변수를 실제로 읽어 썼음을 확인했다(코드에 `num_threads` 파라미터를 명시하지 않음).
- API: TestClient, 워밍업 50회 제외 1,000회 호출. 배치: 테스트셋 7,561건 전체, 1회.
