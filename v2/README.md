# npl-el-pipeline v2

LightGBM 가격 추정과 조건을 줄인 회수액·LGD 계산을 하나로 묶은 두 번째 파이프라인이다. 기존 v1(`src/`, `results/`, `README.md`)은 바꾸지 않는다.

## 3단계 구조

```
① 가격 추정(EXP-008)        data/processed/rh_trade_clean.parquet + geocode_cache.csv
   LightGBM(G2, 8변수) vs B0 vs A(EXP-004 FAISS)
        ↓
② 포트폴리오 재추정(EXP-009) SYN-008 포트폴리오 20건에 ①의 채택 모델을 "최초 개찰일 연월 이전 거래"로
   다시 학습해 적용 → v2 시세·낙찰가율(전역 분위수) = SYN-009
        ↓
③ 회수액·LGD(EXP-009)       6줄 계산(원금까지만 회수) → v2/results/npl_v2_review.xlsx
```

## 변수 8개 (EXP-008 가격 모델)

| 변수 | 설명 |
|---|---|
| lat, lng | 위도·경도(원값, km 평면 변환 안 함) |
| 전용면적 | ㎡ |
| 연식 | 계약연도(또는 최초 개찰일 연도) − 건축년도 |
| 층 | floor |
| 계약 연월 | 2024-11=0부터 정수 |
| knn_ppm2_5 | 학습 데이터 중 이웃 5건 ㎡당 가격 중앙값(좌표 거리, 자기 자신 제외) |
| same_bldg | 같은 구·동·지번 학습 거래가 있으면 1 |

B0·A(EXP-004 FAISS)는 6개·8개 변수를 쓰지 않는 별도 비교 모델이다(`v2/results/exp008_price.md` 참조).

## 결과 숫자 3개

| 항목 | 값 |
|---|---|
| EXP-008 테스트 MdAPE | B0 24.85% / A(EXP-004) 13.43% / G1 12.21% / **G2(채택) 11.56%** |
| v2 포트폴리오 LGD(18건, rho=기본·보증금=기본·24개월) | **87.5%** (n=18, 20건 중 2건은 v2 시세 추정 실패로 제외) |
| v1 대비 차이 30.1%p의 분해 | 물건 구성 차이 +21.4%p(동대문 대형 물건 제외 효과) + 계산 방식 차이 +8.7%p(전역 rho·원금 한정 회수) |

숫자는 `v2/results/exp008_price.md`, `v2/results/exp009_recovery.md`에서 그대로 옮겼다.

## 단순화 가정 (v1과 다른 점)

- 회수액은 대출원금(EAD)까지만 본다 — 연체이자·청구금액·채권최고액(120%)·당해세를 쓰지 않는다.
- 낙찰가율은 물건별이 아니라 전역 분위수(학습 대상 83건의 25/50/75%)를 비관/기본/낙관으로 쓴다.
- 보증금 "깡통" = v2 시세 × 90%(가상값, v1과 동일 가정).
- v2 필터에는 공공기관 거래 제외가 없다(설계 누락, 테스트 건수가 v1보다 4건 많다).
- 포트폴리오 20건 중 2건(최초 개찰일이 2024-11)은 그 이전 거래가 0건이라 v2 시세를 추정할 수 없어 18건만 계산한다.

## 결과 파일

- `v2/results/exp008_price.md` — 가격 모델 비교(표1~3)
- `v2/results/exp009_recovery.md` — 회수액·LGD(표1~3, v1 비교 포함)
- `v2/results/npl_v2_review.xlsx` — 가정/검토표/요약 3시트, 전부 수식
- `v2/tests/test_recovery_v2.py` — 6줄 계산 단위 테스트 4건(4 passed)

## 재현 방법

```bash
uv run python v2/price_model.py   # EXP-008
uv run pytest v2/tests/           # 단위 테스트 4건
uv run python v2/recovery.py      # EXP-009
uv run python v2/build_excel.py   # npl_v2_review.xlsx
```

전체 실험 기록은 `docs/EXPERIMENTS.md`(EXP-008·EXP-009), 데이터셋 명세는 `docs/DATASETS.md`(SYN-009)를 참조한다.
