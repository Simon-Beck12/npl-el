from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PARQUET = ROOT / "data" / "processed" / "rh_trade_clean.parquet"

TRAIN_START = "2024-11-01"
TRAIN_END = "2026-05-31"
TEST_START = "2026-06-01"
TEST_END = "2026-08-31"


def apply_common_filter(df):
    base = df[(df["direct_flag"] == 0) & (df["outlier_flag"] == 0)].copy()
    is_public = (base["buyerGbn"] == "공공기관") | (base["slerGbn"] == "공공기관")
    return base[~is_public].copy()


def add_contract_date(df):
    df = df.copy()
    df["contract_date"] = pd.to_datetime(
        df["dealYear"].astype(str) + "-" + df["dealMonth"].astype(str).str.zfill(2) + "-" + df["dealDay"].astype(str).str.zfill(2)
    )
    return df


def split_train_test(df):
    df = add_contract_date(apply_common_filter(df))
    train = df[(df["contract_date"] >= TRAIN_START) & (df["contract_date"] <= TRAIN_END)].copy()
    test = df[(df["contract_date"] >= TEST_START) & (df["contract_date"] <= TEST_END)].copy()
    return train, test


# DV-02: 확장 창(expanding window) 3-fold. 공통 필터는 DV-01과 동일.
DV02_FOLDS = [
    ("F1", "2024-11-01", "2025-11-30", "2025-12-01", "2026-02-28"),
    ("F2", "2024-11-01", "2026-02-28", "2026-03-01", "2026-05-31"),
    ("F3", "2024-11-01", "2026-05-31", "2026-06-01", "2026-08-31"),
]
SMALL_AREA_QUANTILE = 0.25


def small_area_threshold(df):
    filtered = add_contract_date(apply_common_filter(df))
    return filtered["excluUseAr"].quantile(SMALL_AREA_QUANTILE)


def is_new_mask(df):
    age_years = df["dealYear"].astype(int) - df["buildYear"].astype(int)
    return age_years <= 2


def split_dv02(df):
    """fold마다 (fold명, train, test) 튜플을 반환한다. train은 그 fold의 학습 구간만 포함한다."""
    filtered = add_contract_date(apply_common_filter(df))
    folds = []
    for name, tr_s, tr_e, te_s, te_e in DV02_FOLDS:
        train = filtered[(filtered["contract_date"] >= tr_s) & (filtered["contract_date"] <= tr_e)].copy()
        test = filtered[(filtered["contract_date"] >= te_s) & (filtered["contract_date"] <= te_e)].copy()
        folds.append((name, train, test))
    return folds


def main():
    df = pd.read_parquet(CLEAN_PARQUET)
    print(f"원본(SYN-001): {len(df):,}건")

    filtered = apply_common_filter(df)
    print(f"공통 필터(direct_flag=0, outlier_flag=0, 공공기관 제외) 후: {len(filtered):,}건")

    train, test = split_train_test(df)
    print(f"학습 구간({TRAIN_START}~{TRAIN_END}): {len(train):,}건")
    print(f"테스트 구간({TEST_START}~{TEST_END}): {len(test):,}건")

    excluded_tail = len(filtered) - len(train) - len(test) - len(
        filtered[add_contract_date(filtered)["contract_date"] < TRAIN_START]
    )
    all_dated = add_contract_date(filtered)
    before_train = all_dated[all_dated["contract_date"] < TRAIN_START]
    after_test = all_dated[all_dated["contract_date"] > TEST_END]
    print(f"학습 구간 이전(2024-11-01 미만): {len(before_train):,}건")
    print(f"테스트 구간 이후(2026-09 이후 등): {len(after_test):,}건")

    new_age_years = test["dealYear"].astype(int) - test["buildYear"].astype(int)
    new_count = int((new_age_years <= 2).sum())
    print(f"테스트 구간 신축(거래연도-건축년도<=2) 건수: {new_count:,}")

    print("\n=== DV-02 (확장 창 3-fold) ===")
    small_thr = small_area_threshold(df)
    print(f"소형 기준(전용면적 하위 {SMALL_AREA_QUANTILE:.0%}, ㎡): {small_thr:.2f}")

    folds = split_dv02(df)
    pooled_test_n = 0
    pooled_new_n = 0
    for name, ftrain, ftest in folds:
        new_n = int(is_new_mask(ftest).sum())
        basement_n = int((ftest["basement_flag"] == 1).sum())
        small_n = int((ftest["excluUseAr"] < small_thr).sum())
        print(f"{name}: train={len(ftrain):,}, test={len(ftest):,}, 신축={new_n}, 지하={basement_n}, 소형={small_n}")
        pooled_test_n += len(ftest)
        pooled_new_n += new_n

    print(f"pooled 테스트: {pooled_test_n:,}, pooled 신축: {pooled_new_n:,}")
    if pooled_new_n < 100:
        print("[멈춤 지점] pooled 테스트 신축이 100건 미만입니다. 중단하고 보고해야 합니다.")
    else:
        print("pooled 테스트 신축 100건 이상 — 멈춤 지점 미해당.")

    return train, test


if __name__ == "__main__":
    main()
