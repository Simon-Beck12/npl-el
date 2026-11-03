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

    return train, test


if __name__ == "__main__":
    main()
