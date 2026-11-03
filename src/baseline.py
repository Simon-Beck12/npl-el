from pathlib import Path

import numpy as np
import pandas as pd

from split import apply_common_filter, add_contract_date, split_dv02, small_area_threshold

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PARQUET = ROOT / "data" / "processed" / "rh_trade_clean.parquet"
LAWD_CSV = ROOT / "docs" / "seoul_lawd_cd.csv"
PRED_PARQUET = ROOT / "data" / "processed" / "exp002_predictions.parquet"

AGE_BINS = [-np.inf, 2, 10, 20, np.inf]
AGE_LABELS = ["신축(0~2년)", "3~10년", "11~20년", "21년 이상"]
AREA_BINS = [0, 30, 45, 60, np.inf]
AREA_LABELS = ["~30㎡", "30~45㎡", "45~60㎡", "60㎡~"]
MIN_GROUP_COUNT = 5


def build_features(df, lawd_map):
    df = df.copy()
    df["gu_name"] = df["sggCd"].astype(str).map(lawd_map)
    age_years = df["dealYear"].astype(int) - df["buildYear"].astype(int)
    df["age_bucket"] = pd.cut(age_years, bins=AGE_BINS, labels=AGE_LABELS)
    df["area_bucket"] = pd.cut(df["excluUseAr"], bins=AREA_BINS, labels=AREA_LABELS)
    df["dong_code"] = df["sggCd"].astype(str) + "_" + df["umdNm"].str.strip()
    df["cell_id"] = df["dong_code"] + "|" + df["age_bucket"].astype(str) + "|" + df["area_bucket"].astype(str)
    df["is_new"] = age_years <= 2
    return df


def fit_baseline_tables(train):
    dong_median = train.groupby("dong_code")["price_per_m2"].median()
    dong_count = train.groupby("dong_code")["price_per_m2"].count()
    gu_median = train.groupby("gu_name")["price_per_m2"].median()
    cell_median = train.groupby("cell_id")["price_per_m2"].median()
    cell_count = train.groupby("cell_id")["price_per_m2"].count()
    return dong_median, dong_count, gu_median, cell_median, cell_count


def predict_row(row, tables):
    dong_median, dong_count, gu_median, cell_median, cell_count = tables

    dcnt = dong_count.get(row["dong_code"], 0)
    if dcnt >= MIN_GROUP_COUNT:
        b0_pps = dong_median[row["dong_code"]]
        b0_stage = "법정동"
    else:
        b0_pps = gu_median.get(row["gu_name"], np.nan)
        b0_stage = "구 대체"

    ccnt = cell_count.get(row["cell_id"], 0)
    if ccnt >= MIN_GROUP_COUNT:
        b1_pps = cell_median[row["cell_id"]]
        b1_stage = "셀"
    else:
        b1_pps = b0_pps
        b1_stage = "B0 대체"

    return pd.Series({
        "b0_pred": b0_pps * row["excluUseAr"],
        "b0_stage": b0_stage,
        "b1_pred": b1_pps * row["excluUseAr"],
        "b1_stage": b1_stage,
    })


def run_folds(df, lawd_map, small_thr):
    folds = split_dv02(df)
    pred_rows = []
    for name, train, test in folds:
        train_f = build_features(train, lawd_map)
        test_f = build_features(test, lawd_map)
        tables = fit_baseline_tables(train_f)

        preds = test_f.apply(lambda r: predict_row(r, tables), axis=1)
        out = pd.concat([test_f[[
            "deal_id", "dealAmount", "gu_name", "dong_code", "age_bucket", "area_bucket",
            "is_new", "basement_flag", "excluUseAr",
        ]].reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        out["fold"] = name
        out["is_basement"] = out["basement_flag"] == 1
        out["is_small"] = out["excluUseAr"] < small_thr
        out = out.rename(columns={"dealAmount": "actual_price"})
        pred_rows.append(out)

    predictions = pd.concat(pred_rows, ignore_index=True)
    return predictions


def main():
    df = pd.read_parquet(CLEAN_PARQUET)
    lawd_map = pd.read_csv(LAWD_CSV, dtype={"LAWD_CD": str}).set_index("LAWD_CD")["gu_name"]
    small_thr = small_area_threshold(df)

    predictions = run_folds(df, lawd_map, small_thr)

    predictions["b0_ape"] = (predictions["actual_price"] - predictions["b0_pred"]).abs() / predictions["actual_price"]
    predictions["b1_ape"] = (predictions["actual_price"] - predictions["b1_pred"]).abs() / predictions["actual_price"]

    PRED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PRED_PARQUET, index=False)

    print(f"예측 결과 저장: {PRED_PARQUET} ({len(predictions):,}행)")
    print(f"결측(NaN) 예측 — B0: {predictions['b0_pred'].isna().sum()}, B1: {predictions['b1_pred'].isna().sum()}")

    def metrics(ape):
        return {
            "MdAPE": ape.median() * 100,
            "MAPE": ape.mean() * 100,
            "within10": (ape <= 0.10).mean() * 100,
            "within20": (ape <= 0.20).mean() * 100,
            "n": len(ape),
        }

    print("\n=== pooled 전체 ===")
    print("B0:", metrics(predictions["b0_ape"]))
    print("B1:", metrics(predictions["b1_ape"]))

    print("\n=== fold별 ===")
    for name, g in predictions.groupby("fold"):
        print(f"{name} B0:", metrics(g["b0_ape"]))
        print(f"{name} B1:", metrics(g["b1_ape"]))

    return predictions


if __name__ == "__main__":
    main()
