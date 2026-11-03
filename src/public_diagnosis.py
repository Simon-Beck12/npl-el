from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PARQUET = ROOT / "data" / "processed" / "rh_trade_clean.parquet"
LAWD_CSV = ROOT / "docs" / "seoul_lawd_cd.csv"

AGE_BINS = [-np.inf, 2, 10, 20, np.inf]
AGE_LABELS = ["신축(0~2년)", "3~10년", "11~20년", "21년 이상"]
AREA_BINS = [0, 30, 45, 60, np.inf]
AREA_LABELS = ["~30㎡", "30~45㎡", "45~60㎡", "60㎡~"]
MIN_CELL_COUNT = 5


def build_cells(df):
    df = df.copy()
    lawd = pd.read_csv(LAWD_CSV, dtype={"LAWD_CD": str}).set_index("LAWD_CD")["gu_name"]
    df["gu_name"] = df["sggCd"].astype(str).map(lawd)
    df["age_years"] = df["dealYear"].astype(int) - df["buildYear"].astype(int)
    df["age_bucket"] = pd.cut(df["age_years"], bins=AGE_BINS, labels=AGE_LABELS)
    df["area_bucket"] = pd.cut(df["excluUseAr"], bins=AREA_BINS, labels=AREA_LABELS)
    df["dong_code"] = df["sggCd"].astype(str) + "_" + df["umdNm"].str.strip()
    df["cell_id"] = df["dong_code"] + "|" + df["age_bucket"].astype(str) + "|" + df["area_bucket"].astype(str)
    return df


def main():
    df = pd.read_parquet(CLEAN_PARQUET)
    base = df[(df["direct_flag"] == 0) & (df["outlier_flag"] == 0)].copy()
    print(f"direct_flag=0, outlier_flag=0 필터 후: {len(base):,}건")

    is_public_buyer = base["buyerGbn"] == "공공기관"
    is_public_seller = base["slerGbn"] == "공공기관"
    is_public = is_public_buyer | is_public_seller

    print(f"공공기관 매수: {int(is_public_buyer.sum()):,}건")
    print(f"공공기관 매도: {int(is_public_seller.sum()):,}건")
    print(f"양쪽(매수+매도 중복 포함 OR): {int(is_public.sum()):,}건 ({is_public.mean():.1%})")
    both = (is_public_buyer & is_public_seller).sum()
    print(f"매수·매도 모두 공공기관: {int(both):,}건")

    base = build_cells(base)
    public_tx = base[is_public].copy()
    general_tx = base[~is_public].copy()

    new_ratio = (public_tx["age_bucket"] == "신축(0~2년)").mean()
    print(f"\n공공기관 거래 중 신축 비율: {new_ratio:.1%} ({int((public_tx['age_bucket']=='신축(0~2년)').sum())}/{len(public_tx)})")

    cell_median = general_tx.groupby("cell_id")["price_per_m2"].median()
    cell_count = general_tx.groupby("cell_id")["price_per_m2"].count()
    valid_cells = cell_count[cell_count >= MIN_CELL_COUNT].index

    public_valid = public_tx[public_tx["cell_id"].isin(valid_cells)].copy()
    public_valid["cell_median"] = public_valid["cell_id"].map(cell_median)
    public_valid["rel_price"] = public_valid["price_per_m2"] / public_valid["cell_median"]

    general_valid = general_tx[general_tx["cell_id"].isin(valid_cells)].copy()
    general_valid["cell_median"] = general_valid["cell_id"].map(cell_median)
    general_valid["rel_price"] = general_valid["price_per_m2"] / general_valid["cell_median"]

    print(f"\n유효 셀(일반 중개거래 5건 이상)에 포함된 공공기관 거래: {len(public_valid)}/{len(public_tx)}")

    pub_median = public_valid["rel_price"].median()
    pub_iqr = public_valid["rel_price"].quantile(.75) - public_valid["rel_price"].quantile(.25)
    gen_median = general_valid["rel_price"].median()
    gen_iqr = general_valid["rel_price"].quantile(.75) - general_valid["rel_price"].quantile(.25)

    print(f"공공기관 상대가격 중앙값: {pub_median:.3f}, IQR: {pub_iqr:.3f} (n={len(public_valid)})")
    print(f"일반 중개거래 상대가격 중앙값: {gen_median:.3f}, IQR: {gen_iqr:.3f} (n={len(general_valid)})")

    result = {
        "base_n": len(base),
        "public_buyer_n": int(is_public_buyer.sum()),
        "public_seller_n": int(is_public_seller.sum()),
        "public_either_n": int(is_public.sum()),
        "public_either_ratio": float(is_public.mean()),
        "public_both_n": int(both),
        "public_new_ratio": float(new_ratio),
        "public_valid_n": len(public_valid),
        "public_rel_median": float(pub_median),
        "public_rel_iqr": float(pub_iqr),
        "general_rel_median": float(gen_median),
        "general_rel_iqr": float(gen_iqr),
    }
    return result


if __name__ == "__main__":
    main()
