"""EXP-006 단계 11: 공매 물건별 선순위 보증금 비율 계산.

보증금 비율 = 같은(구,동,지번)의 전월세 계약 중 계약일이 해당 물건의 최초 개찰일보다 이른 계약의
보증금 중앙값(만원→원) ÷ 추정 시세(원). 같은 지번 계약이 없으면 결측. 최초 개찰일 이후 계약은
절대 쓰지 않는다(누수 금지).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RENT_PARQUET = ROOT / "data" / "processed" / "rh_rent_clean.parquet"
FEATURES_V3_PARQUET = ROOT / "data" / "processed" / "onbid_features_v3.parquet"
LAWD_CSV = ROOT / "docs" / "seoul_lawd_cd.csv"
OUT_PARQUET = ROOT / "data" / "processed" / "onbid_deposit_ratio.parquet"


def load_rent_with_key():
    rent = pd.read_parquet(RENT_PARQUET).copy()
    lawd_map = pd.read_csv(LAWD_CSV, dtype={"LAWD_CD": str}).set_index("LAWD_CD")["gu_name"]
    rent["gu_name"] = rent["sggCd"].astype(str).map(lawd_map)
    rent["key"] = rent["gu_name"] + "|" + rent["umdNm"].str.strip() + "|" + rent["jibun"].str.strip()
    return rent


def compute_deposit_ratio(onbid, rent):
    onbid = onbid.copy()
    onbid["key"] = onbid["구"] + "|" + onbid["동"].str.strip() + "|" + onbid["지번"].str.strip()
    onbid["cutoff"] = pd.to_datetime(onbid["최초 개찰일"], format="%Y%m%d%H%M")

    rent_by_key = {k: g for k, g in rent.groupby("key")}

    deposit_med_won = []
    n_rent_matched = []
    for key, cutoff in zip(onbid["key"], onbid["cutoff"]):
        g = rent_by_key.get(key)
        if g is None:
            deposit_med_won.append(float("nan"))
            n_rent_matched.append(0)
            continue
        prior = g[g["contract_date"] < cutoff]
        if len(prior) == 0:
            deposit_med_won.append(float("nan"))
            n_rent_matched.append(0)
            continue
        med_deposit_man = prior["deposit"].median()
        deposit_med_won.append(med_deposit_man * 10000)
        n_rent_matched.append(len(prior))

    onbid["deposit_median_won"] = deposit_med_won
    onbid["n_rent_matched"] = n_rent_matched
    onbid["deposit_ratio"] = onbid["deposit_median_won"] / onbid["market_price"]
    return onbid


def main():
    rent = load_rent_with_key()
    onbid = pd.read_parquet(FEATURES_V3_PARQUET)

    out = compute_deposit_ratio(onbid, rent)

    n_calc = int(out["deposit_ratio"].notna().sum())
    print(f"보증금 비율 계산 가능: {n_calc}/{len(out)} ({n_calc/len(out):.1%})")

    out.to_parquet(OUT_PARQUET, index=False)
    print(f"저장: {OUT_PARQUET} ({len(out)}행)")
    return out


if __name__ == "__main__":
    main()
