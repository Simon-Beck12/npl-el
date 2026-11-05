"""EXP-010: 일괄매각 의심 면적 필터(건물면적>200㎡) + v1·v2 LGD 재계산.

기존 src/, v2/의 .py 파일은 수정하지 않는다. v2/recovery.py의 함수를 그대로 재사용한다.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))

from recovery import load_rh_base, estimate_v2_price, compute_recovery_v2, MODEL_CHOICE_JSON  # noqa: E402

AREA_THRESHOLD = 200.0
RESULTS_DIR = ROOT / "results"
ONBID_PARQUET = ROOT / "data" / "processed" / "onbid_features_v3.parquet"
PORTFOLIO_PARQUET = ROOT / "data" / "processed" / "portfolio_v1.parquet"
EXP007_GRID = ROOT / "data" / "processed" / "exp007_recovery.parquet"
PORTFOLIO_V2_PARQUET = ROOT / "v2" / "data" / "processed" / "portfolio_v2.parquet"


def find_big_items():
    onbid = pd.read_parquet(ONBID_PARQUET)
    big_all = onbid[onbid["건물면적"] > AREA_THRESHOLD].copy()

    target99 = onbid[(onbid["최종 결과"] == "낙찰") & (onbid["bad_lowest"] == 0)
                      & (onbid["bad_ratio"] == 0) & (onbid["bad_bid"] == 0)].copy()
    big_target = target99[target99["건물면적"] > AREA_THRESHOLD].copy()

    port = pd.read_parquet(PORTFOLIO_PARQUET)
    big_port = port[port["건물면적"] > AREA_THRESHOLD].copy()

    return big_all, target99, big_target, port, big_port


def recompute_v1(big_port_ids):
    grid = pd.read_parquet(EXP007_GRID)
    sub = grid[(grid["경로"] == "시세") & (~grid["cltrMngNo"].isin(big_port_ids))]
    rows = []
    for dep in ["없음", "기본", "깡통"]:
        for scen in ["낙관", "기본", "비관"]:
            s = sub[(sub["보증금시나리오"] == dep) & (sub["매각시점시나리오"] == scen)]
            lgd = 1 - s["PV"].sum() / s["EAD"].sum()
            rows.append({"보증금시나리오": dep, "매각시점시나리오": scen, "LGD": lgd, "n": s["cltrMngNo"].nunique()})
    return pd.DataFrame(rows)


def recompute_v2(big_target_ids):
    with open(MODEL_CHOICE_JSON, encoding="utf-8") as f:
        model_choice = json.load(f)
    rh_base = load_rh_base()
    onbid = pd.read_parquet(ONBID_PARQUET)
    target99 = onbid[(onbid["최종 결과"] == "낙찰") & (onbid["bad_lowest"] == 0)
                      & (onbid["bad_ratio"] == 0) & (onbid["bad_bid"] == 0)].copy()
    port = pd.read_parquet(PORTFOLIO_PARQUET)
    port_sold_ids = set(port.loc[port["결과구분"] == "낙찰", "cltrMngNo"])
    rho_pool = target99[~target99["cltrMngNo"].isin(port_sold_ids)].copy()
    rho_pool_before_n = len(rho_pool)
    rho_pool = rho_pool[~rho_pool["cltrMngNo"].isin(big_target_ids)].copy()
    rho_pool_after_n = len(rho_pool)

    model_cache = {}
    rho_rows = []
    for _, r in rho_pool.iterrows():
        pred_price, _ = estimate_v2_price(r, rh_base, model_choice, model_cache)
        if pred_price is None or pred_price <= 0:
            continue
        rho_rows.append({"cltrMngNo": r["cltrMngNo"], "rho": r["낙찰가"] / pred_price})
    rho_df = pd.DataFrame(rho_rows)
    q25, q50, q75 = rho_df["rho"].quantile([0.25, 0.5, 0.75])
    rho_scenarios = {"비관": float(q25), "기본": float(q50), "낙관": float(q75)}

    v2port = pd.read_parquet(PORTFOLIO_V2_PARQUET)  # 이미 18건(971㎡ 물건 포함 2건 제외됨)
    rows = []
    for _, r in v2port.iterrows():
        for name, rho in rho_scenarios.items():
            res = compute_recovery_v2(r["v2_price"], rho, r["보증금_기본_v2"], r["EAD"], 24)
            rows.append({"cltrMngNo": r["cltrMngNo"], "낙찰가율시나리오": name, **res})
    grid2 = pd.DataFrame(rows)
    out = []
    for name in ["비관", "기본", "낙관"]:
        s = grid2[grid2["낙찰가율시나리오"] == name]
        lgd = 1 - s["PV"].sum() / s["EAD"].sum()
        out.append({"낙찰가율시나리오": name, "rho": rho_scenarios[name], "LGD": lgd, "n": len(s)})
    return pd.DataFrame(out), rho_scenarios, len(rho_df), rho_pool_before_n, rho_pool_after_n


def main():
    big_all, target99, big_target, port, big_port = find_big_items()
    print(f"SYN-005-v3 852건 중 >200㎡: {len(big_all)}건")
    print(f"학습 대상 {len(target99)}건 중 >200㎡: {len(big_target)}건")
    print(big_target[["cltrMngNo", "구", "건물면적", "감정가", "낙찰가"]].to_string(index=False))
    print(f"포트폴리오 20건 중 >200㎡: {len(big_port)}건")
    print(big_port[["cltrMngNo", "구", "건물면적", "감정가", "낙찰가", "masked_주소"]].to_string(index=False))

    big_port_ids = set(big_port["cltrMngNo"])
    big_target_ids = set(big_target["cltrMngNo"])

    v1_table = recompute_v1(big_port_ids)
    print("\n=== v1 재계산(시세 경로, 제외 후) ===")
    print(v1_table.to_string(index=False))

    v2_table, rho_scenarios, rho_n, pool_before, pool_after = recompute_v2(big_target_ids)
    print(f"\nrho 풀: {pool_before}건 → {pool_after}건(면적 필터 후) → 유효 {rho_n}건")
    print("=== v2 재계산(보증금 기본, 24개월, 제외 후) ===")
    print(v2_table.to_string(index=False))

    return {
        "big_all": big_all, "big_target": big_target, "big_port": big_port,
        "v1_table": v1_table, "v2_table": v2_table, "rho_scenarios": rho_scenarios,
        "rho_n": rho_n, "pool_before": pool_before, "pool_after": pool_after,
    }


if __name__ == "__main__":
    main()
