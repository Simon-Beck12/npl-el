"""EXP-007 단계 5~6: v3 학습 대상(99건)으로 M2를 다시 적합하고 LOO 예측을 저장한다.
감정가 있는 물건(88건)에서 시세 경로 vs 감정가 경로 예상 낙찰가의 MdAPE도 계산한다(판정 없음, 기록만).

src/auction_rate.py의 함수(assign_round_bin, build_merged_bin_map, loo_mdae_m2, fit_full_m2)를
가져다 쓰고 수정하지 않는다. src/mixture_test.py의 load_learning_target_v3()도 그대로 재사용한다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from auction_rate import assign_round_bin, build_merged_bin_map, loo_mdae_m2
from mixture_test import load_learning_target_v3

ROOT = Path(__file__).resolve().parent.parent
LOO_PARQUET = ROOT / "data" / "processed" / "exp007_rho_loo_v3.parquet"
GU_MEDIAN_JSON = ROOT / "data" / "processed" / "exp007_rho_gu_median_v3.json"
TWO_PATH_JSON = ROOT / "results" / "exp007_two_path_mdape.json"

# EXP-005(v2) 채택 LOO MdAE — 비교용(그대로 옮김, results/exp005_auction.md 정정 실행 표 참조)
EXP005_LOO_MDAE = {"rho_appr": 6.67, "rho_mkt": 14.00}


def refit_and_loo(target):
    merge_map = build_merged_bin_map(target)
    target = target.copy()
    target["round_bin"] = target["최종 회차"].apply(assign_round_bin).map(merge_map)

    out = target[["cltrMngNo", "구", "rho_appr", "rho_mkt", "market_price", "감정가", "낙찰가"]].copy()
    gu_median_ref = {}

    for col in ["rho_appr", "rho_mkt"]:
        valid = target.dropna(subset=[col]).copy()
        mdae, preds = loo_mdae_m2(valid, col, "round_bin")
        print(f"{col}: v3 재적합 LOO MdAE(M2)={mdae:.2f}%p (EXP-005 v2={EXP005_LOO_MDAE[col]:.2f}%p, n={len(valid)})")

        loo_col = f"{col}_loo_m2"
        out.loc[valid.index, loo_col] = preds
        if loo_col not in out.columns:
            out[loo_col] = np.nan
            out.loc[valid.index, loo_col] = preds

        overall_median = float(valid[col].median())
        gu_median = valid.groupby("구")[col].median().to_dict()
        gu_median_ref[col] = {"전체중앙값": overall_median, "구별중앙값": {k: float(v) for k, v in gu_median.items()}}

    return out, gu_median_ref


def two_path_mdape(rho_out):
    """감정가 있는 물건(rho_appr 유효)에서 시세 경로 vs 감정가 경로 예상 낙찰가 MdAPE, 같은 물건 집합."""
    both_valid = rho_out.dropna(subset=["rho_appr", "rho_appr_loo_m2", "rho_mkt_loo_m2", "market_price", "감정가"]).copy()
    n = len(both_valid)
    print(f"두 경로 비교 대상(감정가·시세·LOO 예측 모두 유효): {n}건")

    pred_market = both_valid["market_price"] * both_valid["rho_mkt_loo_m2"] / 100
    pred_appr = both_valid["감정가"] * both_valid["rho_appr_loo_m2"] / 100
    actual = both_valid["낙찰가"]

    ape_market = (pred_market - actual).abs() / actual
    ape_appr = (pred_appr - actual).abs() / actual

    mdape_market = float(ape_market.median())
    mdape_appr = float(ape_appr.median())
    print(f"시세 경로 MdAPE={mdape_market:.4f}, 감정가 경로 MdAPE={mdape_appr:.4f}")

    return {"n": n, "mdape_시세경로": mdape_market, "mdape_감정가경로": mdape_appr}


def main():
    target = load_learning_target_v3()
    print(f"v3 학습 대상: {len(target)}건")

    rho_out, gu_median_ref = refit_and_loo(target)
    rho_out.to_parquet(LOO_PARQUET, index=False)
    print(f"저장: {LOO_PARQUET} ({len(rho_out)}행)")

    with open(GU_MEDIAN_JSON, "w", encoding="utf-8") as f:
        json.dump(gu_median_ref, f, ensure_ascii=False, indent=2)
    print(f"저장: {GU_MEDIAN_JSON}")

    two_path = two_path_mdape(rho_out)
    with open(TWO_PATH_JSON, "w", encoding="utf-8") as f:
        json.dump(two_path, f, ensure_ascii=False, indent=2)
    print(f"저장: {TWO_PATH_JSON}")

    return rho_out, gu_median_ref, two_path


if __name__ == "__main__":
    main()
