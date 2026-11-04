"""EXP-006 단계 12~15: 낙찰가율 혼합 구조(정상 매각 vs 인수부담 의심) 검증.

[실험 등록 내용]의 인수부담 기준(rho_appr<30% 또는 rho_mkt<30%), 가설 지지 조건(보증금 비율
중앙값 차이 ≥0.20 & 인수부담 집단 보증금 비율 있는 물건 ≥5건), 채택 규칙은 실행 전 고정이며
결과를 본 뒤 바꾸지 않는다. src/auction_rate.py는 수정하지 않고 import만 재사용한다.

추가(사용자 결정, 2026-11-04): 14단계 LOO 평가에 "같은 건물(구,동,지번) 단위로 빼는 LOO"를
함께 기록한다. 판정은 기존(행 단위) LOO 규칙대로 하고, 건물 단위 LOO는 기록만 한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from auction_rate import (
    ROUND_LABELS, MIN_BIN_N, POOL_PSEUDO_N,
    assign_round_bin, build_merged_bin_map,
    loo_mdae_m1, evaluate_candidates, fit_full_m0, fit_full_m1, fit_full_m2,
)

ROOT = Path(__file__).resolve().parent.parent
DEPOSIT_PARQUET = ROOT / "data" / "processed" / "onbid_deposit_ratio.parquet"
ROUNDS_PARQUET = ROOT / "data" / "processed" / "onbid_rounds_raw.parquet"

RISK_THRESHOLD = 30.0
SUPPORT_GAP = 0.20
SUPPORT_MIN_N = 5


def load_learning_target_v3():
    f = pd.read_parquet(DEPOSIT_PARQUET)
    cond = (f["최종 결과"] == "낙찰") & (f["bad_ratio"] == 0) & (f["bad_bid"] == 0) & (f["bad_lowest"] == 0)
    target = f[cond].copy()
    target["rho_appr"] = target["낙찰가"] / target["감정가"] * 100
    target["rho_mkt"] = target["낙찰가"] / target["market_price"] * 100
    return target


def step12_extreme_history(target, rounds):
    extreme = target[(target["rho_appr"] < RISK_THRESHOLD) | (target["rho_mkt"] < RISK_THRESHOLD)].copy()
    print(f"극단 물건(rho_appr<{RISK_THRESHOLD}% 또는 rho_mkt<{RISK_THRESHOLD}%): {len(extreme)}건")

    rows = []
    for _, item in extreme.iterrows():
        cid = item["cltrMngNo"]
        appr = item["감정가"]
        r = rounds[rounds["cltrMngNo"] == cid].sort_values("pbctNsq")
        ratios = (r["lowstBidPrcIndctCont"] / appr * 100).round(1).tolist()
        monotonic_down = all(a >= b - 1e-6 for a, b in zip(ratios, ratios[1:]))
        rows.append({
            "cltrMngNo": cid, "구": item["구"], "동": item["동"],
            "n_rounds": len(r), "최저가비율_1회차": ratios[0] if ratios else None,
            "최저가비율_최종회차": ratios[-1] if ratios else None,
            "최저가비율_이력": ratios, "단계적_하락": monotonic_down,
            "rho_appr": round(item["rho_appr"], 1), "rho_mkt": round(item["rho_mkt"], 1),
            "deposit_ratio": item["deposit_ratio"],
        })
    return pd.DataFrame(rows)


def step13_group_compare(target):
    target = target.copy()
    target["risk_group"] = np.where(
        (target["rho_appr"] < RISK_THRESHOLD) | (target["rho_mkt"] < RISK_THRESHOLD), "인수부담 의심", "정상 매각"
    )
    rows = []
    for g, sub in target.groupby("risk_group"):
        n_dep = int(sub["deposit_ratio"].notna().sum())
        rows.append({
            "집단": g, "n": len(sub), "보증금비율_중앙값": sub["deposit_ratio"].median(),
            "보증금비율_있는건수": n_dep,
        })
    summary = pd.DataFrame(rows).set_index("집단")

    med_risk = summary.loc["인수부담 의심", "보증금비율_중앙값"] if "인수부담 의심" in summary.index else np.nan
    med_normal = summary.loc["정상 매각", "보증금비율_중앙값"] if "정상 매각" in summary.index else np.nan
    n_risk_dep = int(summary.loc["인수부담 의심", "보증금비율_있는건수"]) if "인수부담 의심" in summary.index else 0

    gap = med_risk - med_normal if pd.notna(med_risk) and pd.notna(med_normal) else np.nan
    supported = bool(pd.notna(gap) and gap >= SUPPORT_GAP and n_risk_dep >= SUPPORT_MIN_N)

    print(f"집단 비교: {summary.to_dict('index')}")
    print(f"보증금비율 차이(인수부담-정상)={gap}, 인수부담 보증금비율 있는 건수={n_risk_dep}, 가설 지지={supported}")
    return target, summary, gap, supported


def building_key(df):
    return (df["구"] + "|" + df["동"] + "|" + df["지번"]).to_numpy()


def loo_mdae_m0_building(values, groups):
    n = len(values)
    preds = np.empty(n)
    for i in range(n):
        mask = groups != groups[i]
        preds[i] = np.median(values[mask]) if mask.any() else np.nan
    ae = np.abs(values - preds)
    return float(np.nanmedian(ae)), preds


def loo_mdae_m1_building(df, value_col, bin_col, groups):
    values = df[value_col].to_numpy()
    bins = df[bin_col].to_numpy()
    preds = np.empty(len(df))
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        v = values[idx]
        g = groups[idx]
        for pos, i in enumerate(idx):
            mask = g != g[pos]
            preds[i] = np.median(v[mask]) if mask.any() else np.nan
    ae = np.abs(values - preds)
    return float(np.nanmedian(ae)), preds


def loo_mdae_m2_building(df, value_col, bin_col, groups, gu_col="구"):
    values = df[value_col].to_numpy()
    gus = df[gu_col].to_numpy()
    _, m1_preds = loo_mdae_m1_building(df, value_col, bin_col, groups)

    preds = np.empty(len(df))
    for i in range(len(df)):
        same_gu = (gus == gus[i]) & (groups != groups[i])
        n_gu = int(same_gu.sum())
        m1_val = m1_preds[i]
        if n_gu == 0 or np.isnan(m1_val):
            preds[i] = m1_val
        else:
            gu_med = np.median(values[same_gu])
            preds[i] = (n_gu * gu_med + POOL_PSEUDO_N * m1_val) / (n_gu + POOL_PSEUDO_N)
    ae = np.abs(values - preds)
    return float(np.nanmedian(ae)), preds


def evaluate_candidates_building(df, value_col, bin_col):
    valid = df.dropna(subset=[value_col]).copy()
    groups = building_key(valid)
    m0, _ = loo_mdae_m0_building(valid[value_col].to_numpy(), groups)
    m1, _ = loo_mdae_m1_building(valid, value_col, bin_col, groups)
    m2, _ = loo_mdae_m2_building(valid, value_col, bin_col, groups)
    mdaes = {"M0": m0, "M1": m1, "M2": m2}
    best = min(mdaes.values())
    chosen = "M2"
    for cand in ["M0", "M1", "M2"]:
        if mdaes[cand] - best < 1.0:
            chosen = cand
            break
    return mdaes, chosen


def step14_refit_normal(target, risk_col="risk_group"):
    normal = target[target[risk_col] == "정상 매각"].copy()
    merge_map = build_merged_bin_map(normal)
    normal["round_bin"] = normal["최종 회차"].apply(assign_round_bin).map(merge_map)

    results = {}
    for name, col in [("rho_appr", "rho_appr"), ("rho_mkt", "rho_mkt")]:
        mdaes, chosen, valid = evaluate_candidates(normal, col, "round_bin")
        mdaes_bld, chosen_bld = evaluate_candidates_building(normal, col, "round_bin")
        differs = chosen != chosen_bld
        print(f"{name}(정상 집단, n={len(valid)}): 행단위 LOO={mdaes}, 채택={chosen} | "
              f"건물단위 LOO={mdaes_bld}, 채택(기록용)={chosen_bld}{' [판정 후보 다름!]' if differs else ''}")
        results[name] = {
            "mdaes_row": mdaes, "chosen_row": chosen, "n": len(valid),
            "mdaes_building": mdaes_bld, "chosen_building": chosen_bld, "chosen_differs": differs,
        }

    final_estimates = {}
    for name, col in [("rho_appr", "rho_appr"), ("rho_mkt", "rho_mkt")]:
        valid = normal.dropna(subset=[col])
        chosen = results[name]["chosen_row"]
        if chosen == "M0":
            est = fit_full_m0(valid[col].to_numpy())
        elif chosen == "M1":
            est = fit_full_m1(valid, col, "round_bin")
        else:
            est = fit_full_m2(valid, col, "round_bin")
        final_estimates[name] = {"type": chosen, "value": est}

    return results, final_estimates, normal, merge_map


def main():
    target = load_learning_target_v3()
    print(f"학습 대상(v3, 최종 결과=낙찰, bad_lowest=0, bad_ratio=0, bad_bid=0): {len(target)}건")

    rounds = pd.read_parquet(ROUNDS_PARQUET)
    history = step12_extreme_history(target, rounds)

    target_g, group_summary, gap, supported = step13_group_compare(target)

    out = {"target": target_g, "history": history, "group_summary": group_summary, "gap": gap, "supported": supported}

    if supported:
        n_risk = int((target_g["risk_group"] == "인수부담 의심").sum())
        p = n_risk / len(target_g)
        print(f"[가설 지지] 인수부담 비율 p={p:.3f}")
        refit_results, final_estimates, normal, merge_map = step14_refit_normal(target_g)
        out.update({"p": p, "refit_results": refit_results, "final_estimates": final_estimates,
                    "normal": normal, "merge_map": merge_map})
    else:
        print("[가설 미지지] EXP-005 M2(단일 집단) 유지 — 혼합 결과는 기록만 함")

    return out


if __name__ == "__main__":
    main()
