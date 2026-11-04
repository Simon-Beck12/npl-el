"""EXP-005: 공매 빌라 낙찰가율(rho_appr, rho_mkt) 추정. M0/M1/M2 후보를 LOO MdAE로 비교·채택한다."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = ROOT / "data" / "processed" / "onbid_features.parquet"
RATES_PARQUET = ROOT / "data" / "processed" / "exp005_rates.parquet"

ROUND_BINS = [(1, 2), (3, 5), (6, 9), (10, np.inf)]
ROUND_LABELS = ["1~2회", "3~5회", "6~9회", "10회 이상"]
MIN_BIN_N = 10
POOL_PSEUDO_N = 10


def load_learning_target(include_bad_lowest=False):
    f = pd.read_parquet(FEATURES_PARQUET)
    cond = (f["최종 결과"] == "낙찰") & (f["bad_ratio"] == 0) & (f["bad_bid"] == 0)
    if not include_bad_lowest:
        cond &= f["bad_lowest"] == 0
    target = f[cond].copy()
    target["rho_appr"] = target["낙찰가"] / target["감정가"] * 100
    target["rho_mkt"] = target["낙찰가"] / target["market_price"] * 100
    return target


def assign_round_bin(round_no):
    for (lo, hi), label in zip(ROUND_BINS, ROUND_LABELS):
        if lo <= round_no <= hi:
            return label
    return None


def build_merged_bin_map(df):
    """회차 구간을 만들고, 표본 10건 미만 구간은 인접 구간과 합친다. {원래 라벨: 병합 라벨} 반환."""
    df = df.copy()
    df["round_bin"] = df["최종 회차"].apply(assign_round_bin)
    groups = [[label] for label in ROUND_LABELS]

    def group_count(g):
        return int(df["round_bin"].isin(g).sum())

    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i, g in enumerate(groups):
            if group_count(g) < MIN_BIN_N:
                j = 1 if i == 0 else i - 1
                lo, hi = min(i, j), max(i, j)
                merged = groups[lo] + groups[hi]
                groups = groups[:lo] + [merged] + groups[lo + 1:hi] + groups[hi + 1:]
                changed = True
                break

    merge_map = {}
    for g in groups:
        label = "+".join(g)
        for orig in g:
            merge_map[orig] = label
    return merge_map


def loo_mdae_m0(values):
    n = len(values)
    total = values.sum()
    preds = (total - values) / (n - 1)
    ae = np.abs(values - preds)
    return float(np.median(ae)), preds


def loo_mdae_m1(df, value_col, bin_col):
    preds = np.empty(len(df))
    values = df[value_col].to_numpy()
    bins = df[bin_col].to_numpy()
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        v = values[idx]
        n = len(v)
        total = v.sum()
        preds[idx] = (total - v) / (n - 1) if n > 1 else np.nan
    ae = np.abs(values - preds)
    return float(np.nanmedian(ae)), preds


def loo_mdae_m2(df, value_col, bin_col, gu_col="구"):
    values = df[value_col].to_numpy()
    bins = df[bin_col].to_numpy()
    gus = df[gu_col].to_numpy()

    # M1(회차 구간) LOO 예측을 그대로 재사용
    _, m1_preds = loo_mdae_m1(df, value_col, bin_col)

    preds = np.empty(len(df))
    for i in range(len(df)):
        same_gu = (gus == gus[i])
        same_gu[i] = False
        n_gu = int(same_gu.sum())
        m1_val = m1_preds[i]
        if n_gu == 0 or np.isnan(m1_val):
            preds[i] = m1_val
        else:
            gu_med = np.median(values[same_gu])
            preds[i] = (n_gu * gu_med + POOL_PSEUDO_N * m1_val) / (n_gu + POOL_PSEUDO_N)
    ae = np.abs(values - preds)
    return float(np.nanmedian(ae)), preds


def fit_full_m0(values):
    return float(np.median(values))


def fit_full_m1(df, value_col, bin_col):
    return df.groupby(bin_col)[value_col].median()


def fit_full_m2(df, value_col, bin_col, gu_col="구"):
    m1 = fit_full_m1(df, value_col, bin_col)
    rows = []
    for gu, g in df.groupby(gu_col):
        n = len(g)
        gu_med = g[value_col].median()
        for b, m1_val in m1.items():
            est = (n * gu_med + POOL_PSEUDO_N * m1_val) / (n + POOL_PSEUDO_N)
            rows.append({"구": gu, "round_bin": b, "n_gu": n, "gu_median": gu_med, "m1_value": m1_val, "m2_estimate": est})
    return pd.DataFrame(rows)


def evaluate_candidates(df, value_col, bin_col):
    valid = df.dropna(subset=[value_col]).copy()
    m0_mdae, _ = loo_mdae_m0(valid[value_col].to_numpy())
    m1_mdae, _ = loo_mdae_m1(valid, value_col, bin_col)
    m2_mdae, _ = loo_mdae_m2(valid, value_col, bin_col)
    mdaes = {"M0": m0_mdae, "M1": m1_mdae, "M2": m2_mdae}

    best = min(mdaes.values())
    chosen = "M2"
    for cand in ["M0", "M1", "M2"]:
        if mdaes[cand] - best < 1.0:
            chosen = cand
            break
    return mdaes, chosen, valid


def main():
    target = load_learning_target(include_bad_lowest=False)
    print(f"학습 대상(최종 결과=낙찰, bad_lowest=0, bad_ratio=0, bad_bid=0): {len(target)}건")
    if len(target) < 50:
        raise RuntimeError(f"학습 대상 {len(target)}건이 50건 미만 — 멈춤 지점")

    merge_map = build_merged_bin_map(target)
    target["round_bin"] = target["최종 회차"].apply(assign_round_bin).map(merge_map)
    print("회차 구간(병합 후):", target["round_bin"].value_counts().to_dict())
    print("병합 맵:", {k: v for k, v in merge_map.items() if k != v})

    results = {}
    for name, col in [("rho_appr", "rho_appr"), ("rho_mkt", "rho_mkt")]:
        mdaes, chosen, valid = evaluate_candidates(target, col, "round_bin")
        print(f"\n{name}: n={len(valid)}, LOO MdAE={mdaes}, 채택={chosen}")
        results[name] = {"mdaes": mdaes, "chosen": chosen, "n": len(valid)}

    # ---------- 채택 모델의 전체(비LOO) 추정값 ----------
    final_estimates = {}
    for name, col in [("rho_appr", "rho_appr"), ("rho_mkt", "rho_mkt")]:
        valid = target.dropna(subset=[col])
        chosen = results[name]["chosen"]
        if chosen == "M0":
            est = fit_full_m0(valid[col].to_numpy())
            final_estimates[name] = {"type": "M0", "value": est}
        elif chosen == "M1":
            est = fit_full_m1(valid, col, "round_bin")
            final_estimates[name] = {"type": "M1", "value": est}
        else:
            est = fit_full_m2(valid, col, "round_bin")
            final_estimates[name] = {"type": "M2", "value": est}
        print(f"\n{name} 채택 모델({chosen}) 전체 추정값:")
        print(final_estimates[name]["value"])

    target.to_parquet(RATES_PARQUET, index=False)
    print(f"\n저장: {RATES_PARQUET} ({len(target)}행)")

    # ---------- 민감도: bad_lowest 포함 (판정에는 쓰지 않음) ----------
    target_incl = load_learning_target(include_bad_lowest=True)
    target_incl["round_bin"] = target_incl["최종 회차"].apply(assign_round_bin).map(merge_map)
    print(f"\n[민감도] bad_lowest 포함 학습 대상: {len(target_incl)}건")

    sensitivity = {}
    for name, col in [("rho_appr", "rho_appr"), ("rho_mkt", "rho_mkt")]:
        chosen = results[name]["chosen"]
        valid = target_incl.dropna(subset=[col])
        if chosen == "M0":
            mdae, _ = loo_mdae_m0(valid[col].to_numpy())
        elif chosen == "M1":
            mdae, _ = loo_mdae_m1(valid, col, "round_bin")
        else:
            mdae, _ = loo_mdae_m2(valid, col, "round_bin")
        median_rho = float(valid[col].median())
        median_rho_excl = float(target.dropna(subset=[col])[col].median())
        print(f"{name}(채택={chosen}) 민감도: n={len(valid)}, LOO MdAE={mdae:.2f} "
              f"(제외 시 {results[name]['mdaes'][chosen]:.2f}), 중앙값={median_rho:.2f} (제외 시 {median_rho_excl:.2f})")
        sensitivity[name] = {"n": len(valid), "loo_mdae": mdae, "median_rho": median_rho}

    return results, final_estimates, target, merge_map, sensitivity


if __name__ == "__main__":
    main()
