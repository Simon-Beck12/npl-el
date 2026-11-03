from pathlib import Path

import numpy as np
import pandas as pd

from baseline import AREA_BINS, AREA_LABELS

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PARQUET = ROOT / "data" / "processed" / "onbid_items.parquet"
EXP002_PRED = ROOT / "data" / "processed" / "exp002_predictions.parquet"
EXP003_PRED = ROOT / "data" / "processed" / "exp003_predictions.parquet"
CLEAN_PARQUET = ROOT / "data" / "processed" / "rh_trade_clean.parquet"
LAWD_CSV = ROOT / "docs" / "seoul_lawd_cd.csv"
WEIGHTS_PARQUET = ROOT / "data" / "processed" / "dv03_weights.parquet"

AREA_RATIO_LOW = 0.77
AREA_RATIO_HIGH = 1.30
LINK_RATE_THRESHOLD = 0.60
AGE_BINS = [-np.inf, 2, 10, 20, np.inf]
AGE_LABELS = ["신축(0~2년)", "3~10년", "11~20년", "21년 이상"]


def load_onbid_target():
    items = pd.read_parquet(ITEMS_PARQUET)
    return items[(items["villa_flag"] == 1) & (items["parse_fail"] == 0)].copy()


def load_test_merged():
    exp2 = pd.read_parquet(EXP002_PRED).copy()
    exp3 = pd.read_parquet(EXP003_PRED)[["deal_id", "fold", "faiss_pred", "faiss_ape"]].copy()
    # deal_id는 sggCd|계약일|지번|층|전용면적|거래금액의 해시라 드물게 중복될 수 있어
    # (fold 내) 등장 순서를 덧붙여 1:1로 짝짓는다
    exp2["occ"] = exp2.groupby(["deal_id", "fold"]).cumcount()
    exp3["occ"] = exp3.groupby(["deal_id", "fold"]).cumcount()
    merged = exp2.merge(exp3, on=["deal_id", "fold", "occ"], how="inner")
    assert len(merged) == len(exp2) == len(exp3)
    return merged


def area_comparability(onbid, test):
    onbid_med = onbid["건물면적"].median()
    test_med = test["excluUseAr"].median()
    ratio = onbid_med / test_med
    use_area_cell = AREA_RATIO_LOW <= ratio <= AREA_RATIO_HIGH
    return ratio, use_area_cell, onbid_med, test_med


def build_cells(onbid, test, use_area_cell):
    onbid = onbid.copy()
    test = test.copy()
    if use_area_cell:
        onbid["area_bucket"] = pd.cut(onbid["건물면적"], bins=AREA_BINS, labels=AREA_LABELS)
        onbid["cell"] = onbid["구"] + "|" + onbid["area_bucket"].astype(str)
        test["cell"] = test["gu_name"] + "|" + test["area_bucket"].astype(str)
    else:
        onbid["cell"] = onbid["구"]
        test["cell"] = test["gu_name"]
    return onbid, test


def compute_cell_weights(onbid, test, cell_col="cell"):
    onbid_prop = onbid[cell_col].value_counts(normalize=True)
    test_prop = test[cell_col].value_counts(normalize=True)

    cells = sorted(set(onbid_prop.index) | set(test_prop.index))
    rows = []
    uncovered_onbid_mass = 0.0
    for c in cells:
        op = float(onbid_prop.get(c, 0.0))
        tp = float(test_prop.get(c, 0.0))
        if tp > 0:
            w = op / tp
        else:
            w = np.nan
            uncovered_onbid_mass += op
        rows.append({"cell": c, "onbid_prop": op, "test_prop": tp, "weight": w})
    weights_df = pd.DataFrame(rows)
    return weights_df, uncovered_onbid_mass


def weighted_median(values, weights):
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cum = np.cumsum(w)
    cutoff = cum[-1] / 2
    idx = np.searchsorted(cum, cutoff)
    return float(v[min(idx, len(v) - 1)])


def weighted_metrics(test, weights_df, ape_col):
    merged = test.merge(weights_df[["cell", "weight"]], on="cell", how="left")
    valid = merged.dropna(subset=["weight"])
    w = valid["weight"].to_numpy()
    ape = valid[ape_col].to_numpy()
    wmdape = weighted_median(ape, w) * 100
    w10 = float(np.sum(w * (ape <= 0.10)) / np.sum(w) * 100)
    return wmdape, w10, len(valid), len(merged) - len(valid)


def unweighted_metrics(test, ape_col):
    ape = test[ape_col].to_numpy()
    return float(np.median(ape) * 100), float(np.mean(ape <= 0.10) * 100)


def build_syn001_build_year_map():
    df = pd.read_parquet(CLEAN_PARQUET)
    lawd_map = pd.read_csv(LAWD_CSV, dtype={"LAWD_CD": str}).set_index("LAWD_CD")["gu_name"]
    df = df.copy()
    df["gu_name"] = df["sggCd"].astype(str).map(lawd_map)
    df["key"] = df["gu_name"] + "|" + df["umdNm"].str.strip() + "|" + df["jibun"].str.strip()
    return df.groupby("key")["buildYear"].agg(lambda s: s.value_counts().idxmax())


def try_secondary_weighting(onbid, test):
    build_year_by_key = build_syn001_build_year_map()

    onbid = onbid.copy()
    onbid["key"] = onbid["구"] + "|" + onbid["동"] + "|" + onbid["지번"]
    onbid["matched_build_year"] = onbid["key"].map(build_year_by_key)
    link_rate = float(onbid["matched_build_year"].notna().mean())

    if link_rate < LINK_RATE_THRESHOLD:
        return link_rate, None

    opbd_year = onbid["최종 개찰일"].str[:4].astype(float)
    onbid["age_years"] = opbd_year - onbid["matched_build_year"].astype(float)
    onbid["age_bucket"] = pd.cut(onbid["age_years"], bins=AGE_BINS, labels=AGE_LABELS)
    onbid_valid = onbid.dropna(subset=["age_bucket"]).copy()
    onbid_valid["cell"] = onbid_valid["구"] + "|" + onbid_valid["age_bucket"].astype(str)

    test2 = test.copy()
    test2["cell"] = test2["gu_name"] + "|" + test2["age_bucket"].astype(str)

    weights2_df, uncovered2 = compute_cell_weights(onbid_valid, test2)
    return link_rate, (weights2_df, test2, uncovered2, len(onbid_valid))


def main():
    onbid = load_onbid_target()
    test = load_test_merged()
    print(f"온비드 대상(villa_flag=1, parse_fail=0): {len(onbid):,}건")
    print(f"테스트 행(DV-02 pooled): {len(test):,}건")

    ratio, use_area_cell, onbid_med, test_med = area_comparability(onbid, test)
    cell_kind = "구×면적구간" if use_area_cell else "구만"
    print(f"면적 비교 가능성: 온비드 중앙값={onbid_med:.2f}㎡, 테스트 중앙값={test_med:.2f}㎡, 비율={ratio:.3f} → 기본 셀={cell_kind}")

    onbid_c, test_c = build_cells(onbid, test, use_area_cell)
    weights_df, uncovered_mass = compute_cell_weights(onbid_c, test_c)
    print(f"미포함 비율(테스트 거래 없는 셀의 온비드 비율): {uncovered_mass:.1%}")

    results = {}
    for label, ape_col in [("B1", "b1_ape"), ("FAISS", "faiss_ape")]:
        wmdape, w10, n_valid, n_dropped = weighted_metrics(test_c, weights_df, ape_col)
        umdape, u10 = unweighted_metrics(test_c, ape_col)
        results[label] = {
            "weighted_mdape": wmdape, "weighted_within10": w10,
            "unweighted_mdape": umdape, "unweighted_within10": u10,
            "n_valid": n_valid, "n_dropped": n_dropped,
        }
        print(label, results[label])

    link_rate, secondary = try_secondary_weighting(onbid, test_c)
    print(f"지번 연결률(구,동,지번 → SYN-001 건축년도): {link_rate:.1%}")

    secondary_results = None
    secondary_meta = None
    if secondary is not None:
        weights2_df, test2, uncovered2, n_onbid_valid2 = secondary
        secondary_results = {}
        for label, ape_col in [("B1", "b1_ape"), ("FAISS", "faiss_ape")]:
            wmdape, w10, n_valid, n_dropped = weighted_metrics(test2, weights2_df, ape_col)
            secondary_results[label] = {"weighted_mdape": wmdape, "weighted_within10": w10}
        secondary_meta = {"uncovered_mass": uncovered2, "n_onbid_valid": n_onbid_valid2}
        print("보조 가중(구×연식구간):", secondary_results, secondary_meta)
    else:
        print(f"보조 가중 생략(연결률 {link_rate:.1%} < 기준 {LINK_RATE_THRESHOLD:.0%})")

    WEIGHTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    weights_df.to_parquet(WEIGHTS_PARQUET, index=False)
    print(f"저장: {WEIGHTS_PARQUET}")

    n_onbid_target = len(onbid)
    if n_onbid_target < 200:
        print(f"[결론 줄에 적을 것] 온비드 대상 물건이 200건 미만입니다: {n_onbid_target}건")
    if uncovered_mass > 0.20:
        print(f"[결론 줄에 적을 것] 미포함 비율이 20%를 넘습니다: {uncovered_mass:.1%}")

    return {
        "ratio": ratio, "use_area_cell": use_area_cell, "onbid_med": onbid_med, "test_med": test_med,
        "uncovered_mass": uncovered_mass, "results": results, "link_rate": link_rate,
        "secondary_results": secondary_results, "secondary_meta": secondary_meta,
        "weights_df": weights_df, "n_onbid_target": n_onbid_target,
    }


if __name__ == "__main__":
    main()
