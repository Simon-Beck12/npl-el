"""EXP-008: v2 가격 추정 (LightGBM + k-NN 피처).

기존 src/는 수정하지 않는다. features.py/split.py/knn_comps.py의 함수를 그대로 재사용한다.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import load_merged, build_features  # noqa: E402
from split import add_contract_date, TRAIN_START, TRAIN_END, TEST_START, TEST_END  # noqa: E402
from knn_comps import run_fold  # noqa: E402

V2_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = V2_ROOT / "results"
PROCESSED_DIR = V2_ROOT / "data" / "processed"
EXP004_PRED_PARQUET = ROOT / "data" / "processed" / "exp004_predictions.parquet"

SEED = 42
VAL_START = "2026-03-01"
MONTH0 = (2024, 11)
EXP004_F3_BEST = {"k": 2, "loc": 16.0, "area": 1.0, "age": 2.0, "floor": 0.5}
EXP004_F3_REF_MDAPE = 13.432519325440188
REPRO_TOLERANCE = 0.05

LGBM_GRID = [
    {"num_leaves": nl, "min_child_samples": mc}
    for nl in (31, 63) for mc in (20, 50)
]
LGBM_BASE_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "seed": SEED,
    "verbose": -1,
}
NUM_BOOST_ROUND_MAX = 3000
EARLY_STOPPING_ROUNDS = 100
KNN_K = 5


def mdape_price(actual_price, pred_price):
    """예측 결측(NaN)은 지표 계산에서 제외하고 건수만 따로 센다(SYN-005 "추정 실패" 처리와 동일한 방식)."""
    ape = np.abs(actual_price - pred_price) / actual_price
    valid = ~np.isnan(ape)
    n_na = int((~valid).sum())
    pct10 = float((ape[valid] <= 0.10).mean() * 100) if valid.any() else float("nan")
    mdape = float(np.median(ape[valid]) * 100) if valid.any() else float("nan")
    return mdape, pct10, ape, n_na


# ---------------------------------------------------------------- 데이터 ----
def v2_filter(df):
    """[실험 등록 내용] 필터 3개: 계약 해제(이미 반영) / 직거래 제외 / 상하위 1% 제외.
    EXP-002의 공공기관 거래 제외는 v2 필터 목록에 없으므로 적용하지 않는다."""
    return df[(df["direct_flag"] == 0) & (df["outlier_flag"] == 0)].copy()


def load_v2_split():
    merged = load_merged()
    featured, n_missing = build_features(merged)
    assert n_missing == 0, f"좌표 없는 행 {n_missing}건 — 설계에 없는 처리 필요, 멈춤"

    filtered = v2_filter(featured)
    dated = add_contract_date(filtered)
    train_all = dated[(dated["contract_date"] >= TRAIN_START) & (dated["contract_date"] <= TRAIN_END)].copy()
    test = dated[(dated["contract_date"] >= TEST_START) & (dated["contract_date"] <= TEST_END)].copy()
    core = train_all[train_all["contract_date"] < VAL_START].copy()
    val = train_all[train_all["contract_date"] >= VAL_START].copy()
    assert core["contract_date"].max() < val["contract_date"].min()
    return train_all, core, val, test


# ---------------------------------------------------------------- 재현 확인 (4단계) ----
def reproduce_check():
    exp004_pred = pd.read_parquet(EXP004_PRED_PARQUET)
    f3 = exp004_pred[exp004_pred["fold"] == "F3"]
    mdape = float(np.median(f3["faiss_ape"]) * 100)
    diff = abs(mdape - EXP004_F3_REF_MDAPE)
    print(f"[재현 확인] F3 캐시 재계산 MdAPE={mdape:.4f}%, 등록값={EXP004_F3_REF_MDAPE:.4f}%, 차이={diff:.4f}%p")
    if diff > REPRO_TOLERANCE:
        raise SystemExit(f"재현 확인 실패: 차이 {diff:.4f}%p > {REPRO_TOLERANCE}%p — 멈춤 (오류로 멈춤, 직접 고쳐 재시작하지 않음)")
    print(f"[재현 확인] 통과 (허용 오차 {REPRO_TOLERANCE}%p)")
    return mdape, diff, len(f3)


# ---------------------------------------------------------------- B0 ----
def model_b0(train_all, test):
    dong_median = train_all.groupby("umdNm")["price_per_m2"].median()
    pred_pppm2 = test["umdNm"].map(dong_median)
    pred_price = pred_pppm2 * test["excluUseAr"]
    n_na = int(pred_price.isna().sum())
    return pred_price.to_numpy(), n_na


# ---------------------------------------------------------------- A (EXP-004 F3, 재튜닝 없음) ----
def model_a(train_all, test):
    out, future_violations = run_fold("v2", train_all, test, EXP004_F3_BEST)
    assert future_violations == 0, f"미래 이웃 {future_violations}건 — 누수, 멈춤"
    return out.set_index("deal_id")["faiss_pred"].reindex(test["deal_id"]).to_numpy()


# ---------------------------------------------------------------- G1/G2 피처 ----
def month_index(df):
    year = df["dealYear"].astype(int)
    month = df["dealMonth"].astype(int)
    base_y, base_m = MONTH0
    return (year - base_y) * 12 + (month - base_m)


def bldg_key(df):
    return (df["gu_name"].astype(str) + "|" + df["umdNm"].astype(str).str.strip()
            + "|" + df["jibun"].astype(str).str.strip())


def add_same_bldg(target, ref, exclude_self):
    counts = bldg_key(ref).value_counts()
    key_t = bldg_key(target)
    threshold = 2 if exclude_self else 1
    return (key_t.map(counts).fillna(0) >= threshold).astype(int).to_numpy()


def add_knn_ppm2(target, ref, exclude_self, k=KNN_K):
    ref = ref.reset_index(drop=True)
    target = target.reset_index(drop=True)
    tree = cKDTree(ref[["loc_x_km", "loc_y_km"]].to_numpy())
    ref_price = ref["price_per_m2"].to_numpy()
    q = k + 1 if exclude_self else k
    _, idx = tree.query(target[["loc_x_km", "loc_y_km"]].to_numpy(), k=q)
    if q == 1:
        idx = idx[:, None]
    if not exclude_self:
        return np.median(ref_price[idx], axis=1)

    ref_deal_id = ref["deal_id"].to_numpy()
    target_deal_id = target["deal_id"].to_numpy()
    result = np.empty(len(target))
    for i in range(len(target)):
        neighbor_idx = idx[i]
        neighbor_ids = ref_deal_id[neighbor_idx]
        keep = neighbor_ids != target_deal_id[i]
        chosen = neighbor_idx[keep][:k]
        result[i] = np.median(ref_price[chosen])
    return result


BASE6 = ["lat", "lng", "excluUseAr", "age_years", "floor_feat", "month_idx"]
FULL8 = BASE6 + ["knn_ppm2_5", "same_bldg"]


def build_lgbm_frame(df, ref, exclude_self):
    df = df.copy()
    df["month_idx"] = month_index(df)
    df["knn_ppm2_5"] = add_knn_ppm2(df, ref, exclude_self)
    df["same_bldg"] = add_same_bldg(df, ref, exclude_self)
    return df


# ---------------------------------------------------------------- G1/G2 학습 ----
def tune_lgbm(core, val, feature_cols):
    core_f = build_lgbm_frame(core, core, exclude_self=True)
    val_f = build_lgbm_frame(val, core, exclude_self=False)

    y_core = np.log(core_f["price_per_m2"].to_numpy())
    y_val = np.log(val_f["price_per_m2"].to_numpy())
    dtrain = lgb.Dataset(core_f[feature_cols], label=y_core)
    dval = lgb.Dataset(val_f[feature_cols], label=y_val, reference=dtrain)

    best = None
    for combo in LGBM_GRID:
        params = {**LGBM_BASE_PARAMS, **combo}
        booster = lgb.train(
            params, dtrain, num_boost_round=NUM_BOOST_ROUND_MAX,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        pred_log = booster.predict(val_f[feature_cols], num_iteration=booster.best_iteration)
        pred_price = np.exp(pred_log) * val_f["excluUseAr"].to_numpy()
        val_mdape, _, _, _ = mdape_price(val_f["dealAmount"].to_numpy(), pred_price)
        print(f"  조합 {combo} best_iter={booster.best_iteration} val_MdAPE={val_mdape:.2f}%")
        if best is None or val_mdape < best["val_mdape"]:
            best = {"params": combo, "best_iteration": booster.best_iteration, "val_mdape": val_mdape}
    return best


def fit_final_lgbm(train_all, best, feature_cols):
    train_f = build_lgbm_frame(train_all, train_all, exclude_self=True)
    y = np.log(train_f["price_per_m2"].to_numpy())
    dtrain = lgb.Dataset(train_f[feature_cols], label=y)
    params = {**LGBM_BASE_PARAMS, **best["params"]}
    booster = lgb.train(params, dtrain, num_boost_round=best["best_iteration"])
    return booster


def predict_lgbm(booster, test, train_all, feature_cols):
    test_f = build_lgbm_frame(test, train_all, exclude_self=False)
    pred_log = booster.predict(test_f[feature_cols])
    return np.exp(pred_log) * test_f["excluUseAr"].to_numpy(), test_f


# ---------------------------------------------------------------- 메인 ----
def main():
    repro_mdape, repro_diff, repro_n = reproduce_check()

    train_all, core, val, test = load_v2_split()
    print(f"train_all={len(train_all):,} core={len(core):,} val={len(val):,} test={len(test):,}")

    actual_price = test["dealAmount"].to_numpy()

    pred_b0, _ = model_b0(train_all, test)
    mdape_b0, pct10_b0, ape_b0, na_b0 = mdape_price(actual_price, pred_b0)

    pred_a = model_a(train_all, test)
    mdape_a, pct10_a, ape_a, na_a = mdape_price(actual_price, pred_a)

    print("[G1] 변수 6개 튜닝")
    best_g1 = tune_lgbm(core, val, BASE6)
    booster_g1 = fit_final_lgbm(train_all, best_g1, BASE6)
    pred_g1, test_f_g1 = predict_lgbm(booster_g1, test, train_all, BASE6)
    mdape_g1, pct10_g1, ape_g1, na_g1 = mdape_price(actual_price, pred_g1)

    print("[G2] 변수 8개 튜닝")
    best_g2 = tune_lgbm(core, val, FULL8)
    booster_g2 = fit_final_lgbm(train_all, best_g2, FULL8)
    pred_g2, test_f_g2 = predict_lgbm(booster_g2, test, train_all, FULL8)
    mdape_g2, pct10_g2, ape_g2, na_g2 = mdape_price(actual_price, pred_g2)

    # 같은 건물 유무별 (G2 same_bldg 기준) — ape에 NaN(예측 결측)이 섞여 있을 수 있어 nanmedian 사용
    same_bldg_test = test_f_g2["same_bldg"].to_numpy()
    rows_grp = []
    for label, mask_val in [("있음", 1), ("없음", 0)]:
        mask = same_bldg_test == mask_val
        n = int(mask.sum())
        row = {"구분": label, "n": n}
        for name, ape in [("B0", ape_b0), ("A", ape_a), ("G1", ape_g1), ("G2", ape_g2)]:
            row[name] = float(np.nanmedian(ape[mask]) * 100) if n > 0 else None
        rows_grp.append(row)
    group_table = pd.DataFrame(rows_grp)

    # 채택 규칙(실행 전 고정, 그대로 적용)
    diff_g1g2 = mdape_g2 - mdape_g1
    if abs(diff_g1g2) < 0.3:
        chosen_name, chosen_mdape = "G1", mdape_g1
    else:
        chosen_name, chosen_mdape = ("G1", mdape_g1) if mdape_g1 < mdape_g2 else ("G2", mdape_g2)

    if chosen_mdape < mdape_b0:
        verdict = f"채택({chosen_name})"
        adopted = chosen_name
    else:
        verdict = "기각 — A(EXP-004) 유지"
        adopted = "A"

    summary = pd.DataFrame([
        {"모델": "B0", "MdAPE": mdape_b0, "±10%": pct10_b0, "n": len(test), "n_결측": na_b0},
        {"모델": "A(EXP-004 F3)", "MdAPE": mdape_a, "±10%": pct10_a, "n": len(test), "n_결측": na_a},
        {"모델": "G1(6변수)", "MdAPE": mdape_g1, "±10%": pct10_g1, "n": len(test), "n_결측": na_g1,
         "num_leaves": best_g1["params"]["num_leaves"], "min_child_samples": best_g1["params"]["min_child_samples"],
         "best_iteration": best_g1["best_iteration"], "val_MdAPE": best_g1["val_mdape"]},
        {"모델": "G2(8변수)", "MdAPE": mdape_g2, "±10%": pct10_g2, "n": len(test), "n_결측": na_g2,
         "num_leaves": best_g2["params"]["num_leaves"], "min_child_samples": best_g2["params"]["min_child_samples"],
         "best_iteration": best_g2["best_iteration"], "val_MdAPE": best_g2["val_mdape"]},
    ])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "deal_id": test["deal_id"].to_numpy(), "actual_price": actual_price,
        "pred_b0": pred_b0, "pred_a": pred_a, "pred_g1": pred_g1, "pred_g2": pred_g2,
    }).to_parquet(PROCESSED_DIR / "exp008_predictions.parquet", index=False)

    importance_table = None
    chosen_booster = booster_g1 if adopted == "G1" else (booster_g2 if adopted == "G2" else None)
    chosen_features = BASE6 if adopted == "G1" else (FULL8 if adopted == "G2" else None)
    if chosen_booster is not None:
        gain = chosen_booster.feature_importance(importance_type="gain")
        importance_table = pd.DataFrame({"변수": chosen_features, "gain": gain}).sort_values(
            "gain", ascending=False).reset_index(drop=True)

    print("\n=== 표1: 모델 4개 비교 ===")
    print(summary.to_string(index=False))
    print("\n=== 표2: 같은 건물 유무별 ===")
    print(group_table.to_string(index=False))
    print(f"\n판정: {verdict} (G2-G1 차이={diff_g1g2:+.2f}%p, 채택 기준 0.3%p)")
    if importance_table is not None:
        print(f"\n=== 표3: 채택 모델({adopted}) 피처 중요도(gain) ===")
        print(importance_table.to_string(index=False))

    choice = {
        "adopted": adopted,
        "feature_cols": chosen_features,
        "lgbm_params": {**LGBM_BASE_PARAMS, **(best_g1["params"] if adopted == "G1" else best_g2["params"])}
        if adopted in ("G1", "G2") else None,
        "best_iteration": (best_g1["best_iteration"] if adopted == "G1" else best_g2["best_iteration"])
        if adopted in ("G1", "G2") else None,
        "verdict": verdict,
        "test_mdape": {"B0": mdape_b0, "A": mdape_a, "G1": mdape_g1, "G2": mdape_g2},
    }
    import json
    with open(PROCESSED_DIR / "exp008_model_choice.json", "w", encoding="utf-8") as f:
        json.dump(choice, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {PROCESSED_DIR / 'exp008_model_choice.json'}")

    return {
        "repro_mdape": repro_mdape, "repro_diff": repro_diff, "repro_n": repro_n,
        "summary": summary, "group_table": group_table, "importance_table": importance_table,
        "verdict": verdict, "adopted": adopted,
        "best_g1": best_g1, "best_g2": best_g2,
        "train_n": len(train_all), "core_n": len(core), "val_n": len(val), "test_n": len(test),
    }


if __name__ == "__main__":
    main()
