from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from features import load_merged, build_features, FEATURE_GROUPS
from split import split_dv02

ROOT = Path(__file__).resolve().parent.parent
PRED_PARQUET = ROOT / "data" / "processed" / "exp003_predictions.parquet"
TUNING_CSV = ROOT / "data" / "processed" / "exp003_fold_tuning.csv"

K_GRID = [5, 10, 20]
W_LOC_GRID = [1.0, 2.0, 4.0]
W_OTHER_GRID = [0.5, 1.0, 2.0]
NEIGHBOR_STORE_N = 10

KM_PER_LAT_DEG = 110.574
KM_PER_LNG_DEG = 111.320


def grid_combos():
    for k in K_GRID:
        for w_loc in W_LOC_GRID:
            for w_area in W_OTHER_GRID:
                for w_age in W_OTHER_GRID:
                    for w_floor in W_OTHER_GRID:
                        yield k, {"loc": w_loc, "area": w_area, "age": w_age, "floor": w_floor}


def fit_scaler(df):
    cols = [c for cols in FEATURE_GROUPS.values() for c in cols]
    means = df[cols].mean()
    stds = df[cols].std(ddof=0).replace(0, 1.0)
    return means, stds


def to_weighted_matrix(df, means, stds, weights):
    cols = [c for cols in FEATURE_GROUPS.values() for c in cols]
    standardized = (df[cols] - means) / stds
    for group, gcols in FEATURE_GROUPS.items():
        standardized[gcols] = standardized[gcols] * weights[group]
    return standardized[cols].to_numpy(dtype="float32")


def mdape(actual, pred):
    ape = np.abs(actual - pred) / actual
    return float(np.median(ape) * 100), ape


def predict_with_index(index, query_mat, k, ref_price_per_m2, query_area):
    D, I = index.search(query_mat, k)
    neighbor_pppm2 = ref_price_per_m2[I]
    pred_pppm2 = np.median(neighbor_pppm2, axis=1)
    pred_price = pred_pppm2 * query_area
    return pred_price, D, I


def tune_fold(train):
    train = train.sort_values("contract_date")
    train_end = train["contract_date"].max()
    val_start = train_end - pd.DateOffset(months=3) + pd.Timedelta(days=1)
    core = train[train["contract_date"] < val_start].reset_index(drop=True)
    val = train[train["contract_date"] >= val_start].reset_index(drop=True)

    assert core["contract_date"].max() < val["contract_date"].min(), "검증 구간이 학습 핵심 구간보다 앞서 있음(누수 위험)"

    means, stds = fit_scaler(core)
    core_price = core["price_per_m2"].to_numpy()
    core_area = core["excluUseAr"].to_numpy()

    best = None
    for k, weights in grid_combos():
        core_mat = to_weighted_matrix(core, means, stds, weights)
        val_mat = to_weighted_matrix(val, means, stds, weights)

        index = faiss.IndexFlatL2(core_mat.shape[1])
        index.add(core_mat)
        pred_price, _, _ = predict_with_index(index, val_mat, k, core_price, val["excluUseAr"].to_numpy())
        val_mdape, _ = mdape(val["dealAmount"].to_numpy(), pred_price)

        if best is None or val_mdape < best["val_mdape"]:
            best = {"k": k, **weights, "val_mdape": val_mdape}

    return best, len(core), len(val)


def haversine_km(lat1, lng1, lat2, lng2):
    lat_rad = np.radians((lat1 + lat2) / 2)
    dx = (lng1 - lng2) * KM_PER_LNG_DEG * np.cos(lat_rad)
    dy = (lat1 - lat2) * KM_PER_LAT_DEG
    return np.sqrt(dx ** 2 + dy ** 2)


def run_fold(name, train, test, best):
    train = train.sort_values("contract_date").reset_index(drop=True)
    test = test.reset_index(drop=True)

    means, stds = fit_scaler(train)
    weights = {g: best[g] for g in FEATURE_GROUPS}
    k = int(best["k"])

    train_mat = to_weighted_matrix(train, means, stds, weights)
    test_mat = to_weighted_matrix(test, means, stds, weights)

    index = faiss.IndexFlatL2(train_mat.shape[1])
    index.add(train_mat)

    k_query = max(k, NEIGHBOR_STORE_N)
    train_price = train["price_per_m2"].to_numpy()
    D, I = index.search(test_mat, k_query)

    neighbor_pppm2 = train_price[I][:, :k]
    pred_pppm2 = np.median(neighbor_pppm2, axis=1)
    pred_price = pred_pppm2 * test["excluUseAr"].to_numpy()

    train_dates = train["contract_date"].to_numpy()
    test_dates = test["contract_date"].to_numpy()
    neighbor_dates = train_dates[I]
    future_violations = int((neighbor_dates > test_dates[:, None]).sum())

    train_lat = train["lat"].to_numpy()
    train_lng = train["lng"].to_numpy()
    test_lat = test["lat"].to_numpy()
    test_lng = test["lng"].to_numpy()
    neighbor_lat = train_lat[I]
    neighbor_lng = train_lng[I]
    km_dist = haversine_km(test_lat[:, None], test_lng[:, None], neighbor_lat, neighbor_lng)

    month_diff = np.abs((test_dates[:, None] - neighbor_dates) / np.timedelta64(1, "D")) / 30.44

    train_deal_id = train["deal_id"].to_numpy()
    neighbor_deal_id = train_deal_id[I]

    out = pd.DataFrame({
        "deal_id": test["deal_id"].to_numpy(),
        "fold": name,
        "actual_price": test["dealAmount"].to_numpy(),
        "faiss_pred": pred_price,
        "k_used": k,
    })
    out["faiss_ape"] = (out["actual_price"] - out["faiss_pred"]).abs() / out["actual_price"]
    out["avg_neighbor_km_used"] = km_dist[:, :k].mean(axis=1)
    out["avg_neighbor_month_diff_used"] = month_diff[:, :k].mean(axis=1)

    for j in range(NEIGHBOR_STORE_N):
        out[f"neighbor{j + 1}_deal_id"] = neighbor_deal_id[:, j]
        out[f"neighbor{j + 1}_km"] = km_dist[:, j]
        out[f"neighbor{j + 1}_month_diff"] = month_diff[:, j]

    return out, future_violations


def main():
    merged = load_merged()
    featured, n_missing = build_features(merged)
    assert n_missing / len(featured) <= 0.01, "좌표 없는 행이 1%를 넘음 — 멈춤 지점"

    folds = split_dv02(featured)

    tuning_rows = []
    pred_rows = []
    total_future_violations = 0

    for name, train, test in folds:
        best, core_n, val_n = tune_fold(train)
        print(f"{name} 튜닝: core={core_n:,}, val={val_n:,}, 최적={best}")

        test_pred, future_violations = run_fold(name, train, test, best)
        test_mdape, _ = mdape(test_pred["actual_price"].to_numpy(), test_pred["faiss_pred"].to_numpy())
        print(f"{name} 테스트 MdAPE={test_mdape:.2f} (n={len(test_pred):,}), 미래 이웃 위반={future_violations}")

        total_future_violations += future_violations
        tuning_rows.append({
            "fold": name, **best, "test_mdape": test_mdape, "core_n": core_n, "val_n": val_n, "test_n": len(test_pred),
        })
        pred_rows.append(test_pred)

    assert total_future_violations == 0, f"미래 이웃 {total_future_violations}건 발견 — 멈춤 지점"

    predictions = pd.concat(pred_rows, ignore_index=True)
    PRED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PRED_PARQUET, index=False)
    print(f"저장: {PRED_PARQUET} ({len(predictions):,}행)")

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(TUNING_CSV, index=False)
    print(f"저장: {TUNING_CSV}")
    print(tuning_df.to_string(index=False))

    pooled_mdape, _ = mdape(predictions["actual_price"].to_numpy(), predictions["faiss_pred"].to_numpy())
    print(f"\npooled FAISS MdAPE: {pooled_mdape:.2f} (n={len(predictions):,})")

    return predictions, tuning_df


if __name__ == "__main__":
    main()
