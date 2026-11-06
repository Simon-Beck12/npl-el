"""테스트용 가짜 artifact를 만든다(실제 데이터 없이 실행 가능).

가짜 거래 1,500건으로 작은 LightGBM을 학습한다. golden 기대값은 가능하면 v2/price_model.py의
`build_lgbm_frame`(학습 때 쓴 원래 피처 코드)으로 계산해, predictor.py의 피처 사본이 같은 값을 내는지 본다.
"""
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

SERVING = Path(__file__).resolve().parents[1]
V2_ROOT = SERVING.parent
ROOT = V2_ROOT.parent
sys.path.insert(0, str(SERVING))

FULL8 = ["lat", "lng", "excluUseAr", "age_years", "floor_feat", "month_idx", "knn_ppm2_5", "same_bldg"]
REF_LAT, REF_LNG = 37.5665, 126.9780
KM_LAT, KM_LNG = 110.574, 111.320
MONTH0 = (2024, 11)
RHO = {"비관": 0.5741, "기본": 0.6446, "낙관": 0.8042}


def _price_model_frame_fn():
    """원래 피처 코드(price_model.build_lgbm_frame). faiss 등이 없으면 None."""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(V2_ROOT))
        from price_model import build_lgbm_frame
        return build_lgbm_frame
    except Exception:
        return None


def _fake_deals(rng, n, year_months):
    gus = ["마포구", "은평구", "중랑구"]
    dongs = ["가동", "나동", "다동", "라동"]
    lat = rng.uniform(37.50, 37.62, n)
    lng = rng.uniform(126.88, 127.08, n)
    ym = [year_months[i] for i in rng.integers(0, len(year_months), n)]
    df = pd.DataFrame({
        "deal_id": [f"D{i:05d}" for i in range(n)],
        "lat": lat, "lng": lng,
        "excluUseAr": rng.uniform(20, 85, n).round(2),
        "buildYear": rng.integers(1990, 2023, n),
        "floor": rng.integers(1, 6, n),
        "dealYear": [y for y, _ in ym], "dealMonth": [m for _, m in ym],
        "gu_name": rng.choice(gus, n), "umdNm": rng.choice(dongs, n),
        "jibun": [f"{a}-{b}" for a, b in zip(rng.integers(1, 40, n), rng.integers(1, 5, n))],
    })
    lat_rad = math.radians(REF_LAT)
    df["loc_x_km"] = (df["lng"] - REF_LNG) * KM_LNG * math.cos(lat_rad)
    df["loc_y_km"] = (df["lat"] - REF_LAT) * KM_LAT
    df["age_years"] = df["dealYear"] - df["buildYear"]
    df["floor_feat"] = df["floor"].astype(float)
    ppm2 = 450 - 25 * np.abs(df["loc_x_km"]) - 3 * df["age_years"] + rng.normal(0, 30, n)
    df["price_per_m2"] = np.clip(ppm2, 100, None)
    df["dealAmount"] = df["price_per_m2"] * df["excluUseAr"]
    return df


def _simple_frame(df, ref, exclude_self):
    """price_model을 못 불러올 때 쓰는 대체 피처 계산(학습용)."""
    from scipy.spatial import cKDTree
    df = df.copy()
    df["month_idx"] = (df["dealYear"] - MONTH0[0]) * 12 + (df["dealMonth"] - MONTH0[1])
    tree = cKDTree(ref[["loc_x_km", "loc_y_km"]].to_numpy())
    k = 6 if exclude_self else 5
    _, idx = tree.query(df[["loc_x_km", "loc_y_km"]].to_numpy(), k=k)
    idx = idx[:, 1:] if exclude_self else idx
    df["knn_ppm2_5"] = np.median(ref["price_per_m2"].to_numpy()[idx], axis=1)
    key = lambda d: d["gu_name"] + "|" + d["umdNm"] + "|" + d["jibun"]  # noqa: E731
    counts = key(ref).value_counts()
    df["same_bldg"] = (key(df).map(counts).fillna(0) >= (2 if exclude_self else 1)).astype(int)
    return df


@pytest.fixture(scope="session")
def synthetic_artifacts(tmp_path_factory):
    rng = np.random.default_rng(0)
    train_months = [(2024, 11), (2024, 12)] + [(2025, m) for m in range(1, 13)]
    test_months = [(2026, 1), (2026, 2)]
    train = _fake_deals(rng, 1500, train_months)
    test = _fake_deals(rng, 120, test_months)
    test["deal_id"] = "T" + test["deal_id"]
    # 같은 건물 거래가 생기도록 테스트 일부의 주소·좌표를 학습 거래에서 복사한다
    cols = ["gu_name", "umdNm", "jibun", "lat", "lng", "loc_x_km", "loc_y_km"]
    test.loc[:39, cols] = train.loc[:39, cols].to_numpy()

    pm_frame = _price_model_frame_fn()
    frame_fn = pm_frame if pm_frame is not None else _simple_frame
    train_f = frame_fn(train, train, exclude_self=True)
    booster = lgb.train({"objective": "regression", "learning_rate": 0.1, "num_leaves": 15,
                         "seed": 42, "verbose": -1},
                        lgb.Dataset(train_f[FULL8], label=np.log(train_f["price_per_m2"])), num_boost_round=60)
    test_f = frame_fn(test, train, exclude_self=False)
    pred_manwon = np.exp(booster.predict(test_f[FULL8])) * test_f["excluUseAr"].to_numpy()

    out = tmp_path_factory.mktemp("artifacts")
    booster.save_model(str(out / "model.txt"))
    pd.DataFrame({
        "loc_x_km": train["loc_x_km"], "loc_y_km": train["loc_y_km"], "price_per_m2": train["price_per_m2"],
        "bldg_key": train["gu_name"] + "|" + train["umdNm"] + "|" + train["jibun"],
    }).to_csv(out / "reference.csv", index=False, float_format="%.17g")
    golden = [{
        "input": {"lat": float(r.lat), "lng": float(r.lng), "area_m2": float(r.excluUseAr),
                  "build_year": int(r.buildYear), "floor": int(r.floor), "gu": r.gu_name,
                  "dong": r.umdNm, "jibun": r.jibun, "as_of": f"{int(r.dealYear):04d}-{int(r.dealMonth):02d}"},
        "expected_price_manwon": float(p),
    } for r, p in zip(test.itertuples(), pred_manwon)]
    (out / "golden.json").write_text(json.dumps(golden, ensure_ascii=False), encoding="utf-8")
    meta = {
        "model_name": "synthetic", "model_version": "test", "created_at_kst": "-",
        "feature_cols": FULL8, "knn_k": 5, "month0": list(MONTH0), "max_month_idx": 13,
        "train_start": "2024-11-01", "train_end": "2025-12-31",
        "geo": {"ref_lat": REF_LAT, "ref_lng": REF_LNG, "km_per_lat_deg": KM_LAT, "km_per_lng_deg": KM_LNG},
        "area_q01_q99": [float(train["excluUseAr"].quantile(0.01)), float(train["excluUseAr"].quantile(0.99))],
        "test_mdape_pct": None, "test_mdape_recorded_pct": None, "test_pct_within_10": None, "test_n": 0,
        "rho_scenarios": RHO, "rho_source": "synthetic",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return {"dir": out, "golden_from_price_model": pm_frame is not None}
