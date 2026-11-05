"""EXP-009: v2 회수액·LGD (6줄 계산, 조건 단순화).

기존 src/는 수정하지 않는다. v2/price_model.py(EXP-008)의 함수를 재사용한다.
"""
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V2_ROOT))

from features import load_merged, build_features, REF_LAT, REF_LNG, KM_PER_LAT_DEG, KM_PER_LNG_DEG  # noqa: E402
from split import add_contract_date  # noqa: E402
from price_model import v2_filter, build_lgbm_frame  # noqa: E402

PROCESSED_DIR = V2_ROOT / "data" / "processed"
RESULTS_DIR = V2_ROOT / "results"
ONBID_PARQUET = ROOT / "data" / "processed" / "onbid_features_v3.parquet"
PORTFOLIO_PARQUET = ROOT / "data" / "processed" / "portfolio_v1.parquet"
MODEL_CHOICE_JSON = PROCESSED_DIR / "exp008_model_choice.json"
PORTFOLIO_V2_PARQUET = PROCESSED_DIR / "portfolio_v2.parquet"
GRID_PARQUET = PROCESSED_DIR / "exp009_recovery_v2.parquet"

PRIORITY_CAP = 165_000_000
PRIORITY_MAX = 55_000_000
LTV = 0.70
DISCOUNT_RATE = 0.08
MONTHS_SCENARIOS = [12, 24, 36]
MONTH0 = (2024, 11)


# ---------------------------------------------------------------- 6줄 계산 (EXP-009 설계, 그대로) ----
def compute_recovery_v2(market_price, rho, deposit, ead, months, discount_rate=DISCOUNT_RATE):
    s = market_price * rho                                     # ① 예상 매각가
    k = s * 0.01                                               # ② 집행비용
    if deposit <= PRIORITY_CAP:
        p = min(deposit, PRIORITY_MAX, s * 0.5)                 # ③ 최우선변제
    else:
        p = 0.0
    d2 = deposit - p                                            # ④ 보증금 잔액
    r = max(0.0, min(ead, s - k - p - d2))                      # ⑤ 회수액
    pv = r / ((1 + discount_rate) ** (months / 12))             # ⑥ 현재가치
    lgd = (1 - pv / ead) if ead > 0 else None
    return {"S": s, "K": k, "P": p, "D2": d2, "R": r, "PV": pv, "EAD": ead, "LGD": lgd}


# ---------------------------------------------------------------- 연월 이전 거래로 재학습 ----
def load_rh_base():
    merged = load_merged()
    featured, n_missing = build_features(merged)
    assert n_missing == 0, f"좌표 없는 행 {n_missing}건 — 설계에 없는 처리 필요, 멈춤"
    filtered = v2_filter(featured)
    dated = add_contract_date(filtered)
    return dated.sort_values("contract_date").reset_index(drop=True)


def gu_dong_jibun_key(gu, dong, jibun):
    return f"{gu}|{str(dong).strip()}|{str(jibun).strip()}"


def fit_cutoff_model(train_df, model_choice):
    train_f = build_lgbm_frame(train_df, train_df, exclude_self=True)
    y = np.log(train_f["price_per_m2"].to_numpy())
    feature_cols = model_choice["feature_cols"]
    dtrain = lgb.Dataset(train_f[feature_cols], label=y)
    booster = lgb.train(model_choice["lgbm_params"], dtrain, num_boost_round=model_choice["best_iteration"])
    return booster, train_f


def item_loc_km(item):
    lat_rad = math.radians(REF_LAT)
    loc_x = (item["lng"] - REF_LNG) * KM_PER_LNG_DEG * math.cos(lat_rad)
    loc_y = (item["lat"] - REF_LAT) * KM_PER_LAT_DEG
    return loc_x, loc_y


def estimate_v2_price(item, rh_base, model_choice, model_cache):
    cutoff_dt = pd.to_datetime(item["최초 개찰일"], format="%Y%m%d%H%M")
    cutoff_month_start = pd.Timestamp(year=cutoff_dt.year, month=cutoff_dt.month, day=1)
    cache_key = (cutoff_dt.year, cutoff_dt.month)

    if cache_key not in model_cache:
        train_df = rh_base[rh_base["contract_date"] < cutoff_month_start]
        if len(train_df) == 0:
            model_cache[cache_key] = None
        else:
            booster, train_f = fit_cutoff_model(train_df, model_choice)
            bldg_keys = (train_f["gu_name"].astype(str) + "|" + train_f["umdNm"].astype(str).str.strip()
                         + "|" + train_f["jibun"].astype(str).str.strip()).to_numpy()
            loc_x_arr = train_f["loc_x_km"].to_numpy()
            loc_y_arr = train_f["loc_y_km"].to_numpy()
            price_arr = train_f["price_per_m2"].to_numpy()
            model_cache[cache_key] = (booster, len(train_f), bldg_keys, loc_x_arr, loc_y_arr, price_arr)
    cached = model_cache[cache_key]
    if cached is None:
        return None, 0
    booster, train_n, bldg_keys, loc_x_arr, loc_y_arr, price_arr = cached

    loc_x, loc_y = item_loc_km(item)
    key_item = gu_dong_jibun_key(item["구"], item["동"], item["지번"])
    same_bldg = int((bldg_keys == key_item).any())
    knn_idx = np.argsort((loc_x_arr - loc_x) ** 2 + (loc_y_arr - loc_y) ** 2)[:5]
    knn_ppm2_5 = float(np.median(price_arr[knn_idx]))
    month_idx = (cutoff_dt.year - MONTH0[0]) * 12 + (cutoff_dt.month - MONTH0[1])

    row = {
        "lat": item["lat"], "lng": item["lng"], "excluUseAr": item["건물면적"],
        "age_years": item["age_years"], "floor_feat": item["floor_feat"],
        "month_idx": month_idx, "knn_ppm2_5": knn_ppm2_5, "same_bldg": same_bldg,
    }
    x = pd.DataFrame([row])[model_choice["feature_cols"]]
    pred_log = booster.predict(x)[0]
    # rh_trade dealAmount(학습 타깃)는 만원 단위, onbid 쪽(낙찰가·감정가·보증금 등)은 원 단위라 10,000을 곱해 맞춘다.
    pred_price = float(np.exp(pred_log) * item["건물면적"]) * 10_000
    return pred_price, train_n


# ---------------------------------------------------------------- 메인 ----
def main():
    with open(MODEL_CHOICE_JSON, encoding="utf-8") as f:
        model_choice = json.load(f)
    print(f"EXP-008 채택 모델: {model_choice['adopted']}, 변수 {model_choice['feature_cols']}")

    rh_base = load_rh_base()
    onbid = pd.read_parquet(ONBID_PARQUET)
    target99 = onbid[(onbid["최종 결과"] == "낙찰") & (onbid["bad_lowest"] == 0)
                      & (onbid["bad_ratio"] == 0) & (onbid["bad_bid"] == 0)].copy()

    port = pd.read_parquet(PORTFOLIO_PARQUET)
    port_sold_ids = set(port.loc[port["결과구분"] == "낙찰", "cltrMngNo"])
    rho_pool = target99[~target99["cltrMngNo"].isin(port_sold_ids)].copy()
    print(f"학습 대상 {len(target99)}건 중 포트폴리오 낙찰 12건 제외 → rho 계산용 {len(rho_pool)}건")

    model_cache = {}

    rho_rows = []
    for _, r in rho_pool.iterrows():
        pred_price, train_n = estimate_v2_price(r, rh_base, model_choice, model_cache)
        if pred_price is None or pred_price <= 0:
            continue
        rho_rows.append({"cltrMngNo": r["cltrMngNo"], "v2_price": pred_price,
                          "낙찰가": r["낙찰가"], "rho": r["낙찰가"] / pred_price, "train_n": train_n})
    rho_df = pd.DataFrame(rho_rows)
    n_fail_rho = len(rho_pool) - len(rho_df)
    q25, q50, q75 = rho_df["rho"].quantile([0.25, 0.5, 0.75])
    rho_scenarios = {"비관": float(q25), "기본": float(q50), "낙관": float(q75)}
    print(f"rho 계산 실패(시세추정 불가, 훈련 데이터 0건) {n_fail_rho}건 / {len(rho_pool)}건")
    print(f"rho 분위수(25/50/75): {rho_scenarios}")

    port_rows = []
    for _, r in port.iterrows():
        pred_price, train_n = estimate_v2_price(r, rh_base, model_choice, model_cache)
        row = r.to_dict()
        row["v2_price"] = pred_price
        row["v2_price_train_n"] = train_n
        port_rows.append(row)
    port_v2 = pd.DataFrame(port_rows)
    n_fail_port = int(port_v2["v2_price"].isna().sum())
    print(f"포트폴리오 20건 중 v2 시세 추정 실패(최초 개찰일 이전 거래 0건) {n_fail_port}건")
    if n_fail_port:
        print(port_v2.loc[port_v2["v2_price"].isna(), ["cltrMngNo", "구", "결과구분", "최초 개찰일"]])

    valid_port = port_v2[port_v2["v2_price"].notna()].copy()
    valid_port["EAD"] = valid_port["v2_price"] * LTV
    valid_port["보증금_없음_v2"] = 0.0
    valid_port["보증금_기본_v2"] = valid_port["보증금_기본"]
    valid_port["보증금_깡통_v2"] = valid_port["v2_price"] * 0.9

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    keep_cols = ["cltrMngNo", "masked_주소", "구", "결과구분", "극단물건", "건물면적",
                 "v2_price", "v2_price_train_n", "EAD",
                 "보증금_없음_v2", "보증금_기본_v2", "보증금_깡통_v2", "낙찰가"]
    valid_port[keep_cols].to_parquet(PORTFOLIO_V2_PARQUET, index=False)
    print(f"저장: {PORTFOLIO_V2_PARQUET} ({len(valid_port)}행, SYN-009)")

    grid_rows = []
    for _, r in valid_port.iterrows():
        for rho_name, rho_val in rho_scenarios.items():
            for dep_name, dep_col in [("없음", "보증금_없음_v2"), ("기본", "보증금_기본_v2"), ("깡통", "보증금_깡통_v2")]:
                for months in MONTHS_SCENARIOS:
                    res = compute_recovery_v2(r["v2_price"], rho_val, r[dep_col], r["EAD"], months)
                    grid_rows.append({
                        "cltrMngNo": r["cltrMngNo"], "구": r["구"], "결과구분": r["결과구분"],
                        "낙찰가율시나리오": rho_name, "보증금시나리오": dep_name, "개월수": months, **res,
                    })
    grid = pd.DataFrame(grid_rows)
    grid.to_parquet(GRID_PARQUET, index=False)
    print(f"저장: {GRID_PARQUET} ({len(grid)}행)")

    def portfolio_lgd(df):
        return 1 - df["PV"].sum() / df["EAD"].sum()

    table1 = grid[grid["낙찰가율시나리오"] == "기본"].pivot_table(
        index="보증금시나리오", columns="개월수", values=["PV", "EAD"], aggfunc="sum")
    t1 = pd.DataFrame({m: 1 - table1[("PV", m)] / table1[("EAD", m)] for m in MONTHS_SCENARIOS})
    t1 = t1.reindex(["없음", "기본", "깡통"])

    t2_rows = []
    for rho_name in ["비관", "기본", "낙관"]:
        sub = grid[(grid["낙찰가율시나리오"] == rho_name) & (grid["보증금시나리오"] == "기본") & (grid["개월수"] == 24)]
        t2_rows.append({"낙찰가율시나리오": rho_name, "LGD": portfolio_lgd(sub)})
    t2 = pd.DataFrame(t2_rows)

    print("\n=== 표1: 기본 낙찰가율, 보증금×시점 포트폴리오 LGD ===")
    print(t1.to_string())
    print("\n=== 표2: 낙찰가율 시나리오별(보증금 기본, 24개월) ===")
    print(t2.to_string(index=False))
    print(f"\nrho 분위수: {rho_scenarios}")
    print(f"n_fail_rho={n_fail_rho}, n_fail_port={n_fail_port}, valid_port_n={len(valid_port)}")

    return {
        "rho_scenarios": rho_scenarios, "n_fail_rho": n_fail_rho, "n_fail_port": n_fail_port,
        "valid_port_n": len(valid_port), "table1": t1, "table2": t2, "grid": grid, "port_v2": port_v2,
    }


if __name__ == "__main__":
    main()
