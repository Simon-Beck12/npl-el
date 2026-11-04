"""EXP-007 단계 7: 실제 공매 빌라 20건(낙찰 12 + 미낙찰 8)에 가상 채권 조건을 붙인다(SYN-008).

선택 규칙(실행 전 고정, seed=42, [실험 등록 내용] 그대로):
  - 후보 = SYN-005-v3에서 market_price 있고 bad_lowest·bad_ratio·bad_bid 모두 0
  - 낙찰 12건 = 학습 대상(99건) 중 rho_appr 또는 rho_mkt<30%인 물건 2건을 먼저 무작위로,
    나머지 10건을 그 외 학습 대상(97건)에서 무작위로
  - 미낙찰 8건 = 최종 결과가 유찰 또는 수의계약가능인 물건에서 무작위로
난수는 np.random.default_rng(42) 하나를 만들어 "극단 2건 -> 나머지 10건 -> 미낙찰 8건" 순서로
그대로 소비한다(순서가 바뀌면 결과가 달라지므로 바꾸지 않는다).

보증금 3단계: 없음(0) / 기본(같은 건물 전세[월세=0] 보증금 중앙값, 없으면 같은 구×면적구간 전세
중앙값, 계약일<최초개찰일만) / 깡통(추정시세×90%). 대출원금=추정시세×LTV(0.70).
낙찰가율 값: 낙찰 12건=LOO M2 예측(src/rho_refit_v3.py 결과), 미낙찰 8건=회차 구간 없이
(n_gu×구중앙값+10×전체중앙값)÷(n_gu+10).

비교사례(표시용): EXP-004 F3 설정과 최초 개찰일 이전 인덱스로 20건마다 이웃 10건을 다시 찾는다
(src/features.py, src/knn_comps.py, src/split.py, src/onbid_features.py의 함수를 그대로 재사용).
market_price 자체는 기존 v3 값(k=2)을 그대로 쓰고 덮어쓰지 않는다.
"""

import json
import math
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from deposit_ratio import load_rent_with_key
from features import build_features, load_merged, REF_LAT, REF_LNG, KM_PER_LAT_DEG, KM_PER_LNG_DEG
from knn_comps import fit_scaler, haversine_km, to_weighted_matrix
from onbid_features import load_f3_setting
from split import add_contract_date, apply_common_filter

ROOT = Path(__file__).resolve().parent.parent
FEATURES_V3_PARQUET = ROOT / "data" / "processed" / "onbid_features_v3.parquet"
RHO_LOO_PARQUET = ROOT / "data" / "processed" / "exp007_rho_loo_v3.parquet"
GU_MEDIAN_JSON = ROOT / "data" / "processed" / "exp007_rho_gu_median_v3.json"
PORTFOLIO_PARQUET = ROOT / "data" / "processed" / "portfolio_v1.parquet"
COMPS_PARQUET = ROOT / "data" / "processed" / "portfolio_comps_v1.parquet"

SEED = 42
RISK_THRESHOLD = 30.0
LTV = 0.70
JEONSE_AREA_BINS = [0, 20, 30, 40, 60, np.inf]
JEONSE_AREA_LABELS = ["0-20", "20-30", "30-40", "40-60", "60+"]
N_NEIGHBORS = 10


def mask_address(gu, dong, jibun):
    jibun = str(jibun)
    front = jibun.split("-")[0] if "-" in jibun else jibun
    return f"{gu} {dong} {front}-**"


def select_portfolio(features_v3, rho_loo):
    cand = features_v3[
        features_v3["market_price"].notna()
        & (features_v3["bad_lowest"] == 0) & (features_v3["bad_ratio"] == 0) & (features_v3["bad_bid"] == 0)
    ].copy()

    target = cand[cand["cltrMngNo"].isin(rho_loo["cltrMngNo"])].copy()
    target["rho_appr"] = target["낙찰가"] / target["감정가"] * 100
    target["rho_mkt"] = target["낙찰가"] / target["market_price"] * 100
    assert len(target) == len(rho_loo), "학습 대상 후보 수가 rho_loo와 다름"

    extreme = target[(target["rho_appr"] < RISK_THRESHOLD) | (target["rho_mkt"] < RISK_THRESHOLD)].sort_values("cltrMngNo")
    n_extreme = len(extreme)
    print(f"극단 물건(rho<{RISK_THRESHOLD}%) 후보: {n_extreme}건")
    if n_extreme < 2:
        raise RuntimeError(f"극단 물건이 {n_extreme}건으로 2건 미만 — 멈춤 지점")

    unsold_pool = cand[cand["최종 결과"].isin(["유찰", "수의계약가능"])].sort_values("cltrMngNo")
    print(f"미낙찰 후보: {len(unsold_pool)}건")
    if len(unsold_pool) < 8:
        raise RuntimeError(f"미낙찰 후보가 {len(unsold_pool)}건으로 8건 미만 — 멈춤 지점")

    rng = np.random.default_rng(SEED)

    extreme_pick_idx = rng.choice(len(extreme), size=2, replace=False)
    extreme_picked = extreme.iloc[extreme_pick_idx]

    remaining = target[~target["cltrMngNo"].isin(extreme_picked["cltrMngNo"])].sort_values("cltrMngNo")
    remaining_pick_idx = rng.choice(len(remaining), size=10, replace=False)
    remaining_picked = remaining.iloc[remaining_pick_idx]

    sold_12 = pd.concat([extreme_picked, remaining_picked]).copy()
    sold_12["결과구분"] = "낙찰"
    sold_12["극단물건"] = sold_12["cltrMngNo"].isin(extreme_picked["cltrMngNo"])

    unsold_pick_idx = rng.choice(len(unsold_pool), size=8, replace=False)
    unsold_8 = unsold_pool.iloc[unsold_pick_idx].copy()
    unsold_8["결과구분"] = "미낙찰"
    unsold_8["극단물건"] = False
    unsold_8["rho_appr"] = np.nan
    unsold_8["rho_mkt"] = np.nan

    portfolio = pd.concat([sold_12, unsold_8], ignore_index=True)
    return portfolio, n_extreme, len(unsold_pool)


def attach_rho_estimates(portfolio, rho_loo, gu_median_ref):
    rho_loo_idx = rho_loo.set_index("cltrMngNo")
    n_gu_appr = rho_loo.dropna(subset=["rho_appr"]).groupby("구").size().to_dict()
    n_gu_mkt = rho_loo.dropna(subset=["rho_mkt"]).groupby("구").size().to_dict()

    rho_appr_est, rho_mkt_est, est_source = [], [], []
    for _, row in portfolio.iterrows():
        if row["결과구분"] == "낙찰":
            r = rho_loo_idx.loc[row["cltrMngNo"]]
            rho_appr_est.append(r["rho_appr_loo_m2"])
            rho_mkt_est.append(r["rho_mkt_loo_m2"])
            est_source.append("LOO_M2")
        else:
            gu = row["구"]
            for col, n_gu_map, out_list in [
                ("rho_appr", n_gu_appr, rho_appr_est), ("rho_mkt", n_gu_mkt, rho_mkt_est),
            ]:
                ref = gu_median_ref[col]
                n_gu = n_gu_map.get(gu, 0)
                overall_med = ref["전체중앙값"]
                if n_gu == 0:
                    est = overall_med
                else:
                    gu_med = ref["구별중앙값"].get(gu, overall_med)
                    est = (n_gu * gu_med + 10 * overall_med) / (n_gu + 10)
                out_list.append(est)
            est_source.append("구중앙값보간")

    portfolio["rho_appr_est"] = rho_appr_est
    portfolio["rho_mkt_est"] = rho_mkt_est
    portfolio["rho_est_source"] = est_source
    return portfolio


def jeonse_deposit(portfolio, rent):
    rent_jeonse = rent[rent["monthlyRent"] == 0].copy()
    rent_jeonse["area_bin"] = pd.cut(rent_jeonse["excluUseAr"], JEONSE_AREA_BINS, labels=JEONSE_AREA_LABELS, right=False)
    rent_jeonse["gu_area_key"] = rent_jeonse["gu_name"] + "|" + rent_jeonse["area_bin"].astype(str)

    by_building = {k: g for k, g in rent_jeonse.groupby("key")}
    by_gu_area = {k: g for k, g in rent_jeonse.groupby("gu_area_key")}

    deposits, sources = [], []
    for _, row in portfolio.iterrows():
        cutoff = pd.to_datetime(row["최초 개찰일"], format="%Y%m%d%H%M")
        key = f"{row['구']}|{row['동'].strip()}|{row['지번'].strip()}"
        g = by_building.get(key)
        prior = g[g["contract_date"] < cutoff] if g is not None else None
        if prior is not None and len(prior) > 0:
            deposits.append(prior["deposit"].median() * 10000)
            sources.append("같은건물전세중앙값")
            continue

        area_bin = pd.cut([row["건물면적"]], JEONSE_AREA_BINS, labels=JEONSE_AREA_LABELS, right=False)[0]
        gu_area_key = f"{row['구']}|{area_bin}"
        g2 = by_gu_area.get(gu_area_key)
        prior2 = g2[g2["contract_date"] < cutoff] if g2 is not None else None
        if prior2 is not None and len(prior2) > 0:
            deposits.append(prior2["deposit"].median() * 10000)
            sources.append("같은구X면적구간전세중앙값")
        else:
            deposits.append(np.nan)
            sources.append("결측")

    portfolio["보증금_기본"] = deposits
    portfolio["보증금_기본_출처"] = sources
    portfolio["보증금_없음"] = 0.0
    portfolio["보증금_깡통"] = portfolio["market_price"] * 0.90
    return portfolio


def find_comps(portfolio, pool, k=N_NEIGHBORS):
    _, weights = load_f3_setting()
    lat_rad = math.radians(REF_LAT)
    rows = []
    for _, item in portfolio.iterrows():
        cutoff = pd.to_datetime(item["최초 개찰일"], format="%Y%m%d%H%M").floor("D")
        train_d = pool[pool["contract_date"] < cutoff].reset_index(drop=True)
        if len(train_d) < k:
            continue

        means, stds = fit_scaler(train_d)
        train_mat = to_weighted_matrix(train_d, means, stds, weights)
        index = faiss.IndexFlatL2(train_mat.shape[1])
        index.add(train_mat)

        query = pd.DataFrame([{
            "loc_x_km": (item["lng"] - REF_LNG) * KM_PER_LNG_DEG * math.cos(lat_rad),
            "loc_y_km": (item["lat"] - REF_LAT) * KM_PER_LAT_DEG,
            "log_area": np.log(item["건물면적"]),
            "age_years": item["age_years"],
            "floor_feat": item["floor_feat"],
            "house_type_bin": item["house_type_bin"],
        }])
        query_mat = to_weighted_matrix(query, means, stds, weights)
        D, I = index.search(query_mat, k)

        neighbors = train_d.iloc[I[0]]
        dist_km = haversine_km(item["lat"], item["lng"], neighbors["lat"].to_numpy(), neighbors["lng"].to_numpy())
        for rank, (_, nb) in enumerate(neighbors.iterrows(), start=1):
            rows.append({
                "cltrMngNo": item["cltrMngNo"], "순위": rank,
                "법정동": nb["umdNm"], "면적": nb["excluUseAr"], "연식": nb["age_years"],
                "층": nb["floor"], "㎡당가격_만원": nb["price_per_m2"],
                "계약월": f"{int(nb['dealYear'])}-{int(nb['dealMonth']):02d}",
                "거리_km": round(float(dist_km[rank - 1]), 3),
            })
    return pd.DataFrame(rows)


def main():
    features_v3 = pd.read_parquet(FEATURES_V3_PARQUET)
    rho_loo = pd.read_parquet(RHO_LOO_PARQUET)
    with open(GU_MEDIAN_JSON, encoding="utf-8") as f:
        gu_median_ref = json.load(f)

    portfolio, n_extreme, n_unsold_pool = select_portfolio(features_v3, rho_loo)
    print(f"포트폴리오: 낙찰 {int((portfolio['결과구분'] == '낙찰').sum())}건"
          f"(극단 {int(portfolio['극단물건'].sum())}건), 미낙찰 {int((portfolio['결과구분'] == '미낙찰').sum())}건")
    print("구 분포:", portfolio["구"].value_counts().to_dict())

    portfolio = attach_rho_estimates(portfolio, rho_loo, gu_median_ref)

    rent = load_rent_with_key()
    portfolio = jeonse_deposit(portfolio, rent)

    portfolio["대출원금"] = portfolio["market_price"] * LTV
    portfolio["masked_주소"] = [mask_address(r["구"], r["동"], r["지번"]) for _, r in portfolio.iterrows()]

    featured, _ = build_features(load_merged())
    pool = add_contract_date(apply_common_filter(featured))
    comps = find_comps(portfolio, pool)

    n_dep_missing = int(portfolio["보증금_기본"].isna().sum())
    print(f"보증금_기본 결측(건물·구X면적구간 모두 없음): {n_dep_missing}/20")

    PORTFOLIO_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    portfolio.to_parquet(PORTFOLIO_PARQUET, index=False)
    comps.to_parquet(COMPS_PARQUET, index=False)
    print(f"저장: {PORTFOLIO_PARQUET} ({len(portfolio)}행), {COMPS_PARQUET} ({len(comps)}행)")

    return portfolio, comps


if __name__ == "__main__":
    main()
