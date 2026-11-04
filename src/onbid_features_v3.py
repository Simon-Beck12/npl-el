"""EXP-006 단계 3·4·8: SYN-005-v3 생성.

단계 3~4(호수 추출, 중복 제거) + 단계 8(SYN-007 사용승인연도로 연식 보강, EXP-004 F3 설정으로
시세 재계산). 기존 src/parse_onbid.py, src/onbid_features.py는 수정하지 않고 import만 재사용한다.
EXP-004 F3의 k, w는 그대로 쓴다(재튜닝 안 함).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

import faiss

import knn_comps
import onbid_weights as ow
from features import load_merged, build_features, REF_LAT, REF_LNG, KM_PER_LAT_DEG, KM_PER_LNG_DEG
from split import apply_common_filter, add_contract_date
from onbid_features import (
    load_last_round_names, resolve_coords, extract_floor,
    build_syn001_floor_median, build_syn001_gu_buildyear_median, load_f3_setting,
)

ROOT = Path(__file__).resolve().parent.parent
ITEMS_V2_PARQUET = ROOT / "data" / "processed" / "onbid_items_v2.parquet"
BLD_USE_APR_CSV = ROOT / "data" / "processed" / "bld_use_apr.csv"
OUT_PARQUET = ROOT / "data" / "processed" / "onbid_features_v3.parquet"
DEDUPED_PARQUET = ROOT / "data" / "processed" / "onbid_items_deduped_v3.parquet"

BASEMENT_FLOOR_RE = re.compile(r"지하(\d*)층|지층")
BASEMENT_UNIT_RE = re.compile(r"[비B](\d+)호")
HO_RE = re.compile(r"(\d+)호")

DEDUP_KEY = ["구", "동", "지번", "ho", "감정가", "낙찰가", "최종 개찰일"]

ONBID_HOUSE_TYPE_MAP = {"다세대주택": 1.0, "연립주택": 0.0, "빌라": 0.5}


def extract_ho(name):
    """물건명에서 호수를 추출한다. 반환: (ho 또는 None, ho_fail 0/1)."""
    if not isinstance(name, str):
        return None, 1
    m = BASEMENT_FLOOR_RE.search(name)
    if m:
        return "지하" + (m.group(1) or ""), 0
    m2 = BASEMENT_UNIT_RE.search(name)
    if m2:
        return "지하" + m2.group(1), 0
    m3 = HO_RE.search(name)
    if m3:
        return m3.group(1), 0
    return None, 1


def step3_4_dedup():
    items = pd.read_parquet(ITEMS_V2_PARQUET)
    target = items[(items["villa_flag"] == 1) & (items["parse_fail"] == 0)].copy()
    n_before = len(target)

    names = load_last_round_names()
    target = target.merge(names, on="cltrMngNo", how="left")

    res = target["onbidCltrNm"].apply(extract_ho)
    target["ho"] = [r[0] for r in res]
    target["ho_fail"] = [r[1] for r in res]
    n_ho_fail = int(target["ho_fail"].sum())

    key_no_ho = [c for c in DEDUP_KEY if c != "ho"]
    same_bld_diff_ho = int((target.groupby(key_no_ho, dropna=False)["ho"].nunique() > 1).sum())

    target_sorted = target.sort_values(["최종 개찰일", "cltrMngNo"])
    n_groups = target_sorted.groupby(DEDUP_KEY, dropna=False).ngroups
    deduped = target_sorted.drop_duplicates(subset=DEDUP_KEY, keep="last").copy()
    n_removed = n_before - len(deduped)

    print(f"중복 제거 전: {n_before}건, 그룹 수: {n_groups}, 후: {len(deduped)}건(제거 {n_removed}건)")
    print(f"ho_fail: {n_ho_fail}/{n_before} ({n_ho_fail/n_before:.1%})")
    print(f"같은 건물 다른 호실 그룹 수: {same_bld_diff_ho}")

    diag = {
        "n_before": n_before, "n_groups": n_groups, "n_removed": n_removed,
        "n_after": len(deduped), "n_ho_fail": n_ho_fail, "same_bld_diff_ho": same_bld_diff_ho,
    }
    return deduped, diag


def main():
    deduped, dedup_diag = step3_4_dedup()

    n_fail = int((deduped["coord_source"] == "fail").sum()) if "coord_source" in deduped.columns else None
    deduped["geocode_key"] = "서울특별시 " + deduped["구"] + " " + deduped["동"].str.strip() + " " + deduped["지번"].str.strip()
    deduped = resolve_coords(deduped)
    fail_rate = (deduped["coord_source"] == "fail").mean()
    print(f"좌표 소스: {deduped['coord_source'].value_counts().to_dict()}, 실패율={fail_rate:.1%}")

    floor_results = deduped["onbidCltrNm"].apply(extract_floor)
    deduped["floor_raw"] = [r[0] for r in floor_results]
    deduped["floor_fail"] = [r[1] for r in floor_results]
    floor_median = build_syn001_floor_median()
    deduped["floor_feat"] = deduped["floor_raw"].fillna(floor_median)

    # ---------- 연식: SYN-001 지번 매칭(기존과 동일) + build_fail 물건은 SYN-007로 보강 ----------
    build_year_map = ow.build_syn001_build_year_map()
    deduped["key_gdj"] = deduped["구"] + "|" + deduped["동"] + "|" + deduped["지번"]
    deduped["matched_build_year"] = pd.to_numeric(deduped["key_gdj"].map(build_year_map), errors="coerce")
    deduped["build_fail_pre"] = deduped["matched_build_year"].isna().astype(int)
    n_build_fail_pre = int(deduped["build_fail_pre"].sum())

    bld = pd.read_csv(BLD_USE_APR_CSV, dtype={"cltrMngNo": str})
    bld_year = bld[bld["bld_fail"] == 0].drop_duplicates(subset="cltrMngNo").set_index("cltrMngNo")["build_year"]
    deduped["bld_build_year"] = deduped["cltrMngNo"].map(bld_year)

    deduped["build_year_resolved"] = deduped["matched_build_year"]
    use_bld = deduped["build_year_resolved"].isna() & deduped["bld_build_year"].notna()
    deduped.loc[use_bld, "build_year_resolved"] = deduped.loc[use_bld, "bld_build_year"]
    n_recovered = int(use_bld.sum())

    deduped["build_fail"] = deduped["build_year_resolved"].isna().astype(int)
    n_build_fail_post = int(deduped["build_fail"].sum())

    gu_buildyear_median = build_syn001_gu_buildyear_median()
    deduped["build_year_feat"] = deduped["build_year_resolved"].fillna(deduped["구"].map(gu_buildyear_median))

    print(f"build_fail 보강 전: {n_build_fail_pre}/{len(deduped)} ({n_build_fail_pre/len(deduped):.1%}), "
          f"SYN-007로 복구: {n_recovered}건, 보강 후: {n_build_fail_post}/{len(deduped)} ({n_build_fail_post/len(deduped):.1%})")

    opbd_year = pd.to_datetime(deduped["최초 개찰일"], format="%Y%m%d%H%M").dt.year
    deduped["age_years"] = opbd_year - deduped["build_year_feat"]
    deduped["house_type_bin"] = deduped["소분류명"].map(ONBID_HOUSE_TYPE_MAP)

    lat_rad = np.radians(REF_LAT)
    deduped["loc_x_km"] = (deduped["lng"] - REF_LNG) * KM_PER_LNG_DEG * np.cos(lat_rad)
    deduped["loc_y_km"] = (deduped["lat"] - REF_LAT) * KM_PER_LAT_DEG
    deduped["log_area"] = np.log(deduped["건물면적"])

    # ---------- 추정 시세 재계산 (EXP-004 F3 설정, 최초 개찰일 이전 거래만 인덱싱) ----------
    k, weights = load_f3_setting()
    print(f"EXP-004 F3 설정 사용(재튜닝 안 함): k={k}, weights={weights}")

    featured, _ = build_features(load_merged())
    pool = add_contract_date(apply_common_filter(featured))

    deduped["cutoff"] = pd.to_datetime(deduped["최초 개찰일"], format="%Y%m%d%H%M").dt.floor("D")
    deduped["market_price"] = np.nan
    deduped["market_price_fail"] = 0
    total_future_violations = 0

    for cutoff, grp_idx in deduped.groupby("cutoff").groups.items():
        train_d = pool[pool["contract_date"] < cutoff]
        grp = deduped.loc[grp_idx]
        has_coord = grp[["loc_x_km", "loc_y_km"]].notna().all(axis=1)

        if len(train_d) < k:
            deduped.loc[grp_idx, "market_price_fail"] = 1
            continue
        deduped.loc[grp.index[~has_coord], "market_price_fail"] = 1
        if not has_coord.any():
            continue

        means, stds = knn_comps.fit_scaler(train_d)
        train_mat = knn_comps.to_weighted_matrix(train_d, means, stds, weights)
        index = faiss.IndexFlatL2(train_mat.shape[1])
        index.add(train_mat)
        train_price = train_d["price_per_m2"].to_numpy()
        train_dates = train_d["contract_date"].to_numpy()

        query_rows = grp[has_coord].copy()
        query_mat = knn_comps.to_weighted_matrix(query_rows, means, stds, weights)
        D, I = index.search(query_mat, k)

        neighbor_dates = train_dates[I]
        future_violations = int((neighbor_dates >= np.datetime64(cutoff)).sum())
        total_future_violations += future_violations

        neighbor_pppm2 = train_price[I]
        pred_pppm2 = np.median(neighbor_pppm2, axis=1)
        pred_price_won = pred_pppm2 * query_rows["건물면적"].to_numpy() * 10000
        deduped.loc[query_rows.index, "market_price"] = pred_price_won

    print(f"추정 시세 미래 이웃 위반: {total_future_violations}건")
    assert total_future_violations == 0, "미래 이웃 발견 — 멈춤 지점"

    n_price_fail = int(deduped["market_price_fail"].sum())
    print(f"추정 시세 실패: {n_price_fail}/{len(deduped)} ({n_price_fail/len(deduped):.1%})")

    out_cols = [
        "cltrMngNo", "구", "동", "지번", "ho", "ho_fail", "건물면적", "감정가", "최종 결과", "낙찰가", "낙찰가율",
        "최초 개찰일", "최종 개찰일", "최종 회차", "bad_lowest", "bad_ratio", "bad_bid",
        "lat", "lng", "coord_source", "floor_raw", "floor_feat", "floor_fail",
        "matched_build_year", "bld_build_year", "build_year_resolved", "build_year_feat",
        "build_fail_pre", "build_fail", "age_years", "house_type_bin",
        "market_price", "market_price_fail",
    ]
    out = deduped[out_cols].copy()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"저장: {OUT_PARQUET} ({len(out):,}행)")

    diag = {
        **dedup_diag,
        "build_fail_pre": n_build_fail_pre, "build_fail_recovered": n_recovered,
        "build_fail_post": n_build_fail_post, "market_price_fail": n_price_fail,
    }
    return out, diag


if __name__ == "__main__":
    main()
