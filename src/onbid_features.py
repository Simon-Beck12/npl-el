"""SYN-005: 공매 빌라 물건 피처(좌표, 층, 연식, 추정 시세) 생성.

데이터 소스: SYN-004(villa_flag=1, parse_fail=0, 863건), SYN-002(지오코딩 캐시, 재사용만),
SYN-001(건축년도 연결, 시세 인덱스). 시세 모델은 EXP-004의 F3 설정(data/processed/exp004_fold_tuning.csv)을
그대로 읽어 쓴다 — 다시 튜닝하지 않는다.
"""

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

import faiss

import knn_comps
import onbid_weights as ow
from features import load_merged, build_features, REF_LAT, REF_LNG, KM_PER_LAT_DEG, KM_PER_LNG_DEG
from split import apply_common_filter, add_contract_date, DV02_FOLDS

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PARQUET = ROOT / "data" / "processed" / "onbid_items.parquet"
ROUNDS_PARQUET = ROOT / "data" / "processed" / "onbid_rounds_raw.parquet"
SYN002_CACHE_CSV = ROOT / "data" / "processed" / "geocode_cache.csv"
NEW_CACHE_CSV = ROOT / "data" / "processed" / "geocode_cache_onbid.csv"
TUNING_CSV = ROOT / "data" / "processed" / "exp004_fold_tuning.csv"
OUT_PARQUET = ROOT / "data" / "processed" / "onbid_features.parquet"

KAKAO_KEY = os.getenv("KAKAO_REST_API_KEY")
KAKAO_URL = "https://dapi.kakao.com/v2/local/search/address.json"
CALL_INTERVAL_SEC = 0.1
MAX_NEW_CALLS = 1000
CACHE_FIELDS = ["geocode_key", "lat", "lng", "status"]

COORD_FAIL_STOP_RATE = 0.10

ONBID_HOUSE_TYPE_MAP = {"다세대주택": 1.0, "연립주택": 0.0, "빌라": 0.5}

FLOOR_RE = re.compile(r"제(\d+)층")
BASEMENT_EXPLICIT_RE = re.compile(r"지하층|지층")
BASEMENT_UNIT_RE = re.compile(r"[비B](\d+)호")
HO_RE = re.compile(r"(\d+)호")


def extract_floor(name):
    """물건명에서 층을 추출한다. 반환: (floor 또는 None, floor_fail 0/1)."""
    if not isinstance(name, str):
        return None, 1
    if BASEMENT_EXPLICIT_RE.search(name):
        return -1.0, 0
    m = FLOOR_RE.search(name)
    if m:
        return float(m.group(1)), 0
    m2 = BASEMENT_UNIT_RE.search(name)
    if m2:
        return -1.0, 0
    m3 = HO_RE.search(name)
    if m3:
        digits = m3.group(1)
        if len(digits) >= 3:
            return float(int(digits[:-2])), 0
    return None, 1


def load_last_round_names():
    rounds = pd.read_parquet(ROUNDS_PARQUET)
    rounds = rounds.sort_values(["cltrMngNo", "pbctNsq"])
    last = rounds.groupby("cltrMngNo").last().reset_index()
    return last[["cltrMngNo", "onbidCltrNm"]]


def kakao_geocode_one(key):
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    return requests.get(KAKAO_URL, headers=headers, params={"query": key}, timeout=10)


def load_new_cache():
    cache = {}
    if NEW_CACHE_CSV.exists():
        df = pd.read_csv(NEW_CACHE_CSV, dtype=str)
        for _, row in df.iterrows():
            cache[row["geocode_key"]] = row.to_dict()
    return cache


def append_new_cache_row(row):
    file_exists = NEW_CACHE_CSV.exists()
    pd.DataFrame([row])[CACHE_FIELDS].to_csv(
        NEW_CACHE_CSV, mode="a", header=not file_exists, index=False
    )


def resolve_coords(target):
    """SYN-002 캐시 우선 재사용, 없으면 카카오로 신규 지오코딩(새 캐시 파일에 누적)."""
    syn002 = pd.read_csv(SYN002_CACHE_CSV)
    syn002_ok = syn002[syn002["status"] == "ok"].set_index("geocode_key")[["lat", "lng"]]

    keys = target["geocode_key"].unique()
    reused = [k for k in keys if k in syn002_ok.index]
    missing = [k for k in keys if k not in syn002_ok.index]

    new_cache = load_new_cache()
    still_missing = [k for k in missing if k not in new_cache]

    print(f"고유 키 {len(keys)}개: SYN-002 재사용 {len(reused)}개, 신규 필요 {len(missing)}개 "
          f"(이미 신규 캐시에 있음 {len(missing) - len(still_missing)}개, 실제 호출 {len(still_missing)}개)")

    if len(still_missing) > MAX_NEW_CALLS:
        raise RuntimeError(f"신규 호출 필요 {len(still_missing)}건이 한도 {MAX_NEW_CALLS}건을 넘음 — 멈춤")

    for key in still_missing:
        resp = kakao_geocode_one(key)
        if resp.status_code == 429:
            raise RuntimeError(f"카카오 API HTTP 429 수신 — 멈춤 지점. 진행: {len(new_cache)}/{len(missing)}")
        if resp.status_code != 200:
            raise RuntimeError(f"카카오 API HTTP {resp.status_code} 수신 — 멈춤. 응답: {resp.text[:300]}")

        data = resp.json()
        documents = data.get("documents", [])
        if documents:
            doc = documents[0]
            row = {"geocode_key": key, "lat": doc.get("y", ""), "lng": doc.get("x", ""), "status": "ok"}
        else:
            row = {"geocode_key": key, "lat": "", "lng": "", "status": "no_match"}
        append_new_cache_row(row)
        new_cache[key] = row
        time.sleep(CALL_INTERVAL_SEC)

    print(f"신규 지오코딩 호출: {len(still_missing)}건 완료")

    def lookup(key):
        if key in syn002_ok.index:
            return float(syn002_ok.loc[key, "lat"]), float(syn002_ok.loc[key, "lng"]), "reused"
        row = new_cache.get(key)
        if row is None or row.get("status") != "ok" or row.get("lat") in ("", None):
            return np.nan, np.nan, "fail"
        return float(row["lat"]), float(row["lng"]), "new_ok"

    results = target["geocode_key"].map(lambda k: lookup(k))
    target["lat"] = [r[0] for r in results]
    target["lng"] = [r[1] for r in results]
    target["coord_source"] = [r[2] for r in results]
    return target


def build_syn001_floor_median():
    f3 = next(f for f in DV02_FOLDS if f[0] == "F3")
    _, tr_s, tr_e, _, _ = f3
    raw = pd.read_parquet(ROOT / "data" / "processed" / "rh_trade_clean.parquet")
    filtered = add_contract_date(apply_common_filter(raw))
    train_f3 = filtered[(filtered["contract_date"] >= tr_s) & (filtered["contract_date"] <= tr_e)]
    villa_like = train_f3[train_f3["houseType"].isin(["다세대", "연립", "연립다세대"])]
    return float(villa_like["floor"].median())


def build_syn001_gu_buildyear_median():
    f3 = next(f for f in DV02_FOLDS if f[0] == "F3")
    _, tr_s, tr_e, _, _ = f3
    raw = pd.read_parquet(ROOT / "data" / "processed" / "rh_trade_clean.parquet")
    lawd_map = pd.read_csv(ROOT / "docs" / "seoul_lawd_cd.csv", dtype={"LAWD_CD": str}).set_index("LAWD_CD")["gu_name"]
    raw = raw.copy()
    raw["gu_name"] = raw["sggCd"].astype(str).map(lawd_map)
    raw["buildYear"] = raw["buildYear"].astype(int)
    filtered = add_contract_date(apply_common_filter(raw))
    train_f3 = filtered[(filtered["contract_date"] >= tr_s) & (filtered["contract_date"] <= tr_e)]
    return train_f3.groupby("gu_name")["buildYear"].median()


def load_f3_setting():
    tuning = pd.read_csv(TUNING_CSV)
    row = tuning[tuning["fold"] == "F3"].iloc[0]
    weights = {"loc": float(row["loc"]), "area": float(row["area"]), "age": float(row["age"]), "floor": float(row["floor"])}
    k = int(row["k"])
    return k, weights


def main():
    k, weights = load_f3_setting()
    print(f"EXP-004 F3 설정 사용: k={k}, weights={weights}")

    onbid = ow.load_onbid_target()
    print(f"온비드 대상(villa_flag=1, parse_fail=0): {len(onbid):,}건")

    names = load_last_round_names()
    onbid = onbid.merge(names, on="cltrMngNo", how="left")

    onbid["geocode_key"] = "서울특별시 " + onbid["구"] + " " + onbid["동"].str.strip() + " " + onbid["지번"].str.strip()
    onbid = resolve_coords(onbid)

    n_fail = int((onbid["coord_source"] == "fail").sum())
    fail_rate = n_fail / len(onbid)
    print(f"좌표 소스: {onbid['coord_source'].value_counts().to_dict()}, 실패율={fail_rate:.1%}")
    if fail_rate > COORD_FAIL_STOP_RATE:
        raise RuntimeError(f"좌표 실패율 {fail_rate:.1%}이 기준 {COORD_FAIL_STOP_RATE:.0%}를 넘음 — 멈춤")

    floor_results = onbid["onbidCltrNm"].apply(extract_floor)
    onbid["floor_raw"] = [r[0] for r in floor_results]
    onbid["floor_fail"] = [r[1] for r in floor_results]
    floor_median = build_syn001_floor_median()
    onbid["floor_feat"] = onbid["floor_raw"].fillna(floor_median)
    print(f"floor_fail: {onbid['floor_fail'].sum()}/{len(onbid)} ({onbid['floor_fail'].mean():.1%}), "
          f"대체값(학습구간 빌라 층 중앙값)={floor_median}")

    build_year_map = ow.build_syn001_build_year_map()
    onbid["key_gdj"] = onbid["구"] + "|" + onbid["동"] + "|" + onbid["지번"]
    onbid["matched_build_year"] = pd.to_numeric(onbid["key_gdj"].map(build_year_map), errors="coerce")
    onbid["build_fail"] = onbid["matched_build_year"].isna().astype(int)
    gu_buildyear_median = build_syn001_gu_buildyear_median()
    onbid["build_year_feat"] = onbid["matched_build_year"].fillna(onbid["구"].map(gu_buildyear_median))
    print(f"build_fail: {onbid['build_fail'].sum()}/{len(onbid)} ({onbid['build_fail'].mean():.1%})")

    opbd_year = pd.to_datetime(onbid["최초 개찰일"], format="%Y%m%d%H%M").dt.year
    onbid["age_years"] = opbd_year - onbid["build_year_feat"]
    onbid["house_type_bin"] = onbid["소분류명"].map(ONBID_HOUSE_TYPE_MAP)

    lat_rad = np.radians(REF_LAT)
    onbid["loc_x_km"] = (onbid["lng"] - REF_LNG) * KM_PER_LNG_DEG * np.cos(lat_rad)
    onbid["loc_y_km"] = (onbid["lat"] - REF_LAT) * KM_PER_LAT_DEG
    onbid["log_area"] = np.log(onbid["건물면적"])

    # ---------- 추정 시세 (EXP-004 F3 설정, 최초 개찰일 이전 거래만 인덱싱) ----------
    featured, _ = build_features(load_merged())
    pool = add_contract_date(apply_common_filter(featured))

    onbid["cutoff"] = pd.to_datetime(onbid["최초 개찰일"], format="%Y%m%d%H%M").dt.floor("D")

    onbid["market_price"] = np.nan
    onbid["market_price_fail"] = 0
    total_future_violations = 0

    for cutoff, grp_idx in onbid.groupby("cutoff").groups.items():
        train_d = pool[pool["contract_date"] < cutoff]
        grp = onbid.loc[grp_idx]
        has_coord = grp[["loc_x_km", "loc_y_km"]].notna().all(axis=1)

        if len(train_d) < k:
            onbid.loc[grp_idx, "market_price_fail"] = 1
            continue
        onbid.loc[grp.index[~has_coord], "market_price_fail"] = 1
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
        # SYN-001 dealAmount·price_per_m2는 "만원" 단위, 온비드 감정가·낙찰가는 "원" 단위 — 원 단위로 환산
        pred_price_won = pred_pppm2 * query_rows["건물면적"].to_numpy() * 10000
        onbid.loc[query_rows.index, "market_price"] = pred_price_won

    print(f"추정 시세 미래 이웃 위반: {total_future_violations}건")
    assert total_future_violations == 0, "미래 이웃 발견 — 멈춤 지점"

    n_price_fail = int(onbid["market_price_fail"].sum())
    print(f"추정 시세 실패(학습 풀 부족/좌표 없음): {n_price_fail}/{len(onbid)} ({n_price_fail/len(onbid):.1%})")

    out_cols = [
        "cltrMngNo", "구", "동", "지번", "건물면적", "감정가", "최종 결과", "낙찰가", "낙찰가율",
        "최초 개찰일", "최종 개찰일", "최종 회차", "bad_lowest", "bad_ratio", "bad_bid",
        "lat", "lng", "coord_source", "floor_raw", "floor_feat", "floor_fail",
        "matched_build_year", "build_year_feat", "build_fail", "age_years", "house_type_bin",
        "market_price", "market_price_fail",
    ]
    out = onbid[out_cols].copy()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"저장: {OUT_PARQUET} ({len(out):,}행)")

    return out


if __name__ == "__main__":
    main()
