import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PARQUET = ROOT / "data" / "processed" / "rh_trade_clean.parquet"
GEOCODE_CACHE_CSV = ROOT / "data" / "processed" / "geocode_cache.csv"
LAWD_CSV = ROOT / "docs" / "seoul_lawd_cd.csv"

# 서울시청 좌표를 평면 투영 기준점으로 쓴다 (위도·경도 → km 단위 변환)
REF_LAT = 37.5665
REF_LNG = 126.9780
KM_PER_LAT_DEG = 110.574
KM_PER_LNG_DEG = 111.320

HOUSE_TYPE_MAP = {"다세대": 1.0, "연립": 0.0, "연립다세대": 0.5}


def build_geocode_key(df, lawd_map):
    gu_name = df["sggCd"].astype(str).map(lawd_map)
    return "서울특별시 " + gu_name + " " + df["umdNm"].str.strip() + " " + df["jibun"].str.strip()


def load_merged():
    df = pd.read_parquet(CLEAN_PARQUET)
    lawd_map = pd.read_csv(LAWD_CSV, dtype={"LAWD_CD": str}).set_index("LAWD_CD")["gu_name"]
    df["gu_name"] = df["sggCd"].astype(str).map(lawd_map)
    df["geocode_key"] = build_geocode_key(df, lawd_map)

    geocode = pd.read_csv(GEOCODE_CACHE_CSV)
    geocode = geocode[geocode["status"] == "ok"][["geocode_key", "lat", "lng"]]
    merged = df.merge(geocode, on="geocode_key", how="left")
    return merged


def build_features(df):
    df = df.copy()
    missing_coord = df["lat"].isna() | df["lng"].isna()
    n_missing = int(missing_coord.sum())
    print(f"좌표 없는 행: {n_missing:,} / {len(df):,} ({n_missing / len(df):.2%})")

    lat_rad = math.radians(REF_LAT)
    df["loc_x_km"] = (df["lng"] - REF_LNG) * KM_PER_LNG_DEG * math.cos(lat_rad)
    df["loc_y_km"] = (df["lat"] - REF_LAT) * KM_PER_LAT_DEG

    df["log_area"] = np.log(df["excluUseAr"])
    df["age_years"] = df["dealYear"].astype(int) - df["buildYear"].astype(int)
    df["floor_feat"] = df["floor"].astype(float)
    df["house_type_bin"] = df["houseType"].map(HOUSE_TYPE_MAP)

    return df, n_missing


FEATURE_COLUMNS = ["loc_x_km", "loc_y_km", "log_area", "age_years", "floor_feat", "house_type_bin"]
FEATURE_GROUPS = {
    "loc": ["loc_x_km", "loc_y_km"],
    "area": ["log_area"],
    "age": ["age_years"],
    "floor": ["floor_feat", "house_type_bin"],
}


def main():
    merged = load_merged()
    featured, n_missing = build_features(merged)
    print(f"전체 행: {len(featured):,}")
    return featured, n_missing


if __name__ == "__main__":
    main()
