import csv
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

KAKAO_KEY = os.getenv("KAKAO_REST_API_KEY")

ROOT = Path(__file__).resolve().parent.parent
CACHE_CSV = ROOT / "data" / "processed" / "geocode_cache.csv"
QUALITY_CSV = ROOT / "data" / "processed" / "geocode_quality.csv"
SAMPLE_CSV = ROOT / "data" / "processed" / "geocode_quality_sample.csv"

ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
COORD2REGION_URL = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
CALL_INTERVAL_SEC = 0.1
SAMPLE_SIZE = 500
SHARED_COORD_THRESHOLD = 20


def load_cache():
    return pd.read_csv(CACHE_CSV, dtype={"geocode_key": str, "lat": str, "lng": str, "status": str})


def extract_dong(geocode_key):
    # key format: "서울특별시 {구 이름} {법정동명} {지번}"
    parts = geocode_key.split(" ")
    if len(parts) < 4:
        return None
    return parts[2]


def shared_coord_counts(df):
    return df.groupby(["lat", "lng"]).size()


def search_address(key):
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    resp = requests.get(ADDRESS_URL, headers=headers, params={"query": key}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    documents = data.get("documents", [])
    if not documents:
        return None
    return documents[0].get("address_type")


def coord2region(lat, lng):
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    resp = requests.get(COORD2REGION_URL, headers=headers, params={"x": lng, "y": lat}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    for doc in data.get("documents", []):
        if doc.get("region_type") == "B":
            return doc.get("region_3depth_name")
    return None


def main():
    df = load_cache()
    counts = shared_coord_counts(df)
    df = df.merge(
        counts.rename("shared_coord_count"), left_on=["lat", "lng"], right_index=True, how="left"
    )

    print(f"총 키: {len(df):,}")
    print(f"고유 좌표 개수: {len(counts):,}")
    top20 = counts.sort_values(ascending=False).head(20)
    print("공유 키 수 상위 20 좌표:")
    for (lat, lng), cnt in top20.items():
        example_keys = df.loc[(df["lat"] == lat) & (df["lng"] == lng), "geocode_key"].head(3).tolist()
        print(f"  ({lat}, {lng}) 공유키수={cnt} 예시={example_keys}")

    df["jibun_part"] = df["geocode_key"].str.split(" ").str[-1]
    masked_or_empty = df[
        df["jibun_part"].isna() | (df["jibun_part"].str.strip() == "") | df["jibun_part"].str.contains(r"\*", na=False)
    ]
    print(f"지번이 비었거나 마스킹된 키 개수: {len(masked_or_empty)}")

    has_address_type = "address_type" in df.columns
    print(f"캐시에 address_type 존재: {has_address_type}")

    sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    sample["dong_from_key"] = sample["geocode_key"].apply(extract_dong)

    address_types = []
    for key in sample["geocode_key"]:
        at = search_address(key)
        address_types.append(at)
        time.sleep(CALL_INTERVAL_SEC)
    sample["address_type"] = address_types

    reverse_dong = []
    for lat, lng in zip(sample["lat"], sample["lng"]):
        dong = coord2region(lat, lng)
        reverse_dong.append(dong)
        time.sleep(CALL_INTERVAL_SEC)
    sample["reverse_dong"] = reverse_dong

    sample["dong_match"] = sample["dong_from_key"] == sample["reverse_dong"]

    sample.to_csv(SAMPLE_CSV, index=False)

    print("\naddress_type 분포 (표본 500):")
    print(sample["address_type"].value_counts(dropna=False))

    match_rate = sample["dong_match"].mean()
    print(f"\n역조회 법정동 일치율 (표본 500): {match_rate:.1%}")

    exact_mask = (sample["address_type"] == "REGION_ADDR") & (sample["dong_match"])
    exact_rate = exact_mask.mean()
    print(f"exact 비율 (표본 500): {exact_rate:.1%}")

    def grade_sample_row(row):
        if row["address_type"] == "REGION_ADDR" and row["dong_match"]:
            return "exact", "address_type=REGION_ADDR, 법정동 일치"
        if row["address_type"] == "REGION" or row["shared_coord_count"] >= SHARED_COORD_THRESHOLD:
            return "region", f"address_type={row['address_type']}, 공유좌표수={row['shared_coord_count']}"
        return "unknown", f"address_type={row['address_type']}, 법정동불일치 또는 판단불가"

    sample_grades = sample.apply(lambda r: grade_sample_row(r), axis=1)
    sample["grade"] = [g[0] for g in sample_grades]
    sample["reason"] = [g[1] for g in sample_grades]

    sample_key_set = set(sample["geocode_key"])

    def grade_other_row(row):
        if row["shared_coord_count"] >= SHARED_COORD_THRESHOLD:
            return "region", f"공유좌표수={row['shared_coord_count']} (표본 밖, 3단계 기준만 적용)"
        return "unknown", "표본 밖, 판단 불가"

    other = df[~df["geocode_key"].isin(sample_key_set)].copy()
    other_grades = other.apply(lambda r: grade_other_row(r), axis=1)
    other["grade"] = [g[0] for g in other_grades]
    other["reason"] = [g[1] for g in other_grades]

    quality = pd.concat(
        [
            sample[["geocode_key", "grade", "reason"]],
            other[["geocode_key", "grade", "reason"]],
        ],
        ignore_index=True,
    )

    quality.to_csv(QUALITY_CSV, index=False)
    print(f"\n등급 분포 (전체 {len(quality):,}건):")
    print(quality["grade"].value_counts())

    print(f"\nSYN-003 저장: {QUALITY_CSV}")
    print(f"표본 상세 저장(로컬, git 제외): {SAMPLE_CSV}")


if __name__ == "__main__":
    main()
