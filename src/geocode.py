import csv
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

KAKAO_KEY = os.getenv("KAKAO_REST_API_KEY")

ROOT = Path(__file__).resolve().parent.parent
KEYS_CSV = ROOT / "data" / "processed" / "geocode_keys.csv"
CACHE_CSV = ROOT / "data" / "processed" / "geocode_cache.csv"

URL = "https://dapi.kakao.com/v2/local/search/address.json"
CALL_INTERVAL_SEC = 0.1
CACHE_FIELDS = ["geocode_key", "lat", "lng", "status"]


def load_keys():
    with open(KEYS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["geocode_key"] for row in reader]


def load_cache():
    cache = {}
    if CACHE_CSV.exists():
        with open(CACHE_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cache[row["geocode_key"]] = row
    return cache


def append_cache_row(row):
    file_exists = CACHE_CSV.exists()
    with open(CACHE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def geocode_one(key):
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    return requests.get(URL, headers=headers, params={"query": key}, timeout=10)


def main():
    CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
    keys = load_keys()
    cache = load_cache()

    total = len(keys)
    done = len(cache)
    todo = [k for k in keys if k not in cache]

    for key in todo:
        resp = geocode_one(key)

        if resp.status_code == 429:
            print(f"HTTP 429 수신. 진행률: {done}/{total} ({done/total:.1%}). 중단합니다.")
            return

        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} 수신. 진행률: {done}/{total} ({done/total:.1%}). 중단합니다.")
            print(resp.text[:500])
            return

        data = resp.json()
        documents = data.get("documents", [])

        if documents:
            doc = documents[0]
            lat = doc.get("y", "")
            lng = doc.get("x", "")
            status = "ok"
        else:
            lat = ""
            lng = ""
            status = "no_match"

        row = {"geocode_key": key, "lat": lat, "lng": lng, "status": status}
        append_cache_row(row)
        cache[key] = row
        done += 1

        time.sleep(CALL_INTERVAL_SEC)

    print(f"지오코딩 완료: {done}/{total}")


if __name__ == "__main__":
    main()
