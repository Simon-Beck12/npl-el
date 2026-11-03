import calendar
import json
import os
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

SERVICE_KEY = os.getenv("DATA_GO_KR_API_KEY")
ONBID_URL = os.getenv("ONBID_ITEM_RESULT_LIST_URL")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "onbid"
LOG_PATH = ROOT / "data" / "logs" / "onbid_errors.log"
CALLS_LOG_PATH = ROOT / "data" / "logs" / "onbid_calls.log"

START_YM = "202411"
END_YM = "202610"
NUM_OF_ROWS = 1000
CALL_INTERVAL_SEC = 0.3
MAX_CONSECUTIVE_FAILURES = 5
DAILY_CALL_LIMIT = 900

FIXED_PARAMS = {
    "cltrTypeCd": "0001",
    "dspsMthodCd": "0001",
    "cltrUsgMclsCtgrId": "10200",
    "lctnSdnm": "서울특별시",
    "prptDivCd": "0007,0010,0005,0002,0003,0006,0008,0011,0013",
    "resultType": "json",
}


def month_range(start_ym, end_ym):
    start_y, start_m = int(start_ym[:4]), int(start_ym[4:])
    end_y, end_m = int(end_ym[:4]), int(end_ym[4:])
    months = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        months.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def month_bounds(ym):
    year, month = int(ym[:4]), int(ym[4:])
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}{month:02d}01", f"{year:04d}{month:02d}{last_day:02d}"


def log_error(message):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def today_call_count():
    if not CALLS_LOG_PATH.exists():
        return 0
    today = date.today().isoformat()
    count = 0
    with open(CALLS_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith(today):
                count += 1
    return count


def log_call(ym, page_no):
    CALLS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = date.today().isoformat()
    with open(CALLS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {ym} p{page_no}\n")


def fetch_page(opbd_start, opbd_end, page_no):
    params = dict(
        FIXED_PARAMS,
        serviceKey=SERVICE_KEY,
        opbdDtStart=opbd_start,
        opbdDtEnd=opbd_end,
        numOfRows=NUM_OF_ROWS,
        pageNo=page_no,
    )
    resp = requests.get(ONBID_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def collect():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    consecutive_failures = 0
    months = month_range(START_YM, END_YM)
    calls_today = today_call_count()
    completed_months = 0

    for ym in months:
        opbd_start, opbd_end = month_bounds(ym)
        page_no = 1
        month_done = True
        while True:
            out_path = RAW_DIR / f"{ym}_p{page_no}.json"
            called = False
            if out_path.exists():
                data = json.loads(out_path.read_text(encoding="utf-8"))
            else:
                if calls_today >= DAILY_CALL_LIMIT:
                    print(f"오늘 호출 수가 {DAILY_CALL_LIMIT}건에 닿았습니다. 완료 월: {completed_months}/{len(months)}")
                    return
                try:
                    data = fetch_page(opbd_start, opbd_end, page_no)
                    called = True
                    calls_today += 1
                    log_call(ym, page_no)
                except requests.RequestException as exc:
                    consecutive_failures += 1
                    log_error(f"{ym} p{page_no} REQUEST_ERROR {exc}")
                    month_done = False
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"연속 {consecutive_failures}회 실패로 중단: {ym} p{page_no}")
                        return
                    break

            header = data.get("header", {})
            result_code = header.get("resultCode", "")
            result_msg = header.get("resultMsg", "")

            if result_code not in ("00", "03"):
                log_error(f"{ym} p{page_no} resultCode={result_code} {result_msg}")
                consecutive_failures += 1
                month_done = False
                if called:
                    time.sleep(CALL_INTERVAL_SEC)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"연속 {consecutive_failures}회 실패로 중단: {ym} p{page_no}")
                    return
                break

            consecutive_failures = 0
            if not out_path.exists():
                out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            if called:
                time.sleep(CALL_INTERVAL_SEC)

            if result_code == "03":
                break

            body = data.get("body", {})
            total_count = int(body.get("totalCount", 0) or 0)
            if page_no * NUM_OF_ROWS >= total_count:
                break
            page_no += 1

        if month_done:
            completed_months += 1

    print(f"수집 완료 (완료 월: {completed_months}/{len(months)})")


if __name__ == "__main__":
    collect()
