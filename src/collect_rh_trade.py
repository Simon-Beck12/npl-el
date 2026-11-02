import csv
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

SERVICE_KEY = os.getenv("DATA_GO_KR_API_KEY")
RH_TRADE_URL = os.getenv("RH_TRADE_URL")

ROOT = Path(__file__).resolve().parent.parent
LAWD_CSV = ROOT / "docs" / "seoul_lawd_cd.csv"
RAW_DIR = ROOT / "data" / "raw" / "rh_trade"
LOG_PATH = ROOT / "data" / "logs" / "collect_errors.log"

START_YM = "202411"
END_YM = "202610"
NUM_OF_ROWS = 1000
CALL_INTERVAL_SEC = 0.2
MAX_CONSECUTIVE_FAILURES = 5


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


def load_lawd_codes():
    codes = []
    with open(LAWD_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            codes.append((row["gu_name"], row["LAWD_CD"]))
    return codes


def log_error(message):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def fetch_page(lawd_cd, deal_ymd, page_no):
    params = {
        "serviceKey": SERVICE_KEY,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ymd,
        "numOfRows": NUM_OF_ROWS,
        "pageNo": page_no,
    }
    resp = requests.get(RH_TRADE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_result_code(xml_text):
    root = ET.fromstring(xml_text)
    header = root.find("header")
    if header is not None:
        return header.findtext("resultCode", default=""), header.findtext("resultMsg", default="")
    cmm_header = root.find("cmmMsgHeader")
    if cmm_header is not None:
        return "ERROR", cmm_header.findtext("errMsg", default="unknown error")
    return "ERROR", "no header in response"


def parse_total_count(xml_text):
    root = ET.fromstring(xml_text)
    body = root.find("body")
    if body is None:
        return 0
    return int(body.findtext("totalCount", default="0") or "0")


def collect():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    consecutive_failures = 0
    months = month_range(START_YM, END_YM)
    lawd_codes = load_lawd_codes()

    for gu_name, lawd_cd in lawd_codes:
        for deal_ymd in months:
            page_no = 1
            while True:
                out_path = RAW_DIR / f"{lawd_cd}_{deal_ymd}_p{page_no}.xml"
                called = False
                if out_path.exists():
                    xml_text = out_path.read_text(encoding="utf-8")
                else:
                    try:
                        xml_text = fetch_page(lawd_cd, deal_ymd, page_no)
                        called = True
                    except requests.RequestException as exc:
                        consecutive_failures += 1
                        log_error(f"{lawd_cd} {deal_ymd} p{page_no} REQUEST_ERROR {exc}")
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            print(f"연속 {consecutive_failures}회 실패로 중단: {lawd_cd}({gu_name}) {deal_ymd} p{page_no}")
                            return
                        break

                try:
                    result_code, result_msg = parse_result_code(xml_text)
                except ET.ParseError as exc:
                    consecutive_failures += 1
                    log_error(f"{lawd_cd} {deal_ymd} p{page_no} PARSE_ERROR {exc}")
                    if called:
                        time.sleep(CALL_INTERVAL_SEC)
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"연속 {consecutive_failures}회 실패로 중단: {lawd_cd}({gu_name}) {deal_ymd} p{page_no}")
                        return
                    break

                if result_code != "000":
                    log_error(f"{lawd_cd} {deal_ymd} p{page_no} resultCode={result_code} {result_msg}")
                    consecutive_failures += 1
                    if called:
                        time.sleep(CALL_INTERVAL_SEC)
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"연속 {consecutive_failures}회 실패로 중단: {lawd_cd}({gu_name}) {deal_ymd} p{page_no}")
                        return
                    break

                consecutive_failures = 0
                if not out_path.exists():
                    out_path.write_text(xml_text, encoding="utf-8")
                if called:
                    time.sleep(CALL_INTERVAL_SEC)

                total_count = parse_total_count(xml_text)
                if page_no * NUM_OF_ROWS >= total_count:
                    break
                page_no += 1

    print("수집 완료")


if __name__ == "__main__":
    collect()
