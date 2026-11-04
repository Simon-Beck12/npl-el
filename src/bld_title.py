"""EXP-006 단계 5~7: build_fail 물건의 주소를 카카오로 재조회해 법정동코드(b_code)를 얻고,
건축HUB 건축물대장 표제부 API로 사용승인일을 조회해 SYN-007(건축년도 보강)을 만든다.

입력: data/processed/items_deduped.parquet(단계 3~4 결과, onbid_features_v3.py가 생성)의
build_fail=1 물건. 기존 src/onbid_features.py, src/parse_onbid.py는 수정하지 않는다.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
BCODE_CSV = ROOT / "data" / "processed" / "geocode_bcode_onbid.csv"
BLD_RAW_DIR = ROOT / "data" / "raw" / "bld"
OUT_CSV = ROOT / "data" / "processed" / "bld_use_apr.csv"

KAKAO_URL = os.getenv("KAKAO_GEOCODE_URL")
KAKAO_KEY = os.getenv("KAKAO_REST_API_KEY")
BLD_TITLE_URL = os.getenv("BLD_TITLE_URL")
SERVICE_KEY = os.getenv("DATA_GO_KR_API_KEY")

KAKAO_INTERVAL_SEC = 0.1
BLD_INTERVAL_SEC = 0.2
MAX_CALLS = 1000

GOOD_PURPS = ["공동주택", "다세대", "연립"]

RETRY_DELAYS_SEC = [5, 15, 30]
MAX_CONSECUTIVE_503 = 5


def load_bcode_cache():
    if BCODE_CSV.exists():
        return pd.read_csv(BCODE_CSV, dtype=str)
    return pd.DataFrame(columns=["geocode_key", "b_code", "main_address_no", "sub_address_no", "mountain_yn", "addr_fail"])


def kakao_addr_lookup(query):
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    resp = requests.get(KAKAO_URL, headers=headers, params={"query": query}, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"카카오 API HTTP {resp.status_code} — 멈춤. 응답: {resp.text[:300]}")
    data = resp.json()
    documents = data.get("documents", [])
    if not documents:
        return {"b_code": "", "main_address_no": "", "sub_address_no": "", "mountain_yn": "", "addr_fail": 1}
    doc = documents[0]
    if doc.get("address_type") != "REGION_ADDR":
        return {"b_code": "", "main_address_no": "", "sub_address_no": "", "mountain_yn": "", "addr_fail": 1}
    addr = doc.get("address", {})
    return {
        "b_code": addr.get("b_code", ""),
        "main_address_no": addr.get("main_address_no", ""),
        "sub_address_no": addr.get("sub_address_no", ""),
        "mountain_yn": addr.get("mountain_yn", ""),
        "addr_fail": 0,
    }


def step6_resolve_bcode(target):
    """target: cltrMngNo, 구, 동, 지번 열이 있는 df. geocode_key별 캐시 재사용, 없으면 카카오 신규 호출."""
    cache = load_bcode_cache()
    cache_idx = cache.set_index("geocode_key") if len(cache) else cache

    target = target.copy()
    target["geocode_key"] = "서울특별시 " + target["구"] + " " + target["동"].str.strip() + " " + target["지번"].str.strip()
    keys = target["geocode_key"].unique()
    missing = [k for k in keys if k not in cache_idx.index]

    print(f"b_code 조회 대상 고유 주소: {len(keys)}개, 캐시 재사용 {len(keys) - len(missing)}개, 신규 호출 {len(missing)}개")
    if len(missing) > MAX_CALLS:
        raise RuntimeError(f"카카오 신규 호출 {len(missing)}건이 한도 {MAX_CALLS}건을 넘음 — 멈춤")

    new_rows = []
    for key in missing:
        info = kakao_addr_lookup(key)
        row = {"geocode_key": key, **info}
        new_rows.append(row)
        time.sleep(KAKAO_INTERVAL_SEC)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        file_exists = BCODE_CSV.exists()
        new_df.to_csv(BCODE_CSV, mode="a", header=not file_exists, index=False)
        cache = pd.concat([cache, new_df], ignore_index=True) if len(cache) else new_df

    print(f"카카오 신규 호출 완료: {len(new_rows)}건")
    merged = target.merge(cache.drop_duplicates(subset="geocode_key"), on="geocode_key", how="left")
    return merged


def fetch_bld_title(sigungu_cd, bjdong_cd, plat_gb_cd, bun, ji):
    """반환: (data 또는 None, called(bool), status: "ok"/"cached"/"503_exhausted").
    503 수신 시 요청당 최대 3회 재시도(대기 5초/15초/30초). 503이 아닌 다른 오류는 즉시 멈춤(RuntimeError)."""
    fname = f"{sigungu_cd}_{bjdong_cd}_{bun}_{ji}.json"
    fpath = BLD_RAW_DIR / fname
    if fpath.exists():
        return json.loads(fpath.read_text(encoding="utf-8")), False, "cached"

    params = {
        "serviceKey": SERVICE_KEY,
        "sigunguCd": sigungu_cd,
        "bjdongCd": bjdong_cd,
        "platGbCd": plat_gb_cd,
        "bun": bun,
        "ji": ji,
        "_type": "json",
        "numOfRows": 100,
        "pageNo": 1,
    }
    last_resp = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS_SEC):
        if attempt > 0:
            print(f"  503 재시도 {attempt}/{len(RETRY_DELAYS_SEC)} ({delay}초 대기 후): {sigungu_cd}_{bjdong_cd}_{bun}_{ji}")
            time.sleep(delay)
        resp = requests.get(BLD_TITLE_URL, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            BLD_RAW_DIR.mkdir(parents=True, exist_ok=True)
            fpath.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data, True, "ok"
        if resp.status_code == 503:
            last_resp = resp
            continue
        raise RuntimeError(f"건축HUB API HTTP {resp.status_code} — 멈춤. 응답: {resp.text[:300]}")

    print(f"  503 재시도 {len(RETRY_DELAYS_SEC)}회 모두 실패: {sigungu_cd}_{bjdong_cd}_{bun}_{ji}. "
          f"응답: {last_resp.text[:200] if last_resp is not None else ''}")
    return None, True, "503_exhausted"


def pick_build_year(items):
    if not items:
        return None, None, None, None, 1
    good = [it for it in items if any(p in (it.get("mainPurpsCdNm") or "") for p in GOOD_PURPS)]
    pool = good if good else items
    with_date = [it for it in pool if (it.get("useAprDay") or "").strip()]
    if not with_date:
        return None, None, None, None, 1
    best = max(with_date, key=lambda it: it["useAprDay"])
    use_apr_day = best["useAprDay"]
    return use_apr_day, int(use_apr_day[:4]), best.get("mainPurpsCdNm"), best.get("regstrKindCdNm"), 0


def step7_resolve_build_year(resolved):
    """resolved: step6 출력(geocode_key, b_code, main_address_no, sub_address_no, mountain_yn, addr_fail 포함)."""
    eligible = resolved[resolved["addr_fail"] != 1].copy()
    eligible = eligible[eligible["b_code"].notna() & (eligible["b_code"].astype(str).str.len() >= 10)].copy()
    print(f"표제부 조회 대상(addr_fail=0, b_code 10자리): {len(eligible)}/{len(resolved)}")

    eligible["sigunguCd"] = eligible["b_code"].astype(str).str[:5]
    eligible["bjdongCd"] = eligible["b_code"].astype(str).str[5:10]
    eligible["platGbCd"] = eligible["mountain_yn"].apply(lambda v: "1" if v == "Y" else "0")
    eligible["bun"] = eligible["main_address_no"].fillna("0").astype(str).str.extract(r"(\d+)")[0].fillna("0").str.zfill(4)
    sub = eligible["sub_address_no"].fillna("").astype(str).str.extract(r"(\d+)")[0]
    eligible["ji"] = sub.fillna("0").replace("0", "0").str.zfill(4)
    eligible.loc[sub.isna() | (sub == "") | (sub == "0"), "ji"] = "0000"

    req_keys = eligible[["sigunguCd", "bjdongCd", "platGbCd", "bun", "ji"]].drop_duplicates()
    print(f"건축HUB 조회 고유 요청 수: {len(req_keys)}건 (물건 {len(eligible)}건)")
    if len(req_keys) > MAX_CALLS:
        raise RuntimeError(f"건축HUB 요청 {len(req_keys)}건이 한도 {MAX_CALLS}건을 넘음 — 멈춤")

    results = {}
    n_called = 0
    consecutive_503 = 0
    aborted = False
    for idx, (_, r) in enumerate(req_keys.iterrows()):
        key = (r["sigunguCd"], r["bjdongCd"], r["platGbCd"], r["bun"], r["ji"])
        if aborted:
            results[key] = {"use_apr_day": None, "build_year": None, "main_purps": None,
                             "regstr_kind": None, "n_items": 0, "bld_fail": 1, "source": "대체경로(중단후미시도)"}
            continue

        data, called, status = fetch_bld_title(*key)
        if called:
            n_called += 1

        if status == "503_exhausted":
            consecutive_503 += 1
            print(f"연속 503(서로 다른 요청) {consecutive_503}/{MAX_CONSECUTIVE_503}")
            results[key] = {"use_apr_day": None, "build_year": None, "main_purps": None,
                             "regstr_kind": None, "n_items": 0, "bld_fail": 1, "source": "대체경로(503)"}
            if consecutive_503 >= MAX_CONSECUTIVE_503:
                print(f"[전환] 503이 서로 다른 요청에서 연속 {MAX_CONSECUTIVE_503}건 — 건축HUB 중단, 남은 요청은 대체 경로로 전환")
                aborted = True
            if called:
                time.sleep(BLD_INTERVAL_SEC)
            continue

        consecutive_503 = 0
        body = data.get("response", {}).get("body", {})
        items_obj = body.get("items")
        item_list = []
        if items_obj:
            raw_items = items_obj.get("item", [])
            item_list = raw_items if isinstance(raw_items, list) else [raw_items]
        use_apr_day, build_year, main_purps, regstr_kind, bld_fail = pick_build_year(item_list)
        results[key] = {
            "use_apr_day": use_apr_day, "build_year": build_year,
            "main_purps": main_purps, "regstr_kind": regstr_kind,
            "n_items": len(item_list), "bld_fail": bld_fail,
            "source": "건축HUB" if bld_fail == 0 else "건축HUB(사용승인일없음)",
        }
        if called:
            time.sleep(BLD_INTERVAL_SEC)

    print(f"건축HUB 신규 호출: {n_called}건 (캐시 재사용 {len(req_keys) - n_called}건), 전환 발생: {aborted}")

    out_rows = []
    for _, r in eligible.iterrows():
        key = (r["sigunguCd"], r["bjdongCd"], r["platGbCd"], r["bun"], r["ji"])
        res = results[key]
        out_rows.append({
            "cltrMngNo": r["cltrMngNo"], "sigunguCd": r["sigunguCd"], "bjdongCd": r["bjdongCd"],
            "platGbCd": r["platGbCd"], "bun": r["bun"], "ji": r["ji"],
            "use_apr_day": res["use_apr_day"], "build_year": res["build_year"],
            "main_purps": res["main_purps"], "regstr_kind": res["regstr_kind"],
            "n_items": res["n_items"], "bld_fail": res["bld_fail"], "source": res["source"],
        })

    out = pd.DataFrame(out_rows)
    n_success = int((out["bld_fail"] == 0).sum())
    n_fail = len(out) - n_success
    print(f"표제부 사용승인일 확보: {n_success}/{len(out)} ({n_success/len(out):.1%} of 조회 대상)")
    return out, aborted, n_success, n_fail


def main(target_df):
    """target_df: cltrMngNo, 구, 동, 지번 열이 있는 build_fail=1 물건 목록."""
    resolved = step6_resolve_bcode(target_df)
    addr_fail_n = int((resolved["addr_fail"] == 1).sum())
    print(f"addr_fail: {addr_fail_n}/{len(resolved)} ({addr_fail_n/len(resolved):.1%})")

    out, aborted, n_success_req, n_fail_req = step7_resolve_build_year(resolved)

    # addr_fail=1이라 7단계 대상에서 빠진 물건도 기록(bld_fail=1, 대체경로로 표시)
    addr_fail_ids = resolved[resolved["addr_fail"] == 1]["cltrMngNo"]
    if len(addr_fail_ids):
        extra = pd.DataFrame({"cltrMngNo": addr_fail_ids})
        for col in ["sigunguCd", "bjdongCd", "platGbCd", "bun", "ji", "use_apr_day", "build_year", "main_purps", "regstr_kind"]:
            extra[col] = None
        extra["n_items"] = 0
        extra["bld_fail"] = 1
        extra["source"] = "대체경로(addr_fail)"
        out = pd.concat([out, extra], ignore_index=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"저장: {OUT_CSV} ({len(out)}행)")

    overall_success = int((out["bld_fail"] == 0).sum())
    overall_fail = len(out) - overall_success
    overall_rate = overall_success / len(target_df)
    print(f"전체 build_fail 대상 중 표제부 성공률: {overall_success}/{len(target_df)} ({overall_rate:.1%}), "
          f"전환 발생(503 연속 {MAX_CONSECUTIVE_503}건): {aborted}, 성공 {overall_success}건/실패 {overall_fail}건")
    return out, overall_rate, aborted, overall_success, overall_fail


if __name__ == "__main__":
    bf = pd.read_csv(ROOT / "data" / "processed" / "exp006_build_fail_targets.csv", dtype=str)
    main(bf)
