import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "onbid"
ROUNDS_PARQUET = ROOT / "data" / "processed" / "onbid_rounds_raw.parquet"
ITEMS_PARQUET = ROOT / "data" / "processed" / "onbid_items.parquet"
ITEMS_PARQUET_V2 = ROOT / "data" / "processed" / "onbid_items_v2.parquet"

ADDR_PATTERN = re.compile(r"서울(?:특별시|시)?\s*([가-힣]+구)\s+([가-힣]+동\d*가?)\s+(\d+(?:-\d+)?)")

VILLA_CODES = {"10203", "10206", "10209"}
UH_CODE = "10208"


def load_rounds():
    rows = []
    for fp in sorted(RAW_DIR.glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        items = data.get("body", {}).get("items")
        if not items:
            continue
        item_list = items.get("item", [])
        if isinstance(item_list, dict):
            item_list = [item_list]
        rows.extend(item_list)
    return pd.DataFrame(rows)


def parse_address(name):
    m = ADDR_PATTERN.search(name or "")
    if not m:
        return pd.Series({"gu": None, "dong": None, "jibun": None, "parse_fail": 1})
    gu, dong, jibun = m.groups()
    return pd.Series({"gu": gu, "dong": dong, "jibun": jibun, "parse_fail": 0})


def build_rounds(df):
    df = df.copy()
    df["lowstBidPrcIndctCont"] = pd.to_numeric(df["lowstBidPrcIndctCont"], errors="coerce")
    df["scfbAmt"] = pd.to_numeric(df["scfbAmt"], errors="coerce")
    df["apslEvlAmt"] = pd.to_numeric(df["apslEvlAmt"], errors="coerce")
    df["apslPrcCtrsScfbPrcRto"] = pd.to_numeric(df["apslPrcCtrsScfbPrcRto"], errors="coerce")
    df["pbctNsq"] = pd.to_numeric(df["pbctNsq"], errors="coerce").astype("Int64")
    df["ym"] = df["cltrOpbdDt"].str[:6]

    addr = df["onbidCltrNm"].apply(parse_address)
    df = pd.concat([df, addr], axis=1)

    df["bad_lowest"] = (df["lowstBidPrcIndctCont"] > df["apslEvlAmt"]).astype(int)

    has_bid = df["scfbAmt"].notna()
    computed_ratio = df["scfbAmt"] / df["apslEvlAmt"] * 100
    ratio_diff = (computed_ratio - df["apslPrcCtrsScfbPrcRto"]).abs()
    df["bad_ratio"] = (has_bid & (ratio_diff > 1)).astype(int)
    df["bad_bid"] = (has_bid & (df["scfbAmt"] < df["lowstBidPrcIndctCont"])).astype(int)

    df["villa_flag"] = df["cltrUsgSclsCtgrId"].isin(VILLA_CODES).astype(int)
    df["uh_flag"] = (df["cltrUsgSclsCtgrId"] == UH_CODE).astype(int)

    return df


def agg_item(g):
    """2026-11-04 정정: 감정가는 물건 내 결측이 아닌 마지막 회차 값을 쓴다(모두 결측일 때만 결측).
    "최종 회차 최저가 비율"은 마지막 회차 최저입찰가÷감정가×100으로 직접 계산한다(기존 열은
    "최종 회차 낙찰가/최저가 비율"로 이름을 바꿔 남긴다 — API 제공 lowstBidCtrsScfbPrcRto, 낙찰가÷최저가)."""
    by_round = g.sort_values("pbctNsq")
    last = by_round.iloc[-1]
    by_opbd = g.sort_values("cltrOpbdDt")
    first_opbd = by_opbd.iloc[0]["cltrOpbdDt"]
    last_opbd = by_opbd.iloc[-1]["cltrOpbdDt"]

    appr_valid = by_round[by_round["apslEvlAmt"].notna()]
    appr_value = appr_valid.iloc[-1]["apslEvlAmt"] if len(appr_valid) > 0 else np.nan

    if pd.notna(appr_value) and appr_value > 0 and pd.notna(last["lowstBidPrcIndctCont"]):
        lowest_ratio = last["lowstBidPrcIndctCont"] / appr_value * 100
    else:
        lowest_ratio = np.nan

    return pd.Series({
        "소분류 코드": last["cltrUsgSclsCtgrId"],
        "소분류명": last["cltrUsgSclsCtgrNm"],
        "재산유형": last["prptDivNm"],
        "구": last["gu"],
        "동": last["dong"],
        "지번": last["jibun"],
        "건물면적": last["bldSqms"],
        "토지면적": last["landSqms"],
        "감정가": appr_value,
        "최종 결과": last["pbctStatNm"],
        "최종 회차": last["pbctNsq"],
        "낙찰가": last["scfbAmt"],
        "낙찰가율": last["apslPrcCtrsScfbPrcRto"],
        "최종 회차 최저가 비율": lowest_ratio,
        "최종 회차 낙찰가/최저가 비율": last["lowstBidCtrsScfbPrcRto"],
        "유효 입찰자 수": last["vldBddrNope"],
        "최초 개찰일": first_opbd,
        "최종 개찰일": last_opbd,
        "parse_fail": int(g["parse_fail"].max()),
        "bad_lowest": int(g["bad_lowest"].max()),
        "bad_ratio": int(g["bad_ratio"].max()),
        "bad_bid": int(g["bad_bid"].max()),
        "villa_flag": int(g["villa_flag"].max()),
        "uh_flag": int(g["uh_flag"].max()),
    })


def build_items(rounds_df):
    items = rounds_df.groupby("cltrMngNo").apply(agg_item, include_groups=False)
    return items.reset_index()


def main():
    rounds_df = load_rounds()
    print(f"원본 회차 행 수: {len(rounds_df):,}")

    rounds_built = build_rounds(rounds_df)
    ROUNDS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    rounds_built.to_parquet(ROUNDS_PARQUET, index=False)
    print(f"저장: {ROUNDS_PARQUET} ({len(rounds_built):,}행)")

    items = build_items(rounds_built)
    items.to_parquet(ITEMS_PARQUET, index=False)
    print(f"저장: {ITEMS_PARQUET} ({len(items):,}행, 물건 수)")

    return rounds_built, items


def main_v2():
    """2026-11-04 정정: 기존 onbid_rounds_raw.parquet(회차 데이터, 변경 없음)을 그대로 읽어
    고친 agg_item()으로 onbid_items_v2.parquet만 새로 만든다. 기존 v1 파일은 덮어쓰지 않는다."""
    rounds_built = pd.read_parquet(ROUNDS_PARQUET)
    print(f"기존 회차 데이터 읽음: {ROUNDS_PARQUET} ({len(rounds_built):,}행)")

    items_v1 = pd.read_parquet(ITEMS_PARQUET)
    items_v2 = build_items(rounds_built)

    recovered = items_v1["감정가"].isna() & items_v2.set_index("cltrMngNo").loc[items_v1["cltrMngNo"], "감정가"].notna().to_numpy()
    print(f"감정가 복구 건수: {int(recovered.sum())}건 (v1 결측 {items_v1['감정가'].isna().sum()}건 중)")

    ITEMS_PARQUET_V2.parent.mkdir(parents=True, exist_ok=True)
    items_v2.to_parquet(ITEMS_PARQUET_V2, index=False)
    print(f"저장: {ITEMS_PARQUET_V2} ({len(items_v2):,}행, 물건 수)")

    return items_v2


if __name__ == "__main__":
    main()
