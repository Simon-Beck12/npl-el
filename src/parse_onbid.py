import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "onbid"
ROUNDS_PARQUET = ROOT / "data" / "processed" / "onbid_rounds_raw.parquet"
ITEMS_PARQUET = ROOT / "data" / "processed" / "onbid_items.parquet"

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
    by_round = g.sort_values("pbctNsq")
    last = by_round.iloc[-1]
    by_opbd = g.sort_values("cltrOpbdDt")
    first_opbd = by_opbd.iloc[0]["cltrOpbdDt"]
    last_opbd = by_opbd.iloc[-1]["cltrOpbdDt"]
    return pd.Series({
        "소분류 코드": last["cltrUsgSclsCtgrId"],
        "소분류명": last["cltrUsgSclsCtgrNm"],
        "재산유형": last["prptDivNm"],
        "구": last["gu"],
        "동": last["dong"],
        "지번": last["jibun"],
        "건물면적": last["bldSqms"],
        "토지면적": last["landSqms"],
        "감정가": last["apslEvlAmt"],
        "최종 결과": last["pbctStatNm"],
        "최종 회차": last["pbctNsq"],
        "낙찰가": last["scfbAmt"],
        "낙찰가율": last["apslPrcCtrsScfbPrcRto"],
        "최종 회차 최저가 비율": last["lowstBidCtrsScfbPrcRto"],
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


if __name__ == "__main__":
    main()
