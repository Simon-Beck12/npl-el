import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "rh_trade"
RAW_PARQUET = ROOT / "data" / "processed" / "rh_trade_raw.parquet"
CLEAN_PARQUET = ROOT / "data" / "processed" / "rh_trade_clean.parquet"

FIELDS = [
    "buildYear", "buyerGbn", "cdealDay", "cdealType", "dealAmount",
    "dealDay", "dealMonth", "dealYear", "dealingGbn", "estateAgentSggNm",
    "excluUseAr", "floor", "houseType", "jibun", "landAr", "mhouseNm",
    "rgstDate", "sggCd", "slerGbn", "umdNm",
]


def parse_all_xml():
    rows = []
    for xml_path in sorted(RAW_DIR.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        for item in root.findall(".//item"):
            row = {field: item.findtext(field) for field in FIELDS}
            rows.append(row)
    return pd.DataFrame(rows)


def make_deal_id(row):
    contract_date = f"{row['dealYear']}{int(row['dealMonth']):02d}{int(row['dealDay']):02d}"
    key = "|".join([
        str(row["sggCd"]),
        contract_date,
        str(row["jibun"]),
        str(row["floor"]),
        str(row["excluUseAr"]),
        str(row["dealAmount"]),
    ])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def clean(df_raw):
    df = df_raw.copy()

    df["cdealType"] = df["cdealType"].fillna("").str.strip()
    df["dealAmount"] = df["dealAmount"].str.replace(",", "", regex=False)
    df["dealAmount"] = pd.to_numeric(df["dealAmount"], errors="coerce")
    df["excluUseAr"] = pd.to_numeric(df["excluUseAr"], errors="coerce")
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")

    df = df[df["cdealType"] == ""].copy()
    df = df.dropna(subset=["dealAmount", "excluUseAr"]).copy()

    df["dealAmount"] = df["dealAmount"].astype("int64")

    df["price_per_m2"] = df["dealAmount"] / df["excluUseAr"]

    low, high = df["price_per_m2"].quantile([0.01, 0.99])
    df["outlier_flag"] = ((df["price_per_m2"] < low) | (df["price_per_m2"] > high)).astype(int)

    df["direct_flag"] = (df["dealingGbn"].fillna("").str.strip() == "직거래").astype(int)
    df["basement_flag"] = (df["floor"] < 0).astype(int)

    df["deal_id"] = df.apply(make_deal_id, axis=1)

    return df


def main():
    RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    df_raw = parse_all_xml()
    df_raw.to_parquet(RAW_PARQUET, index=False)

    df_clean = clean(df_raw)
    df_clean.to_parquet(CLEAN_PARQUET, index=False)

    print(f"raw rows: {len(df_raw)}")
    print(f"clean rows: {len(df_clean)}")
    print(f"제거된 행(해제거래+결측): {len(df_raw) - len(df_clean)}")


if __name__ == "__main__":
    main()
