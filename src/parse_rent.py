"""EXP-006 단계 10: DS-03(전월세 원본 XML) 파싱·정제 → SYN-006.

docs/fields_rent.md 확인 결과 cdealType/cdealDay(계약 해제 표시) 필드 자체가 없어
해제 거래 제거는 해당 없음(0건)이다. src/parse_clean.py는 수정하지 않는다.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "rh_rent"
RAW_PARQUET = ROOT / "data" / "processed" / "rh_rent_raw.parquet"
CLEAN_PARQUET = ROOT / "data" / "processed" / "rh_rent_clean.parquet"

FIELDS = [
    "buildYear", "contractTerm", "contractType", "dealDay", "dealMonth", "dealYear",
    "deposit", "excluUseAr", "floor", "houseType", "jibun", "mhouseNm",
    "monthlyRent", "preDeposit", "preMonthlyRent", "sggCd", "umdNm", "useRRRight",
]


def parse_all_xml():
    rows = []
    for xml_path in sorted(RAW_DIR.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        for item in root.findall(".//item"):
            row = {field: item.findtext(field) for field in FIELDS}
            rows.append(row)
    return pd.DataFrame(rows)


def clean(df_raw):
    df = df_raw.copy()

    df["deposit"] = df["deposit"].str.replace(",", "", regex=False)
    df["deposit"] = pd.to_numeric(df["deposit"], errors="coerce")
    df["monthlyRent"] = pd.to_numeric(df["monthlyRent"], errors="coerce")
    df["excluUseAr"] = pd.to_numeric(df["excluUseAr"], errors="coerce")
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")

    df = df.dropna(subset=["deposit", "excluUseAr"]).copy()
    df["deposit"] = df["deposit"].astype("int64")

    df["contract_date"] = pd.to_datetime(
        df["dealYear"].astype(str) + "-" + df["dealMonth"].astype(str).str.zfill(2)
        + "-" + df["dealDay"].astype(str).str.zfill(2)
    )

    return df


def main():
    RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    df_raw = parse_all_xml()
    df_raw.to_parquet(RAW_PARQUET, index=False)

    df_clean = clean(df_raw)
    df_clean.to_parquet(CLEAN_PARQUET, index=False)

    print(f"raw rows: {len(df_raw)}")
    print(f"clean rows: {len(df_clean)}")
    print(f"제거된 행(결측 deposit·excluUseAr): {len(df_raw) - len(df_clean)}")
    print("계약 해제 표시 필드 없음 — 해제 거래 제거 해당 없음(0건)")

    return df_raw, df_clean


if __name__ == "__main__":
    main()
