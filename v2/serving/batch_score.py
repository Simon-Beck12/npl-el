"""EXP-011 8단계: 물건 목록 CSV를 한 번에 시세·회수액으로 계산하는 배치 스크립트.

입력 CSV 컬럼: lat, lng, excluUseAr, floor, buildYear, dealYear, dealMonth,
(선택) 구, umdNm, jibun, deposit, ead, months, rho_scenario(비관/기본/낙관)
출력 CSV 컬럼: 입력 전체 + 시세, S, R, PV, LGD
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import RHO_SCENARIOS  # noqa: E402
from model_loader import load_model  # noqa: E402
from recovery import compute_recovery_v2  # noqa: E402
from valuation_core import build_target_row, predict_valuation  # noqa: E402

DEFAULT_INPUT = Path(__file__).resolve().parent / "examples" / "sample_input.csv"


def score_row(row, ref, booster):
    gu_name = row.get("구") if pd.notna(row.get("구")) else None
    umdNm = row.get("umdNm") if pd.notna(row.get("umdNm")) else None
    jibun = row.get("jibun") if pd.notna(row.get("jibun")) else None
    target = build_target_row(
        lat=row["lat"], lng=row["lng"], excluUseAr=row["excluUseAr"], floor=row["floor"],
        buildYear=row["buildYear"], dealYear=row["dealYear"], dealMonth=row["dealMonth"],
        gu_name=gu_name, umdNm=umdNm, jibun=jibun,
    )
    val = predict_valuation(target, ref, booster)
    rho = RHO_SCENARIOS[row["rho_scenario"]]
    rec = compute_recovery_v2(
        market_price=val["market_price"], rho=rho,
        deposit=row["deposit"], ead=row["ead"], months=row["months"],
    )
    return pd.Series({"시세": val["market_price"], "S": rec["S"], "R": rec["R"], "PV": rec["PV"], "LGD": rec["LGD"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_scored.csv")

    df = pd.read_csv(input_path)
    booster, ref, meta = load_model()
    scored = df.apply(lambda row: score_row(row, ref, booster), axis=1)
    out = pd.concat([df, scored], axis=1)
    out.to_csv(output_path, index=False)
    print(f"{len(out)}건 계산, 저장: {output_path}")


if __name__ == "__main__":
    main()
