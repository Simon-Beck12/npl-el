"""EXP-011 8단계(EXP-013 수정: 구·umdNm·jibun 필수화)(EXP-014 9단계 수정: 벡터화 경로 추가):
물건 목록 CSV를 한 번에 시세·회수액으로 계산하는 배치 스크립트.

입력 CSV 컬럼: lat, lng, excluUseAr, floor, buildYear, dealYear, dealMonth,
구, umdNm, jibun(필수 — EXP-013, same_bldg 계산에 필요), deposit, ead, months,
rho_scenario(비관/기본/낙관)
출력 CSV 컬럼: 입력 전체 + 시세, S, R, PV, LGD, 오류(비어 있으면 정상)

score_row()는 1행씩 처리하던 기존 경로(행마다 cKDTree를 새로 만듦)를 그대로 남겨 둔다(EXP-014
동일성 확인의 비교 기준). main()이 실제로 쓰는 경로는 score_batch()(KnnIndex 1번만 생성, 유효한
행 전체을 한 번에 계산)다.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import RHO_SCENARIOS  # noqa: E402
from model_loader import load_model, KnnIndex  # noqa: E402
from recovery import compute_recovery_v2  # noqa: E402
from valuation_core import build_target_frame, build_target_row, predict_valuation, predict_valuation_batch  # noqa: E402

DEFAULT_INPUT = Path(__file__).resolve().parent / "examples" / "sample_input.csv"
REQUIRED_ADDRESS_COLS = ["구", "umdNm", "jibun"]
EMPTY_RESULT = {"시세": None, "S": None, "R": None, "PV": None, "LGD": None}


def _missing_address(row):
    missing = [c for c in REQUIRED_ADDRESS_COLS
               if c not in row or pd.isna(row.get(c)) or str(row.get(c)).strip() == ""]
    return missing


def score_row(row, ref, booster):
    """기존(EXP-011) 1행 처리 경로. 행마다 predict_valuation()을 호출해 cKDTree를 새로 만든다.
    EXP-014부터는 main()에서 쓰지 않는다(score_batch()의 동일성 확인 비교 기준으로 남겨 둔다)."""
    missing = _missing_address(row)
    if missing:
        return pd.Series({**EMPTY_RESULT, "오류": f"필수 주소 컬럼 없음/빈값: {','.join(missing)}"})
    target = build_target_row(
        lat=row["lat"], lng=row["lng"], excluUseAr=row["excluUseAr"], floor=row["floor"],
        buildYear=row["buildYear"], dealYear=row["dealYear"], dealMonth=row["dealMonth"],
        gu_name=row["구"], umdNm=row["umdNm"], jibun=row["jibun"],
    )
    try:
        val = predict_valuation(target, ref, booster)
        rho = RHO_SCENARIOS[row["rho_scenario"]]
        rec = compute_recovery_v2(
            market_price=val["market_price"], rho=rho,
            deposit=row["deposit"], ead=row["ead"], months=row["months"],
        )
    except Exception as exc:
        return pd.Series({**EMPTY_RESULT, "오류": f"계산 실패: {exc}"})
    return pd.Series({"시세": val["market_price"], "S": rec["S"], "R": rec["R"], "PV": rec["PV"],
                       "LGD": rec["LGD"], "오류": ""})


def score_batch(df, ref, booster, knn_index=None):
    """EXP-014 9단계: 행마다 cKDTree를 새로 만들지 않는다. 주소 없는 행만 골라 오류 처리하고,
    나머지 전체는 build_target_frame()·predict_valuation_batch()로 한 번에 계산한다.
    회수액 계산(compute_recovery_v2, 6줄 산술)은 병목이 아니라서 그대로 행별로 돌린다."""
    if knn_index is None:
        knn_index = KnnIndex(ref)

    missing = df.apply(_missing_address, axis=1)
    is_valid = missing.apply(len) == 0

    out = pd.DataFrame(index=df.index, columns=["시세", "S", "R", "PV", "LGD", "오류"], dtype=object)
    out["오류"] = ""
    out.loc[~is_valid, "오류"] = missing[~is_valid].apply(lambda m: f"필수 주소 컬럼 없음/빈값: {','.join(m)}")

    if is_valid.any():
        valid_df = df.loc[is_valid]
        targets = build_target_frame(
            lat=valid_df["lat"], lng=valid_df["lng"], excluUseAr=valid_df["excluUseAr"],
            floor=valid_df["floor"], buildYear=valid_df["buildYear"], dealYear=valid_df["dealYear"],
            dealMonth=valid_df["dealMonth"], gu_name=valid_df["구"], umdNm=valid_df["umdNm"], jibun=valid_df["jibun"],
        )
        val_df = predict_valuation_batch(targets, knn_index, booster)
        val_df.index = valid_df.index

        for idx in valid_df.index:
            try:
                rho = RHO_SCENARIOS[df.at[idx, "rho_scenario"]]
                rec = compute_recovery_v2(
                    market_price=float(val_df.at[idx, "market_price"]), rho=rho,
                    deposit=df.at[idx, "deposit"], ead=df.at[idx, "ead"], months=df.at[idx, "months"],
                )
                out.at[idx, "시세"] = float(val_df.at[idx, "market_price"])
                out.at[idx, "S"] = rec["S"]
                out.at[idx, "R"] = rec["R"]
                out.at[idx, "PV"] = rec["PV"]
                out.at[idx, "LGD"] = rec["LGD"]
            except Exception as exc:
                out.at[idx, "오류"] = f"계산 실패: {exc}"
    return out


DEFAULT_MODEL_URI = "models:/npl-el-price@champion"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI,
                         help="MLflow 모델 URI. 빈 문자열(--model-uri '')을 주면 MODEL_DIR(1단계 방식)을 쓴다.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_scored.csv")

    df = pd.read_csv(input_path)
    booster, ref, meta = load_model(model_uri=args.model_uri)
    print(f"모델 소스: {meta['source']} (test_mdape={meta.get('test_mdape')})")
    scored = score_batch(df, ref, booster)
    out = pd.concat([df, scored], axis=1)
    out.to_csv(output_path, index=False)
    n_error = int((out["오류"] != "").sum())
    print(f"{len(out)}건 계산(오류 {n_error}건), 저장: {output_path}")
    return out, n_error


if __name__ == "__main__":
    main()
