"""EXP-011 7단계: /valuation, /batch_score가 공유하는 시세 추정 로직.

v2/price_model.py(add_knn_ppm2, add_same_bldg, month_index, FULL8), src/features.py
(REF_LAT 등 좌표 변환 상수)를 그대로 import한다. 두 파일 모두 수정하지 않는다.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "v2"))

from features import REF_LAT, REF_LNG, KM_PER_LAT_DEG, KM_PER_LNG_DEG  # noqa: E402
from price_model import add_knn_ppm2, add_same_bldg, MONTH0, FULL8  # noqa: E402

_LAT_RAD = math.radians(REF_LAT)


def _loc_km(lat, lng):
    loc_x_km = (lng - REF_LNG) * KM_PER_LNG_DEG * math.cos(_LAT_RAD)
    loc_y_km = (lat - REF_LAT) * KM_PER_LAT_DEG
    return loc_x_km, loc_y_km


def _month_idx(deal_year, deal_month):
    base_y, base_m = MONTH0
    return (deal_year - base_y) * 12 + (deal_month - base_m)


def build_target_row(lat, lng, excluUseAr, floor, buildYear, dealYear, dealMonth,
                      gu_name=None, umdNm=None, jibun=None):
    """단일 물건 입력 → FULL8 피처 1행 DataFrame. 주소(구·umdNm·jibun)가 없으면 same_bldg=0으로 둔다."""
    loc_x_km, loc_y_km = _loc_km(lat, lng)
    row = {
        "lat": lat, "lng": lng, "excluUseAr": excluUseAr,
        "age_years": dealYear - buildYear, "floor_feat": float(floor),
        "month_idx": _month_idx(dealYear, dealMonth),
        "loc_x_km": loc_x_km, "loc_y_km": loc_y_km,
        "gu_name": gu_name, "umdNm": umdNm, "jibun": jibun,
        "deal_id": "__query__",
    }
    return pd.DataFrame([row])


def predict_valuation(target, ref, booster):
    """target: build_target_row()의 1행 DataFrame. ref: model_loader.load_model()의 ref.parquet."""
    target = target.copy()
    target["knn_ppm2_5"] = add_knn_ppm2(target, ref, exclude_self=False)
    has_address = target[["gu_name", "umdNm", "jibun"]].notna().all(axis=1)
    same_bldg = np.zeros(len(target), dtype=int)
    if has_address.any():
        same_bldg[has_address] = add_same_bldg(target[has_address], ref, exclude_self=False)
    target["same_bldg"] = same_bldg

    pred_log = booster.predict(target[FULL8])
    price_per_m2 = float(np.exp(pred_log)[0])
    market_price = price_per_m2 * float(target["excluUseAr"].iloc[0])
    return {
        "market_price": market_price,
        "price_per_m2": price_per_m2,
        "knn_ppm2_5": float(target["knn_ppm2_5"].iloc[0]),
        "same_bldg": int(target["same_bldg"].iloc[0]),
    }
