"""EXP-008 채택 모델(G2, LightGBM 8변수)을 불러와 물건 1건의 시세와 회수액을 계산한다.

피처 계산은 v2/price_model.py의 테스트 예측 경로(`build_lgbm_frame(test, train_all, exclude_self=False)`)와
같은 식을 쓴다. 같은 값이 나오는지는 export_artifacts.py가 만든 golden.json으로 확인한다.

artifact 폴더 구성 (export_artifacts.py가 생성):
- model.txt       LightGBM 모델(텍스트)
- reference.csv   학습 거래(train_all) — loc_x_km, loc_y_km, price_per_m2(만원/㎡), bldg_key, excluUseAr
- meta.json       피처 순서, 좌표 변환 상수, 학습 기간, 낙찰가율 시나리오 등
- golden.json     테스트 거래 일부의 입력과 export 시점 예측값(일치 확인용)
"""
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from recovery_core import LTV, compute_recovery

FAR_NEIGHBOR_KM = 1.0


def bldg_key(gu, dong, jibun):
    return f"{gu}|{str(dong).strip()}|{str(jibun).strip()}"


class Predictor:
    def __init__(self, artifacts_dir):
        self.dir = Path(artifacts_dir)
        with open(self.dir / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.booster = lgb.Booster(model_file=str(self.dir / "model.txt"))
        ref = pd.read_csv(self.dir / "reference.csv", dtype={"bldg_key": str})
        self.ref_xy = ref[["loc_x_km", "loc_y_km"]].to_numpy()
        self.ref_ppm2 = ref["price_per_m2"].to_numpy()
        self.ref_bldg = set(ref["bldg_key"])
        self.tree = cKDTree(self.ref_xy)
        self.feature_cols = self.meta["feature_cols"]
        self.knn_k = int(self.meta["knn_k"])

    # ------------------------------------------------------------ 피처 ----
    def _loc_km(self, lat, lng):
        m = self.meta["geo"]
        lat_rad = math.radians(m["ref_lat"])
        x = (lng - m["ref_lng"]) * m["km_per_lng_deg"] * math.cos(lat_rad)
        y = (lat - m["ref_lat"]) * m["km_per_lat_deg"]
        return x, y

    def _month_idx(self, year, month):
        y0, m0 = self.meta["month0"]
        return (year - y0) * 12 + (month - m0)

    def build_features(self, item):
        x, y = self._loc_km(item["lat"], item["lng"])
        # price_model.add_knn_ppm2와 같은 호출 방식(2차원 배열, k=5, 자기 자신 제외 없음)
        dist, idx = self.tree.query(np.array([[x, y]]), k=self.knn_k)
        idx = idx[0]
        dist = dist[0]
        feats = {
            "lat": item["lat"],
            "lng": item["lng"],
            "excluUseAr": item["area_m2"],
            "age_years": item["as_of_year"] - item["build_year"],
            "floor_feat": float(item["floor"]),
            "month_idx": self._month_idx(item["as_of_year"], item["as_of_month"]),
            "knn_ppm2_5": float(np.median(self.ref_ppm2[idx])),
            "same_bldg": int(bldg_key(item["gu"], item["dong"], item["jibun"]) in self.ref_bldg),
        }
        neighbors = [
            {"distance_km": round(float(d), 4), "price_per_m2_manwon": float(self.ref_ppm2[i])}
            for d, i in zip(dist, idx)
        ]
        return feats, neighbors

    def _flags(self, feats, neighbors):
        flags = []
        if feats["month_idx"] > self.meta["max_month_idx"]:
            flags.append(
                f"기준월이 학습 마지막 월({self.meta['train_end'][:7]}) 이후다. "
                "LightGBM은 시간 추세를 외삽하지 않아 마지막 월 수준으로 고정된다."
            )
        if neighbors[0]["distance_km"] > FAR_NEIGHBOR_KM:
            flags.append(f"가장 가까운 학습 거래가 {neighbors[0]['distance_km']:.2f} km 떨어져 있다.")
        lo, hi = self.meta["area_q01_q99"]
        if not (lo <= feats["excluUseAr"] <= hi):
            flags.append(f"전용면적이 학습 데이터 1~99% 범위({lo:.1f}~{hi:.1f}㎡) 밖이다.")
        if feats["same_bldg"] == 0:
            flags.append("같은 건물(구·동·지번)의 학습 거래가 없다. EXP-008 표2에서 이 그룹 오차가 더 컸다.")
        return flags

    # ------------------------------------------------------------ 예측 ----
    def predict_price(self, item):
        feats, neighbors = self.build_features(item)
        x = pd.DataFrame([feats])[self.feature_cols]
        pred_log = float(self.booster.predict(x)[0])
        price_manwon = math.exp(pred_log) * item["area_m2"]
        return {
            "price_manwon": price_manwon,
            # 학습 타깃(dealAmount)은 만원 단위, 회수액 계산은 원 단위(v2/recovery.py와 동일)
            "price_won": price_manwon * 10_000,
            "features": feats,
            "neighbors": neighbors,
            "flags": self._flags(feats, neighbors),
        }

    def predict_recovery(self, item, deposit_won, ead_won=None, months_list=(12, 24, 36)):
        price = self.predict_price(item)
        market = price["price_won"]
        ead_assumed = ead_won is None
        ead = market * LTV if ead_assumed else float(ead_won)
        rows = []
        for name, rho in self.meta["rho_scenarios"].items():
            for months in months_list:
                res = compute_recovery(market, rho, deposit_won, ead, months)
                rows.append({"rho_scenario": name, "rho": rho, "months": months, **res})
        flags = list(price["flags"])
        if ead_assumed:
            flags.append(f"EAD를 입력하지 않아 시세 × {LTV:.2f}로 가정했다.")
        return {"price": price, "ead_won": ead, "ead_assumed": ead_assumed, "grid": rows, "flags": flags}
