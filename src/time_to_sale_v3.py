"""EXP-006 단계 16: 매각 시점 — 기본 KM(관측 첫 회차=1 물건만, 사용자 결정 2026-11-04) vs
지연 진입 KM(왼쪽 절단 물건의 진입 시점을 회차 간격 중앙값으로 보정, 민감도).

src/time_to_sale.py의 kaplan_meier/survival_at은 그대로 재사용(수정 안 함)한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from time_to_sale import kaplan_meier, survival_at

ROOT = Path(__file__).resolve().parent.parent
ROUNDS_PARQUET = ROOT / "data" / "processed" / "onbid_rounds_raw.parquet"
FEATURES_V3_PARQUET = ROOT / "data" / "processed" / "onbid_features_v3.parquet"


def round_interval_median(rounds, target_ids):
    r = rounds[rounds["cltrMngNo"].isin(target_ids)].sort_values(["cltrMngNo", "pbctNsq"])
    r = r.copy()
    r["cltrOpbdDt_dt"] = pd.to_datetime(r["cltrOpbdDt"], format="%Y%m%d%H%M")
    gaps = r.groupby("cltrMngNo")["cltrOpbdDt_dt"].diff().dt.total_seconds() / 86400
    gaps = gaps.dropna()
    return float(gaps.median())


def main():
    rounds = pd.read_parquet(ROUNDS_PARQUET)
    f = pd.read_parquet(FEATURES_V3_PARQUET)
    target_ids = set(f["cltrMngNo"])

    first_round = rounds[rounds["cltrMngNo"].isin(target_ids)].groupby("cltrMngNo")["pbctNsq"].min()
    f = f.set_index("cltrMngNo")
    f["first_round_obs"] = first_round
    f = f.reset_index()

    interval_med = round_interval_median(rounds, target_ids)
    print(f"회차 간격 중앙값: {interval_med:.1f}일")

    opened = pd.to_datetime(f["최초 개찰일"], format="%Y%m%d%H%M")
    closed = pd.to_datetime(f["최종 개찰일"], format="%Y%m%d%H%M")
    f["event_sold"] = (f["최종 결과"] == "낙찰").astype(int)

    # ---------- 기본 KM: 관측 첫 회차=1인 물건만 ----------
    base = f[f["first_round_obs"] == 1].copy()
    base["duration_days"] = (closed.loc[base.index] - opened.loc[base.index]).dt.total_seconds() / 86400
    curve_base, n_base = kaplan_meier(base["duration_days"].to_numpy(), base["event_sold"].to_numpy())
    max_t_base = base["duration_days"].max()

    prob_base = {t: (1 - survival_at(curve_base, t)) * 100 for t in [90, 180, 365]}
    prob_base_max = (1 - survival_at(curve_base, max_t_base)) * 100
    print(f"[기본 KM] n={n_base}(낙찰 {int(base['event_sold'].sum())}건), "
          f"90/180/365일 매각확률={prob_base[90]:.1f}%/{prob_base[180]:.1f}%/{prob_base[365]:.1f}%, "
          f"관측최대({max_t_base:.0f}일) 매각확률={prob_base_max:.1f}%")

    # ---------- 지연 진입 KM: 전체 물건, 절단 물건은 진입 시점 보정 ----------
    delayed = f.copy()
    entry = opened - pd.to_timedelta((delayed["first_round_obs"] - 1) * interval_med, unit="D")
    delayed["duration_adj_days"] = (closed - entry).dt.total_seconds() / 86400
    curve_delay, n_delay = kaplan_meier(delayed["duration_adj_days"].to_numpy(), delayed["event_sold"].to_numpy())
    max_t_delay = delayed["duration_adj_days"].max()

    prob_delay = {t: (1 - survival_at(curve_delay, t)) * 100 for t in [90, 180, 365]}
    prob_delay_max = (1 - survival_at(curve_delay, max_t_delay)) * 100
    print(f"[지연 진입 KM] n={n_delay}(낙찰 {int(delayed['event_sold'].sum())}건), "
          f"90/180/365일 매각확률={prob_delay[90]:.1f}%/{prob_delay[180]:.1f}%/{prob_delay[365]:.1f}%, "
          f"관측최대({max_t_delay:.0f}일) 매각확률={prob_delay_max:.1f}%")

    return {
        "interval_median_days": interval_med,
        "base": {"n": n_base, "n_events": int(base["event_sold"].sum()), "prob": prob_base,
                  "prob_max": prob_base_max, "max_t": float(max_t_base)},
        "delayed": {"n": n_delay, "n_events": int(delayed["event_sold"].sum()), "prob": prob_delay,
                     "prob_max": prob_delay_max, "max_t": float(max_t_delay)},
    }


if __name__ == "__main__":
    main()
