"""EXP-005: 공매 빌라 매각 시점(일수·회차) Kaplan-Meier 추정.

사건=낙찰. 유찰/취소/수의계약가능 등 낙찰이 아닌 모든 종료 상태는 그 물건의 마지막 개찰 시점에서
중도절단한다(취소 시점을 별도로 기록한 원본 필드가 없어 최종 개찰일로 근사).
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = ROOT / "data" / "processed" / "onbid_features.parquet"
FIG_PATH = ROOT / "results" / "figures" / "exp005_km.png"


def kaplan_meier(durations, events):
    df = pd.DataFrame({"t": durations, "e": events})
    times = sorted(df["t"].unique())
    survival = 1.0
    curve = [(0.0, 1.0)]
    n_total = len(df)
    for t in times:
        n_at_risk = int((df["t"] >= t).sum())
        d = int(df.loc[df["t"] == t, "e"].sum())
        if n_at_risk > 0:
            survival *= (1 - d / n_at_risk)
        curve.append((float(t), float(survival)))
    return curve, n_total


def survival_at(curve, t_query):
    s = 1.0
    for t, surv in curve:
        if t <= t_query:
            s = surv
        else:
            break
    return s


def median_survival_time(curve):
    for t, surv in curve:
        if surv <= 0.5:
            return t
    return float("nan")


def main():
    f = pd.read_parquet(FEATURES_PARQUET)

    opened = pd.to_datetime(f["최초 개찰일"], format="%Y%m%d%H%M")
    closed = pd.to_datetime(f["최종 개찰일"], format="%Y%m%d%H%M")
    f = f.copy()
    f["duration_days"] = (closed - opened).dt.total_seconds() / 86400
    f["duration_rounds"] = f["최종 회차"].astype(float)
    f["event_sold"] = (f["최종 결과"] == "낙찰").astype(int)

    print("상태별 건수:", f["최종 결과"].value_counts().to_dict())
    print(f"일수 범위: {f['duration_days'].min():.1f} ~ {f['duration_days'].max():.1f}")

    curve_days, n_days = kaplan_meier(f["duration_days"].to_numpy(), f["event_sold"].to_numpy())
    curve_rounds, n_rounds = kaplan_meier(f["duration_rounds"].to_numpy(), f["event_sold"].to_numpy())

    median_days = median_survival_time(curve_days)
    median_rounds = median_survival_time(curve_rounds)

    prob_sold = {}
    for t in [90, 180, 365]:
        s = survival_at(curve_days, t)
        prob_sold[t] = (1 - s) * 100

    print(f"중앙 매각 일수: {median_days}")
    print(f"중앙 매각 회차: {median_rounds}")
    print(f"매각 확률(일수): {prob_sold}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    t_d, s_d = zip(*curve_days)
    axes[0].step(t_d, s_d, where="post")
    axes[0].set_xlabel("일수")
    axes[0].set_ylabel("매각되지 않을 확률 S(t)")
    axes[0].set_title("매각 시점 (일수 기준)")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.3)

    t_r, s_r = zip(*curve_rounds)
    axes[1].step(t_r, s_r, where="post", color="darkorange")
    axes[1].set_xlabel("회차")
    axes[1].set_ylabel("매각되지 않을 확률 S(t)")
    axes[1].set_title("매각 시점 (회차 기준)")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"EXP-005 공매 빌라 매각 시점 Kaplan-Meier (n={len(f)}, 사건=낙찰 {f['event_sold'].sum()}건)")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=110)
    plt.close(fig)

    size_kb = FIG_PATH.stat().st_size / 1024
    print(f"저장: {FIG_PATH} ({size_kb:.1f} KB)")
    assert size_kb <= 300, f"그림 용량 {size_kb:.1f}KB가 300KB를 넘음 — 멈춤"

    return {
        "n": len(f), "median_days": median_days, "median_rounds": median_rounds,
        "prob_sold": prob_sold, "n_events": int(f["event_sold"].sum()),
    }


if __name__ == "__main__":
    main()
