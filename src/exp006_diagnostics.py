"""EXP-006 보고서용 진단(판정에 쓰지 않음, 표 5 아래 기록용).

(a) 월세가 0인 전세 계약만으로 계산한 보증금 비율 중앙값·건수(인수부담 의심 vs 정상 매각).
(b) 두 집단의 보증금 비율 계산에 쓰인 전월세 계약의 계약일 분포(중앙값)와 최초 개찰일까지의
    기간(일, 중앙값). 집단 내 모든 매칭 계약을 풀링해 계산한다(물건별 아님).

deposit_ratio.py/mixture_test.py는 수정하지 않고 두 모듈의 로직만 재사용한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from deposit_ratio import load_rent_with_key
from mixture_test import RISK_THRESHOLD, load_learning_target_v3

ROOT = Path(__file__).resolve().parent.parent
DIAG_JEONSE_CSV = ROOT / "results" / "exp006_diag_jeonse.csv"


def build_risk_groups():
    target = load_learning_target_v3()
    target["risk_group"] = np.where(
        (target["rho_appr"] < RISK_THRESHOLD) | (target["rho_mkt"] < RISK_THRESHOLD),
        "인수부담 의심", "정상 매각",
    )
    target = target.copy()
    target["key"] = target["구"] + "|" + target["동"].str.strip() + "|" + target["지번"].str.strip()
    target["cutoff"] = pd.to_datetime(target["최초 개찰일"], format="%Y%m%d%H%M")
    return target


def matched_prior_rows(target, rent, jeonse_only=False):
    rent_by_key = {k: g for k, g in rent.groupby("key")}
    rows = []
    for _, item in target.iterrows():
        g = rent_by_key.get(item["key"])
        if g is None:
            continue
        prior = g[g["contract_date"] < item["cutoff"]]
        if jeonse_only:
            prior = prior[prior["monthlyRent"] == 0]
        if len(prior) == 0:
            continue
        rows.append((item["cltrMngNo"], item["risk_group"], item["market_price"], item["cutoff"], prior))
    return rows


def diag_a_jeonse_only(target, rent):
    """전세(월세=0) 계약만으로 물건별 보증금 비율을 다시 계산해 집단별 중앙값·건수."""
    matches = matched_prior_rows(target, rent, jeonse_only=True)
    out = []
    for cid, grp, mp, cutoff, prior in matches:
        med_deposit_won = prior["deposit"].median() * 10000
        out.append({"cltrMngNo": cid, "risk_group": grp, "deposit_ratio_jeonse": med_deposit_won / mp})
    df = pd.DataFrame(out)
    summary = df.groupby("risk_group")["deposit_ratio_jeonse"].agg(
        중앙값="median", 건수="count"
    )
    return summary


def diag_b_contract_date_dist(target, rent):
    """보증금 비율 계산(전체 계약유형)에 쓰인 전월세 계약들의 계약일·개찰일까지 기간을 집단별로 풀링."""
    matches = matched_prior_rows(target, rent, jeonse_only=False)
    pooled = {"인수부담 의심": [], "정상 매각": []}
    for cid, grp, mp, cutoff, prior in matches:
        gap_days = (cutoff - prior["contract_date"]).dt.total_seconds() / 86400
        for d, g in zip(prior["contract_date"], gap_days):
            pooled[grp].append((d, g))

    rows = []
    for grp, items in pooled.items():
        if not items:
            rows.append({"risk_group": grp, "n_contracts": 0, "계약일_중앙값": None, "개찰일까지_기간_중앙값_일": None})
            continue
        dates = pd.Series([d for d, _ in items])
        gaps = pd.Series([g for _, g in items])
        rows.append({
            "risk_group": grp, "n_contracts": len(items),
            "계약일_중앙값": dates.median(), "개찰일까지_기간_중앙값_일": round(gaps.median(), 1),
        })
    return pd.DataFrame(rows).set_index("risk_group")


def main():
    target = build_risk_groups()
    rent = load_rent_with_key()

    a = diag_a_jeonse_only(target, rent)
    print("[진단 a] 전세(월세=0)만 사용한 보증금 비율:")
    print(a)
    a.to_csv(DIAG_JEONSE_CSV)
    print(f"저장(EXP-007 0단계, 2026-11-04): {DIAG_JEONSE_CSV}")

    b = diag_b_contract_date_dist(target, rent)
    print("[진단 b] 매칭 계약 계약일·개찰일까지 기간 분포(풀링):")
    print(b)

    return a, b


if __name__ == "__main__":
    main()
