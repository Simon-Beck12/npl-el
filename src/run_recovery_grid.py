"""EXP-007 단계 8: 포트폴리오 20건 × 경로 2(시세/감정가) × 매각 시점 시나리오 3 × 보증금 시나리오 3
그리드로 src/recovery_engine.compute_recovery를 돌려 exp007_recovery.parquet에 저장한다(360행).

예상 매각가: 시세 경로 = market_price × rho_mkt_est/100, 감정가 경로 = 감정가 × rho_appr_est/100
(감정가 결측 물건은 감정가 경로가 결측으로 남는다). 대출원금(EAD) = market_price × LTV(0.70),
경로·시나리오와 무관하게 고정. 가정값은 src/recovery_engine.DEFAULT_ASSUMPTIONS 그대로 사용한다.
"""

from pathlib import Path

import pandas as pd

from recovery_engine import compute_recovery

ROOT = Path(__file__).resolve().parent.parent
PORTFOLIO_PARQUET = ROOT / "data" / "processed" / "portfolio_v1.parquet"
OUT_PARQUET = ROOT / "data" / "processed" / "exp007_recovery.parquet"

SCENARIOS_MONTHS = [("낙관", 12), ("기본", 24), ("비관", 36)]
DEPOSIT_SCENARIOS = ["없음", "기본", "깡통"]
DEPOSIT_COL = {"없음": "보증금_없음", "기본": "보증금_기본", "깡통": "보증금_깡통"}
PATHS = [
    ("시세", lambda row: row["market_price"] * row["rho_mkt_est"] / 100),
    ("감정가", lambda row: row["감정가"] * row["rho_appr_est"] / 100 if pd.notna(row["감정가"]) else float("nan")),
]


def main():
    portfolio = pd.read_parquet(PORTFOLIO_PARQUET)

    rows = []
    for _, item in portfolio.iterrows():
        for path_name, price_fn in PATHS:
            sale_price = price_fn(item)
            for scenario_name, months in SCENARIOS_MONTHS:
                for dep_name in DEPOSIT_SCENARIOS:
                    deposit = item[DEPOSIT_COL[dep_name]]
                    if pd.isna(sale_price) or pd.isna(deposit):
                        waterfall = {k: float("nan") for k in [
                            "매각가", "집행비용", "최우선변제", "당해세", "선순위보증금잔액",
                            "남은금액(본채권 배당 전)", "채권최고액", "청구금액", "연체연수",
                            "연체금리", "R", "PV", "EAD", "LGD",
                        ]}
                    else:
                        waterfall = compute_recovery(
                            sale_price=sale_price, deposit=deposit,
                            loan_principal=item["대출원금"], scenario_months=months,
                        )
                    rows.append({
                        "cltrMngNo": item["cltrMngNo"], "masked_주소": item["masked_주소"],
                        "결과구분": item["결과구분"], "극단물건": item["극단물건"], "구": item["구"],
                        "경로": path_name, "매각시점시나리오": scenario_name, "개월수": months,
                        "보증금시나리오": dep_name, **waterfall,
                    })

    out = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"저장: {OUT_PARQUET} ({len(out)}행, 기대값=20*2*3*3=360)")

    print("\n[포트폴리오 LGD, 경로×보증금시나리오×매각시점 요약]")
    summary = out.dropna(subset=["LGD"]).groupby(["경로", "매각시점시나리오", "보증금시나리오"]).agg(
        n=("LGD", "size"), PV합계=("PV", "sum"), EAD합계=("EAD", "sum")
    )
    summary["포트폴리오LGD"] = 1 - summary["PV합계"] / summary["EAD합계"]
    print(summary.to_string())

    return out


if __name__ == "__main__":
    main()
