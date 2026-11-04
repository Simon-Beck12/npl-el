"""EXP-007 단계 3: 회수액·LGD 계산 엔진.

입력(예상 매각가, 보증금, 시나리오 개월 수, 가정값)을 받아 배당표(단계별 금액), 본채권 회수액 R,
현재가치 PV, LGD를 돌려준다. 배당 순서(사전 고정, [실험 등록 내용] 그대로):
  1. 집행비용(매각가×비율)
  2. 최우선변제: 보증금<=1.65억(서울)이면 min(보증금, 5,500만), 합계는 매각가의 1/2을 넘지 않음
  3. 당해세(이번 가정값은 0)
  4. 선순위 임차보증금 중 최우선변제로 받지 못한 나머지
  5. 본 채권 R = min(채권최고액, 청구금액, 남은 금액). 음수면 0
청구금액 = 대출원금 × (1 + 연체금리 × 연체연수), 연체연수 = (연체시작_개월전 + 시나리오개월수) / 12.
PV = R ÷ (1+할인율)^(시나리오개월수/12). LGD = 1 − PV/EAD(EAD=대출원금).
범위 밖: 흡수배당, 임금채권, 가압류 안분배당, 당해세 예외(2023-04-01) 세부, 상가 환산보증금.
"""

DEFAULT_ASSUMPTIONS = {
    "ltv": 0.70,
    "bond_max_ratio": 1.20,
    "contract_rate": 0.12,
    "default_rate_premium": 0.03,
    "default_start_months_before_cutoff": 12,
    "execution_cost_ratio": 0.01,
    "property_tax": 0.0,
    "discount_rate": 0.08,
    "priority_deposit_cap": 165_000_000,
    "priority_deposit_max_payout": 55_000_000,
    "priority_payout_sale_ratio_cap": 0.5,
}


def compute_recovery(sale_price, deposit, loan_principal, scenario_months, assumptions=None):
    """배당표·R·PV·LGD를 계산해 dict로 돌려준다. 금액 단위는 입력과 동일(원)."""
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}

    bond_max = loan_principal * a["bond_max_ratio"]
    default_years = (a["default_start_months_before_cutoff"] + scenario_months) / 12
    default_rate = a["contract_rate"] + a["default_rate_premium"]
    claim_amount = loan_principal * (1 + default_rate * default_years)

    execution_cost = sale_price * a["execution_cost_ratio"]
    remaining_after_cost = max(sale_price - execution_cost, 0.0)

    if deposit <= a["priority_deposit_cap"]:
        priority_candidate = min(deposit, a["priority_deposit_max_payout"])
    else:
        priority_candidate = 0.0
    priority_cap_by_sale = sale_price * a["priority_payout_sale_ratio_cap"]
    priority_payout = max(min(priority_candidate, priority_cap_by_sale, remaining_after_cost), 0.0)
    remaining_after_priority = max(remaining_after_cost - priority_payout, 0.0)

    property_tax = min(a["property_tax"], remaining_after_priority)
    remaining_after_tax = max(remaining_after_priority - property_tax, 0.0)

    deposit_remainder = max(deposit - priority_payout, 0.0)
    deposit_remainder_payout = max(min(deposit_remainder, remaining_after_tax), 0.0)
    remaining_after_deposit = max(remaining_after_tax - deposit_remainder_payout, 0.0)

    recovery = max(min(bond_max, claim_amount, remaining_after_deposit), 0.0)

    pv = recovery / ((1 + a["discount_rate"]) ** (scenario_months / 12))
    ead = loan_principal
    lgd = (1 - pv / ead) if ead > 0 else None

    return {
        "매각가": sale_price,
        "집행비용": execution_cost,
        "최우선변제": priority_payout,
        "당해세": property_tax,
        "선순위보증금잔액": deposit_remainder_payout,
        "남은금액(본채권 배당 전)": remaining_after_deposit,
        "채권최고액": bond_max,
        "청구금액": claim_amount,
        "연체연수": default_years,
        "연체금리": default_rate,
        "R": recovery,
        "PV": pv,
        "EAD": ead,
        "LGD": lgd,
    }
