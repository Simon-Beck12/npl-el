"""EXP-007 단계 4: src/recovery_engine.compute_recovery 단위 테스트 6건.

각 테스트는 기대값을 손으로(또는 같은 공식을 그대로 문자로) 계산해 주석/표현식으로 남긴다.
기본 가정값(DEFAULT_ASSUMPTIONS): ltv=0.70, bond_max_ratio=1.20, contract_rate=0.12,
default_rate_premium=0.03(연체금리=0.15), default_start_months_before_cutoff=12,
execution_cost_ratio=0.01, property_tax=0, discount_rate=0.08,
priority_deposit_cap=165,000,000, priority_deposit_max_payout=55,000,000,
priority_payout_sale_ratio_cap=0.5.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recovery_engine import compute_recovery


def test_1_no_tenant_recovery_below_claim():
    """임차인 없음, 회수액이 청구금액보다 작음(남은 금액이 채권최고액·청구금액보다 작게 설계).

    집행비용 = 150,000,000*0.01 = 1,500,000
    남은금액 = 150,000,000 - 1,500,000 = 148,500,000 (임차인 없어 추가 공제 없음)
    연체연수 = (12+12)/12 = 2.0, 청구금액 = 200,000,000*(1+0.15*2.0) = 260,000,000
    채권최고액 = 200,000,000*1.20 = 240,000,000
    R = min(240,000,000, 260,000,000, 148,500,000) = 148,500,000 (남은금액이 가장 작음)
    PV = 148,500,000 / 1.08 = 137,500,000 (137,500,000*1.08=148,500,000 검증됨)
    LGD = 1 - 137,500,000/200,000,000 = 0.3125
    """
    r = compute_recovery(sale_price=150_000_000, deposit=0, loan_principal=200_000_000, scenario_months=12)
    assert r["집행비용"] == 1_500_000
    assert r["최우선변제"] == 0
    assert r["선순위보증금잔액"] == 0
    assert r["청구금액"] == 260_000_000
    assert r["채권최고액"] == 240_000_000
    assert r["남은금액(본채권 배당 전)"] == 148_500_000
    assert r["R"] == 148_500_000
    assert r["R"] < r["청구금액"]
    assert r["PV"] == pytest.approx(137_500_000)
    assert r["LGD"] == pytest.approx(0.3125)


def test_2_small_tenant_priority_55m_cap():
    """소액임차인(보증금 1억 원) 최우선변제 5,500만 원 적용(55M 절대 상한이 매각가/2 상한보다 먼저 걸림).

    보증금 100,000,000 <= 165,000,000 -> candidate = min(100,000,000, 55,000,000) = 55,000,000
    매각가/2 상한 = 300,000,000*0.5 = 150,000,000 (55,000,000보다 커서 상한 아님)
    집행비용 = 3,000,000, 남은금액 = 297,000,000 (55,000,000보다 커서 지급 가능)
    -> 최우선변제 = 55,000,000, 보증금 잔액 = 100,000,000-55,000,000 = 45,000,000(전액 지급 가능)
    """
    r = compute_recovery(sale_price=300_000_000, deposit=100_000_000, loan_principal=150_000_000, scenario_months=24)
    assert r["최우선변제"] == 55_000_000
    assert r["선순위보증금잔액"] == 45_000_000
    assert r["R"] >= 0


def test_3_underwater_deposit_zero_recovery():
    """깡통 시나리오(선순위 보증금이 매각가를 크게 웃돎, 추정시세×90%가 매각가보다 훨씬 큰 극단 저가낙찰을 흉내냄)에서 R=0.

    보증금 180,000,000 > 165,000,000 -> 최우선변제 후보 = 0 (최우선변제 전액 0)
    집행비용 = 50,000,000*0.01 = 500,000, 남은금액 = 49,500,000
    보증금잔액 = 180,000,000(최우선변제 0 공제) -> min(180,000,000, 49,500,000) = 49,500,000 전액 소진
    남은금액(본채권 배당 전) = 49,500,000 - 49,500,000 = 0 -> R = min(채권최고액,청구금액,0) = 0
    """
    r = compute_recovery(sale_price=50_000_000, deposit=180_000_000, loan_principal=100_000_000, scenario_months=24)
    assert r["최우선변제"] == 0
    assert r["남은금액(본채권 배당 전)"] == 0
    assert r["R"] == 0
    assert r["PV"] == 0
    assert r["LGD"] == pytest.approx(1.0)


def test_4_bond_max_cap_binds():
    """채권최고액이 상한으로 작동(청구금액·남은금액 모두 채권최고액보다 커서 채권최고액이 R을 결정).

    채권최고액 = 100,000,000*1.20 = 120,000,000
    연체연수 = (12+36)/12 = 4.0, 청구금액 = 100,000,000*(1+0.15*4.0) = 160,000,000 (>120,000,000)
    집행비용 = 500,000,000*0.01=5,000,000, 남은금액 = 495,000,000 (>120,000,000)
    R = min(120,000,000, 160,000,000, 495,000,000) = 120,000,000
    """
    r = compute_recovery(sale_price=500_000_000, deposit=0, loan_principal=100_000_000, scenario_months=36)
    assert r["채권최고액"] == 120_000_000
    assert r["청구금액"] == 160_000_000
    assert r["남은금액(본채권 배당 전)"] == 495_000_000
    assert r["R"] == 120_000_000
    assert r["PV"] == pytest.approx(120_000_000 / (1.08 ** 3))


def test_5_priority_payout_half_sale_price_cap_binds():
    """최우선변제가 매각가의 1/2 상한에 걸림(55M 절대 상한보다 매각가/2 상한이 더 작게 설계).

    매각가/2 상한 = 80,000,000*0.5 = 40,000,000 (< 55,000,000 절대 상한)
    보증금 100,000,000 <= 165,000,000 -> candidate = min(100,000,000,55,000,000) = 55,000,000
    집행비용 = 800,000, 남은금액 = 79,200,000(>40,000,000이라 40,000,000 전액 지급 가능)
    -> 최우선변제 = min(55,000,000, 40,000,000, 79,200,000) = 40,000,000 (매각가/2 상한이 결정)
    """
    r = compute_recovery(sale_price=80_000_000, deposit=100_000_000, loan_principal=50_000_000, scenario_months=12)
    assert r["최우선변제"] == 40_000_000
    assert r["R"] >= 0


def test_6_sale_price_below_execution_cost_no_negative():
    """매각가가 집행비용보다 작음(R=0, 음수 없음). execution_cost_ratio를 150%로 올려
    집행비용(15,000,000)이 매각가(10,000,000)를 넘는 극단 상황을 강제로 만든다.

    남은금액 = max(10,000,000-15,000,000, 0) = 0 (음수로 내려가지 않음)
    이후 모든 단계(최우선변제·당해세·보증금잔액·R)도 0에서 더 못 내려가므로 전부 0 이상
    """
    r = compute_recovery(
        sale_price=10_000_000, deposit=0, loan_principal=20_000_000, scenario_months=12,
        assumptions={"execution_cost_ratio": 1.5},
    )
    assert r["남은금액(본채권 배당 전)"] == 0
    assert r["R"] == 0
    assert r["PV"] == 0
    for key in ["최우선변제", "당해세", "선순위보증금잔액", "남은금액(본채권 배당 전)", "R", "PV"]:
        assert r[key] >= 0
