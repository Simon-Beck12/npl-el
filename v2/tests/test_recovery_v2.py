"""EXP-009 단계 7: v2/recovery.compute_recovery_v2 단위 테스트 4건.

6줄 계산(그대로): S=시세×낙찰가율, K=S*1%, P=(보증금<=16,500만이면)min(보증금,5,500만,S*0.5) 아니면 0,
D2=보증금-P, R=max(0,min(EAD,S-K-P-D2)), PV=R/(1+할인율)^(개월/12), LGD=1-PV/EAD.
각 테스트는 손으로 계산한 기대값을 주석에 남긴다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recovery import compute_recovery_v2


def test_1_no_deposit():
    """보증금 없음.
    S = 100,000,000*0.7 = 70,000,000
    K = 700,000
    P = min(0,55,000,000,35,000,000) = 0 (보증금 0)
    D2 = 0 - 0 = 0
    R = max(0, min(70,000,000, 70,000,000-700,000-0-0)) = 69,300,000
    PV = 69,300,000 / 1.08^2 = 59,414,609.05...
    """
    res = compute_recovery_v2(market_price=100_000_000, rho=0.7, deposit=0, ead=70_000_000, months=24)
    assert res["P"] == 0
    assert res["D2"] == 0
    assert res["R"] == pytest.approx(69_300_000)
    assert res["PV"] == pytest.approx(69_300_000 / (1.08 ** 2))
    assert res["LGD"] == pytest.approx(1 - res["PV"] / 70_000_000)


def test_2_small_tenant_55m_cap():
    """소액임차인, 최우선변제 5,500만원 상한(보증금·매각가 상한보다 낮음).
    S = 300,000,000*0.8 = 240,000,000, S*0.5 = 120,000,000
    보증금 100,000,000 <= 165,000,000 → P = min(100,000,000, 55,000,000, 120,000,000) = 55,000,000 (상한 고정값이 가장 작음)
    D2 = 100,000,000 - 55,000,000 = 45,000,000
    """
    res = compute_recovery_v2(market_price=300_000_000, rho=0.8, deposit=100_000_000, ead=200_000_000, months=24)
    assert res["P"] == pytest.approx(55_000_000)
    assert res["D2"] == pytest.approx(45_000_000)


def test_3_underwater_zero_recovery():
    """깡통: 보증금이 1.65억 초과라 최우선변제 0, 보증금 전액이 남은금액을 넘어 R이 0으로 바닥.
    S = 100,000,000*0.5 = 50,000,000, K = 500,000
    보증금 200,000,000 > 165,000,000 → P = 0, D2 = 200,000,000
    S-K-P-D2 = 50,000,000-500,000-0-200,000,000 = -150,500,000 (음수)
    R = max(0, min(70,000,000, -150,500,000)) = 0
    """
    res = compute_recovery_v2(market_price=100_000_000, rho=0.5, deposit=200_000_000, ead=70_000_000, months=24)
    assert res["R"] == 0
    assert res["PV"] == 0
    assert res["LGD"] == pytest.approx(1.0)


def test_4_recovery_exceeds_principal_capped_at_ead():
    """회수액이 원금(EAD)을 넘는 경우 EAD에서 자른다(연체이자·채권최고액 없음, v1과 다른 핵심 차이).
    S = 100,000,000*1.5 = 150,000,000, K = 1,500,000, P=D2=0
    S-K-P-D2 = 148,500,000 > EAD(50,000,000) → R = min(50,000,000, 148,500,000) = 50,000,000 = EAD
    """
    res = compute_recovery_v2(market_price=100_000_000, rho=1.5, deposit=0, ead=50_000_000, months=24)
    assert res["R"] == pytest.approx(50_000_000)
    assert res["R"] == pytest.approx(res["EAD"])
    assert res["PV"] == pytest.approx(50_000_000 / (1.08 ** 2))
