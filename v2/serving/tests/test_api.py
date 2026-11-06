"""EXP-011 9단계: FastAPI TestClient 기반 API 테스트.

효창동 예시 기대값(LGD 58.6%, ±0.1%p)은 compute_recovery_v2(market_price=48974,
rho=0.644620, deposit=14700, ead=34281.8, months=24)를 직접 계산해 미리 확인한 값이다.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "test_mdape" in body["meta"]


def test_recovery_hyochang_example():
    r = client.post("/recovery", json={
        "market_price": 48974, "rho": 0.644620,
        "deposit": 14700, "ead": 34281.8, "months": 24,
    })
    assert r.status_code == 200
    lgd = r.json()["LGD"]
    assert abs(lgd - 0.586) <= 0.001


def test_recovery_hyochang_example_via_scenario():
    r = client.post("/recovery", json={
        "market_price": 48974, "rho_scenario": "기본",
        "deposit": 14700, "ead": 34281.8, "months": 24,
    })
    assert r.status_code == 200
    lgd = r.json()["LGD"]
    assert abs(lgd - 0.586) <= 0.001


@pytest.mark.parametrize("payload", [
    {"lat": 37.6036, "lng": 126.9589, "excluUseAr": 0, "floor": 3,
     "buildYear": 2005, "dealYear": 2026, "dealMonth": 7},
    {"lat": 37.0, "lng": 126.9589, "excluUseAr": 36.5, "floor": 3,
     "buildYear": 2005, "dealYear": 2026, "dealMonth": 7},
])
def test_valuation_invalid_input_422(payload):
    r = client.post("/valuation", json=payload)
    assert r.status_code == 422


@pytest.mark.parametrize("payload", [
    {"market_price": 48974, "deposit": 14700, "ead": 34281.8, "months": 0},
    {"market_price": 48974, "deposit": 14700, "ead": 34281.8, "months": 24},  # rho도 rho_scenario도 없음
])
def test_recovery_invalid_input_422(payload):
    r = client.post("/recovery", json=payload)
    assert r.status_code == 422


def test_valuation_ok():
    r = client.post("/valuation", json={
        "lat": 37.6036, "lng": 126.9589, "excluUseAr": 36.5, "floor": 3,
        "buildYear": 2005, "dealYear": 2026, "dealMonth": 7,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["market_price"] > 0
    assert body["same_bldg"] in (0, 1)
