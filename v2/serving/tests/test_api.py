"""FastAPI 엔드포인트 테스트(가짜 artifact 사용)."""
import pytest
from fastapi.testclient import TestClient

import app as app_module
from recovery_core import compute_recovery

BASE = {"lat": 37.556, "lng": 126.91, "area_m2": 45.0, "build_year": 2005, "floor": 3,
        "gu": "마포구", "dong": "가동", "jibun": "12-3", "as_of": "2025-10"}


@pytest.fixture()
def client(synthetic_artifacts, monkeypatch):
    monkeypatch.setattr(app_module, "ARTIFACTS_DIR", synthetic_artifacts["dir"])
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["n_reference"] == 1500


def test_price_ok(client):
    r = client.post("/v1/price", json=BASE)
    assert r.status_code == 200
    body = r.json()
    assert body["price_manwon"] > 0
    assert abs(body["price_won"] - body["price_manwon"] * 10_000) <= 10_000 * 0.1 + 1
    assert len(body["neighbors"]) == 5
    assert set(body["features"]) == {"lat", "lng", "excluUseAr", "age_years", "floor_feat",
                                     "month_idx", "knn_ppm2_5", "same_bldg"}
    assert body["features"]["age_years"] == 20
    assert body["features"]["month_idx"] == 11


@pytest.mark.parametrize("patch", [
    {"lat": 35.1},                 # 서울 밖
    {"as_of": "2025-13"},          # 잘못된 월
    {"build_year": 2026},          # 기준 연도보다 늦은 건축년도
    {"area_m2": 0},                # 면적 0
])
def test_price_validation(client, patch):
    r = client.post("/v1/price", json={**BASE, **patch})
    assert r.status_code == 422


def test_extrapolation_flag(client):
    r = client.post("/v1/price", json={**BASE, "as_of": "2026-09"})
    assert any("외삽" in f for f in r.json()["flags"])


def test_recovery_matches_core(client):
    req = {**BASE, "deposit_won": 80_000_000, "ead_won": 150_000_000, "months": [12, 24]}
    r = client.post("/v1/recovery", json=req)
    assert r.status_code == 200
    body = r.json()
    assert len(body["grid"]) == 3 * 2
    assert body["ead_assumed"] is False
    price = client.post("/v1/price", json=BASE).json()["price_won"]
    row = next(g for g in body["grid"] if g["rho_scenario"] == "기본" and g["months"] == 24)
    expected = compute_recovery(row["S"] / row["rho"], row["rho"], 80_000_000, 150_000_000, 24)
    assert row["LGD"] == pytest.approx(expected["LGD"])
    assert row["S"] / row["rho"] == pytest.approx(price, abs=1)


def test_recovery_ead_assumed(client):
    r = client.post("/v1/recovery", json=BASE)
    body = r.json()
    assert body["ead_assumed"] is True
    assert body["ead_won"] == pytest.approx(body["price_won"] * 0.70, abs=1)
    assert any("EAD" in f for f in body["flags"])
