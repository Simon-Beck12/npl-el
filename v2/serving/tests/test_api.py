"""EXP-011 9단계: FastAPI TestClient 기반 API 테스트.

효창동 예시 기대값(LGD 58.6%, ±0.1%p)은 compute_recovery_v2(market_price=48974,
rho=0.644620, deposit=14700, ead=34281.8, months=24)를 직접 계산해 미리 확인한 값이다.
"""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402
from model_loader import load_model  # noqa: E402

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


def test_model_loader_mlflow_path():
    """EXP-012 12단계: MLflow 경로(models:/npl-el-price@champion) 확인.
    MLFLOW_TRACKING_URI가 가리키는 서버가 없으면(로컬 파일 경로만 쓰는 환경) 건너뛴다."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        pytest.skip("MLFLOW_TRACKING_URI 미설정 — MLflow 경로 테스트 건너뜀")
    try:
        booster, ref, meta = load_model(model_uri="models:/npl-el-price@champion")
    except Exception as exc:
        pytest.skip(f"MLflow 서버에 연결할 수 없어 건너뜀: {exc}")
    assert meta["source"] == "MLFLOW"
    assert meta["registered_model_name"] == "npl-el-price"
    assert meta["alias"] == "champion"
    assert meta["run_id"]
    assert abs(meta["test_mdape"] - 11.5635) < 0.01
