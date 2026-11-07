"""EXP-011 9단계: FastAPI TestClient 기반 API 테스트.

효창동 예시 기대값(LGD 58.6%, ±0.1%p)은 compute_recovery_v2(market_price=48974,
rho=0.644620, deposit=14700, ead=34281.8, months=24)를 직접 계산해 미리 확인한 값이다.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402
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
        "gu_name": "종로구", "umdNm": "신영동", "jibun": "213-1",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["market_price"] > 0
    assert body["same_bldg"] in (0, 1)


@pytest.mark.parametrize("payload", [
    {"lat": 37.6036, "lng": 126.9589, "excluUseAr": 36.5, "floor": 3,
     "buildYear": 2005, "dealYear": 2026, "dealMonth": 7},  # 구·umdNm·jibun 아예 없음
    {"lat": 37.6036, "lng": 126.9589, "excluUseAr": 36.5, "floor": 3,
     "buildYear": 2005, "dealYear": 2026, "dealMonth": 7,
     "gu_name": "", "umdNm": "신영동", "jibun": "213-1"},  # 빈 문자열
])
def test_valuation_missing_address_422(payload):
    """EXP-013 4단계: 주소(구·umdNm·jibun) 없는 /valuation 요청은 422다."""
    r = client.post("/valuation", json=payload)
    assert r.status_code == 422


def test_knn_index_matches_price_model_functions():
    """EXP-014 12단계(새 테스트 1): fast_features.KnnIndex(ref로 1번만 만듦)의 knn_ppm2·same_bldg
    결과가 v2/price_model.py의 add_knn_ppm2·add_same_bldg(exclude_self=False, 서빙이 쓰는 경로)와 같아야 한다."""
    from fast_features import KnnIndex
    from price_model import add_knn_ppm2, add_same_bldg

    ref = app_module._ref
    target = ref.head(20).reset_index(drop=True)  # 실제 ref 데이터 앞 20행(좌표·주소 모두 있음)
    knn_index = KnnIndex(ref)

    got_knn = knn_index.knn_ppm2(target)
    want_knn = add_knn_ppm2(target, ref, exclude_self=False)
    assert np.allclose(got_knn, want_knn)

    got_same = knn_index.same_bldg(target)
    want_same = add_same_bldg(target, ref, exclude_self=False)
    assert np.array_equal(got_same, want_same)


def test_predict_valuation_batch_matches_single_row():
    """EXP-014 12단계(새 테스트 2): predict_valuation_batch()로 여러 행을 한 번에 예측한 결과가
    predict_valuation()을 행마다 돌린 결과와 같아야 한다(examples/sample_input.csv 3행)."""
    from valuation_core import build_target_row, build_target_frame, predict_valuation, predict_valuation_batch

    samples = [
        {"lat": 37.6036, "lng": 126.9589, "excluUseAr": 36.5, "floor": 3, "buildYear": 2005,
         "dealYear": 2026, "dealMonth": 9, "gu_name": "종로구", "umdNm": "신영동", "jibun": "213-**"},
        {"lat": 37.5805, "lng": 126.9886, "excluUseAr": 42.1, "floor": 2, "buildYear": 2010,
         "dealYear": 2026, "dealMonth": 9, "gu_name": "종로구", "umdNm": "원서동", "jibun": "135-**"},
        {"lat": 37.6116, "lng": 126.9586, "excluUseAr": 29.8, "floor": 4, "buildYear": 1998,
         "dealYear": 2026, "dealMonth": 9, "gu_name": "종로구", "umdNm": "구기동", "jibun": "138-**"},
    ]
    ref = app_module._ref
    booster = app_module._booster
    knn_index = app_module._knn_index

    single_results = [predict_valuation(build_target_row(**s), ref, booster) for s in samples]

    batch_targets = build_target_frame(
        lat=[s["lat"] for s in samples], lng=[s["lng"] for s in samples],
        excluUseAr=[s["excluUseAr"] for s in samples], floor=[s["floor"] for s in samples],
        buildYear=[s["buildYear"] for s in samples], dealYear=[s["dealYear"] for s in samples],
        dealMonth=[s["dealMonth"] for s in samples], gu_name=[s["gu_name"] for s in samples],
        umdNm=[s["umdNm"] for s in samples], jibun=[s["jibun"] for s in samples],
    )
    batch_result = predict_valuation_batch(batch_targets, knn_index, booster)

    for i, single in enumerate(single_results):
        assert abs(single["market_price"] - batch_result["market_price"].iloc[i]) < 1e-6
        assert abs(single["knn_ppm2_5"] - batch_result["knn_ppm2_5"].iloc[i]) < 1e-6
        assert single["same_bldg"] == batch_result["same_bldg"].iloc[i]


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
