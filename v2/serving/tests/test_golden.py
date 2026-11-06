"""predictor.py 피처 사본 + 저장된 모델이 학습 코드 경로와 같은 예측을 내는지 확인한다.

- synthetic: 가짜 artifact. 기대값은 price_model.build_lgbm_frame으로 계산(불러올 수 있을 때).
- real: v2/serving/artifacts/golden.json이 있으면(export_artifacts.py 실행 후) 실제 모델로 확인.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from predictor import Predictor

REAL_DIR = Path(__file__).resolve().parents[1] / "artifacts"
REL_TOL = 1e-6


def _to_item(inp):
    d = dict(inp)
    y, m = d.pop("as_of").split("-")
    d["as_of_year"], d["as_of_month"] = int(y), int(m)
    return d


def _check(artifacts_dir):
    p = Predictor(artifacts_dir)
    golden = json.loads((Path(artifacts_dir) / "golden.json").read_text(encoding="utf-8"))
    rel = []
    for g in golden:
        got = p.predict_price(_to_item(g["input"]))["price_manwon"]
        rel.append(abs(got - g["expected_price_manwon"]) / g["expected_price_manwon"])
    rel = np.array(rel)
    assert len(rel) > 0
    assert rel.max() < REL_TOL, f"최대 상대오차 {rel.max():.2e}, 불일치 {int((rel >= REL_TOL).sum())}/{len(rel)}건"
    return len(rel), float(rel.max())


def test_golden_synthetic(synthetic_artifacts):
    if not synthetic_artifacts["golden_from_price_model"]:
        pytest.skip("price_model을 불러오지 못해(faiss 등 미설치) 원래 피처 코드와 비교할 수 없다")
    _check(synthetic_artifacts["dir"])


@pytest.mark.skipif(not (REAL_DIR / "golden.json").exists(), reason="실제 artifact 없음(export_artifacts.py 실행 전)")
def test_golden_real():
    n, max_rel = _check(REAL_DIR)
    print(f"real golden {n}건, 최대 상대오차 {max_rel:.2e}")
