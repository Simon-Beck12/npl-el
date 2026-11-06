"""recovery_core.compute_recovery가 v2/recovery.compute_recovery_v2와 같은 값을 내는지 확인한다."""
import sys
from pathlib import Path

import numpy as np
import pytest

from recovery_core import compute_recovery

V2_ROOT = Path(__file__).resolve().parents[2]


def _original():
    try:
        sys.path.insert(0, str(V2_ROOT.parent / "src"))
        sys.path.insert(0, str(V2_ROOT))
        from recovery import compute_recovery_v2
        return compute_recovery_v2
    except Exception as e:  # faiss 등 학습 환경 패키지가 없을 때
        pytest.skip(f"v2/recovery.py를 불러올 수 없다: {e}")


def test_parity_random_2000():
    orig = _original()
    rng = np.random.default_rng(7)
    for _ in range(2000):
        args = dict(
            market_price=float(rng.uniform(5e7, 1.5e9)),
            rho=float(rng.uniform(0.3, 1.6)),
            deposit=float(rng.choice([0, rng.uniform(1e7, 4e8), 165_000_000, 55_000_000])),
            ead=float(rng.uniform(1e7, 1e9)),
            months=int(rng.choice([1, 6, 12, 24, 36, 60])),
        )
        a, b = compute_recovery(**args), orig(**args)
        for key in a:
            assert a[key] == pytest.approx(b[key], rel=1e-12, abs=1e-6), (key, args)


def test_hand_calc_no_deposit():
    """v2/tests/test_recovery_v2.py test_1과 같은 손계산."""
    r = compute_recovery(market_price=100_000_000, rho=0.7, deposit=0, ead=70_000_000, months=24)
    assert r["R"] == pytest.approx(69_300_000)
    assert r["PV"] == pytest.approx(69_300_000 / 1.08 ** 2)
