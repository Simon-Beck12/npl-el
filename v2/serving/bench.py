"""EXP-014 5·11단계: 서빙 경로 성능 측정(API 단건, 배치 전체, 메모리, 비교용 parquet).

측정 항목(EXP-014 프롬프트 5단계 그대로):
- API 단건 응답 시간: FastAPI TestClient로 /valuation 1,000회 호출(워밍업 50회 제외), p50·p95·평균(ms).
  app.py에 있는 "현재" 경로를 그대로 측정한다(이 스크립트 자체는 old/new를 구분하지 않는다).
- 배치 처리 시간: 테스트 7,561건 전체의 "시세 추정"(predict_valuation 또는 predict_valuation_batch)
  시간(초)과 초당 처리 행 수. 병목(cKDTree 재생성)이 있는 부분만 재서, 전후 비교가 그 지점을 바로 보게 한다.
  회수액(compute_recovery_v2, 6줄 산술)은 두 경로 모두 동일해서 재지 않는다.
- 모델을 불러온 직후(배치·API 호출 전) 프로세스 RSS(MB).
- 테스트 7,561건의 행별 시세(market_price)·knn_ppm2_5·same_bldg를 비교용 parquet에 저장한다(로컬 전용·커밋 안 함).

실행(측정 조건 고정 — OMP_NUM_THREADS=1로 LightGBM num_threads=1 효과를 낸다):
  OMP_NUM_THREADS=1 uv run python bench.py --path old --out results/bench_before.json \
      --compare-parquet /tmp/.../compare_before.parquet
  OMP_NUM_THREADS=1 uv run python bench.py --path new --out results/bench_after.json \
      --compare-parquet /tmp/.../compare_after.parquet
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

SERVING_DIR = Path(__file__).resolve().parent
ROOT = SERVING_DIR.parent.parent
sys.path.insert(0, str(SERVING_DIR))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "v2"))

WARMUP = 50
N_API_CALLS = 1000

VALUATION_PAYLOAD = {
    "lat": 37.6036, "lng": 126.9589, "excluUseAr": 36.5, "floor": 3,
    "buildYear": 2005, "dealYear": 2026, "dealMonth": 9,
    "gu_name": "종로구", "umdNm": "신영동", "jibun": "213-1",
}


def _rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _cpu_model():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if line.startswith("모델 이름:") or line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def measurement_conditions(path, batch_n):
    import lightgbm
    import scipy
    mem = psutil.virtual_memory()
    return {
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "mem_total_gb": round(mem.total / (1024 ** 3), 1),
        "python_version": platform.python_version(),
        "lightgbm_version": lightgbm.__version__,
        "scipy_version": scipy.__version__,
        "lightgbm_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "api_warmup": WARMUP,
        "api_n_calls": N_API_CALLS,
        "batch_n_rows": batch_n,
        "path": path,
    }


def load_test_set():
    from price_model import load_v2_split
    _, _, _, test = load_v2_split()
    return test.reset_index(drop=True)


def bench_api():
    import app as app_module
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    for _ in range(WARMUP):
        r = client.post("/valuation", json=VALUATION_PAYLOAD)
        assert r.status_code == 200
    latencies = []
    for _ in range(N_API_CALLS):
        t0 = time.perf_counter()
        r = client.post("/valuation", json=VALUATION_PAYLOAD)
        latencies.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    arr = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
    }


def bench_valuation_batch(test, ref, booster, path):
    """시세 추정 단계만 잰다(회수액 제외). old: 행마다 predict_valuation 호출(cKDTree 재생성).
    new: KnnIndex를 1번 만들고 predict_valuation_batch를 1번 호출한다."""
    from valuation_core import build_target_row, build_target_frame, predict_valuation, predict_valuation_batch

    n = len(test)
    if path == "old":
        t0 = time.perf_counter()
        rows = []
        for _, r in test.iterrows():
            target = build_target_row(
                lat=r["lat"], lng=r["lng"], excluUseAr=r["excluUseAr"], floor=r["floor"],
                buildYear=int(r["buildYear"]), dealYear=int(r["dealYear"]), dealMonth=int(r["dealMonth"]),
                gu_name=r["gu_name"], umdNm=r["umdNm"], jibun=r["jibun"],
            )
            rows.append(predict_valuation(target, ref, booster))
        elapsed = time.perf_counter() - t0
        result = pd.DataFrame(rows)
    elif path == "new":
        from model_loader import KnnIndex
        t0 = time.perf_counter()
        knn_index = KnnIndex(ref)
        targets = build_target_frame(
            lat=test["lat"], lng=test["lng"], excluUseAr=test["excluUseAr"], floor=test["floor"],
            buildYear=test["buildYear"], dealYear=test["dealYear"], dealMonth=test["dealMonth"],
            gu_name=test["gu_name"], umdNm=test["umdNm"], jibun=test["jibun"],
        )
        result = predict_valuation_batch(targets, knn_index, booster)
        elapsed = time.perf_counter() - t0
    else:
        raise ValueError(path)
    return elapsed, result.reset_index(drop=True)


def test_mdape(test, market_price):
    actual = test["dealAmount"].to_numpy()
    ape = np.abs(actual - market_price) / actual
    return float(np.median(ape) * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=["old", "new"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare-parquet", default=None,
                         help="행별 deal_id·시세·knn_ppm2_5·same_bldg 저장 경로(로컬 전용, 커밋 안 함)")
    args = parser.parse_args()

    mem_after_import = _rss_mb()

    from model_loader import load_model

    booster, ref, meta = load_model()
    mem_after_load = _rss_mb()

    test = load_test_set()

    api = bench_api()
    elapsed, val_df = bench_valuation_batch(test, ref, booster, args.path)
    mdape = test_mdape(test, val_df["market_price"].to_numpy())

    result = {
        "path": args.path,
        "api": api,
        "batch_valuation": {
            "n_rows": len(test),
            "elapsed_sec": elapsed,
            "rows_per_sec": len(test) / elapsed,
        },
        "test_mdape": mdape,
        "memory_rss_mb": {
            "after_import": mem_after_import,
            "after_model_load": mem_after_load,
        },
        "conditions": measurement_conditions(args.path, len(test)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.compare_parquet:
        cmp_df = pd.DataFrame({
            "deal_id": test["deal_id"].to_numpy(),
            "시세": val_df["market_price"].to_numpy(),
            "knn_ppm2_5": val_df["knn_ppm2_5"].to_numpy(),
            "same_bldg": val_df["same_bldg"].to_numpy(),
        })
        cmp_path = Path(args.compare_parquet)
        cmp_path.parent.mkdir(parents=True, exist_ok=True)
        cmp_df.to_parquet(cmp_path, index=False)
        print(f"비교용 parquet 저장(로컬 전용): {cmp_path}")


if __name__ == "__main__":
    main()
