"""EXP-011: EXP-008 채택 모델(G2)을 다시 학습해 서빙용 artifact로 저장하고 재현을 확인한다.

학습 데이터·하이퍼파라미터·반복 수는 EXP-008 최종 모델과 같다(v2/data/processed/exp008_model_choice.json).
기존 v2/price_model.py의 함수를 그대로 쓴다. 기존 파일은 수정하지 않는다.

확인 3가지 (하나라도 실패하면 멈춘다):
1. 재학습 모델의 테스트 MdAPE가 EXP-008 기록값과 0.01%p 이내로 같다.
2. 저장한 model.txt를 다시 불러온 예측이 메모리 모델 예측과 같다(상대오차 1e-9 이내).
3. 낙찰가율 시나리오(rho)가 EXP-009 기록값(소수 4자리)과 같다.

실행 (저장소 루트에서):  uv run python v2/serving/export_artifacts.py
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = ROOT / "v2"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V2_ROOT))

import features as feat_mod  # noqa: E402
from split import TRAIN_START, TRAIN_END  # noqa: E402
from price_model import (  # noqa: E402
    FULL8, KNN_K, MONTH0, bldg_key, fit_final_lgbm, load_v2_split, mdape_price, month_index, predict_lgbm,
)

PROCESSED_DIR = V2_ROOT / "data" / "processed"
MODEL_CHOICE_JSON = PROCESSED_DIR / "exp008_model_choice.json"
GRID_PARQUET = PROCESSED_DIR / "exp009_recovery_v2.parquet"
PORTFOLIO_V2_PARQUET = PROCESSED_DIR / "portfolio_v2.parquet"
DEFAULT_OUT = V2_ROOT / "serving" / "artifacts"

MDAPE_TOL_PP = 0.01
RELOAD_TOL = 1e-9
RHO_RECORDED = {"비관": 0.5741, "기본": 0.6446, "낙관": 0.8042}  # v2/results/exp009_recovery.md 표2
RHO_TOL = 5e-5
GOLDEN_N = 200
SEED = 42
KST = timezone(timedelta(hours=9))


def stop(msg):
    raise SystemExit(f"[멈춤] {msg}")


def git_short_hash():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_rho_scenarios():
    grid = pd.read_parquet(GRID_PARQUET)
    port = pd.read_parquet(PORTFOLIO_V2_PARQUET)[["cltrMngNo", "v2_price"]]
    g = grid.merge(port, on="cltrMngNo", how="inner")
    g["rho"] = g["S"] / g["v2_price"]
    rho = g.groupby("낙찰가율시나리오")["rho"].median().to_dict()
    spread = g.groupby("낙찰가율시나리오")["rho"].agg(lambda s: s.max() - s.min()).max()
    if spread > 1e-9:
        stop(f"같은 시나리오 안에서 rho 값이 다르다(최대 차이 {spread}) — 계산 확인 필요")
    for name, recorded in RHO_RECORDED.items():
        if abs(rho[name] - recorded) > RHO_TOL:
            stop(f"rho {name}={rho[name]:.6f}가 기록값 {recorded}와 다르다")
    return {name: float(rho[name]) for name in ["비관", "기본", "낙관"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true", help="기존 artifact 덮어쓰기 허용")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.force:
        stop(f"{out}에 이미 파일이 있다. 덮어쓰려면 --force를 붙인다(멈춤 규칙 4).")

    with open(MODEL_CHOICE_JSON, encoding="utf-8") as f:
        choice = json.load(f)
    if choice["adopted"] != "G2" or choice["feature_cols"] != FULL8:
        stop(f"EXP-008 채택 모델이 G2(8변수)가 아니다: {choice['adopted']}")
    lp = choice["lgbm_params"]
    best = {"params": {"num_leaves": lp["num_leaves"], "min_child_samples": lp["min_child_samples"]},
            "best_iteration": choice["best_iteration"]}
    print(f"[1/5] EXP-008 설정: {best}")

    train_all, _, _, test = load_v2_split()
    print(f"[2/5] train_all={len(train_all):,} test={len(test):,}")

    booster = fit_final_lgbm(train_all, best, FULL8)
    pred, test_f = predict_lgbm(booster, test, train_all, FULL8)
    mdape, pct10, _, n_na = mdape_price(test["dealAmount"].to_numpy(), pred)
    recorded = choice["test_mdape"]["G2"]
    diff = abs(mdape - recorded)
    print(f"[3/5] 재현 MdAPE={mdape:.4f}% 기록={recorded:.4f}% 차이={diff:.4f}%p (허용 {MDAPE_TOL_PP}%p)")
    if diff > MDAPE_TOL_PP:
        stop("재학습 모델의 테스트 MdAPE가 EXP-008 기록과 다르다")

    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.txt"
    booster.save_model(str(model_path))
    reloaded = lgb.Booster(model_file=str(model_path))
    p_mem = booster.predict(test_f[FULL8])
    p_file = reloaded.predict(test_f[FULL8])
    rel = float(np.max(np.abs(p_file - p_mem) / np.maximum(np.abs(p_mem), 1e-12)))
    print(f"[4/5] model.txt 재로드 예측 최대 상대오차={rel:.2e} (허용 {RELOAD_TOL:.0e})")
    if rel > RELOAD_TOL:
        stop("저장한 모델의 예측이 메모리 모델과 다르다")

    rho = load_rho_scenarios()
    print(f"      rho 시나리오={rho}")

    # reference: train_all 순서를 그대로 유지한다(cKDTree 이웃 선택이 학습 때와 같아지도록).
    ref = pd.DataFrame({
        "loc_x_km": train_all["loc_x_km"].to_numpy(),
        "loc_y_km": train_all["loc_y_km"].to_numpy(),
        "price_per_m2": train_all["price_per_m2"].to_numpy(),
        "bldg_key": bldg_key(train_all).to_numpy(),
    })
    ref.to_csv(out / "reference.csv", index=False, float_format="%.17g")

    sample_idx = np.random.default_rng(SEED).choice(len(test), size=min(GOLDEN_N, len(test)), replace=False)
    t = test.reset_index(drop=True).iloc[sample_idx]
    golden = []
    for i, (_, r) in zip(sample_idx, t.iterrows()):
        if pd.isna(r["floor"]) or pd.isna(r["buildYear"]):
            continue  # API 입력은 층·건축년도 필수라 결측 행은 golden에서 뺀다
        golden.append({
            "input": {
                "lat": float(r["lat"]), "lng": float(r["lng"]), "area_m2": float(r["excluUseAr"]),
                "build_year": int(r["buildYear"]), "floor": int(float(r["floor"])),
                "gu": str(r["gu_name"]), "dong": str(r["umdNm"]).strip(), "jibun": str(r["jibun"]).strip(),
                "as_of": f"{int(r['dealYear']):04d}-{int(r['dealMonth']):02d}",
            },
            "expected_price_manwon": float(pred[i]),
        })
    with open(out / "golden.json", "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False)

    meta = {
        "model_name": "EXP-008 G2 LightGBM 8변수",
        "model_version": f"exp008-g2-{git_short_hash()}",
        "created_at_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "feature_cols": FULL8,
        "knn_k": KNN_K,
        "month0": list(MONTH0),
        "max_month_idx": int(month_index(train_all).max()),
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "geo": {"ref_lat": feat_mod.REF_LAT, "ref_lng": feat_mod.REF_LNG,
                "km_per_lat_deg": feat_mod.KM_PER_LAT_DEG, "km_per_lng_deg": feat_mod.KM_PER_LNG_DEG},
        "area_q01_q99": [float(train_all["excluUseAr"].quantile(0.01)),
                         float(train_all["excluUseAr"].quantile(0.99))],
        "test_mdape_pct": mdape,
        "test_mdape_recorded_pct": recorded,
        "test_pct_within_10": pct10,
        "test_n": int(len(test)),
        "rho_scenarios": rho,
        "rho_source": "v2/data/processed/exp009_recovery_v2.parquet (EXP-009 표2)",
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    sizes = {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in sorted(out.iterdir())}
    print(f"[5/5] 저장 완료: {out}  {sizes}")
    print(f"      golden {len(golden)}건, reference {len(ref):,}건, n_결측={n_na}")


if __name__ == "__main__":
    main()
