"""EXP-011 4단계: v2 G2 모델을 artifact/로 내보낸다.

v2/price_model.py의 함수를 그대로 import해서 쓴다(수정 없음).
모델 설정(num_leaves=63, min_child_samples=20, best_iteration=2554)은
EXP-008 채택값(v2/data/processed/exp008_model_choice.json)을 그대로 쓴다.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "v2"))

from price_model import load_v2_split, build_lgbm_frame, fit_final_lgbm, mdape_price, FULL8  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifact"
MODEL_CHOICE_JSON = ROOT / "v2" / "data" / "processed" / "exp008_model_choice.json"
REF_COLUMNS = ["deal_id", "gu_name", "umdNm", "jibun", "loc_x_km", "loc_y_km", "price_per_m2"]
REPRO_TOLERANCE = 0.01  # %p, EXP-011 재현 기준


def git_commit_hash():
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def main():
    model_choice = json.loads(MODEL_CHOICE_JSON.read_text())
    assert model_choice["adopted"] == "G2", f"EXP-008 채택 모델이 G2가 아님: {model_choice['adopted']} — 멈춤"
    best = {
        "params": {
            "num_leaves": model_choice["lgbm_params"]["num_leaves"],
            "min_child_samples": model_choice["lgbm_params"]["min_child_samples"],
        },
        "best_iteration": model_choice["best_iteration"],
    }
    assert best["params"] == {"num_leaves": 63, "min_child_samples": 20}
    assert best["best_iteration"] == 2554

    train_all, core, val, test = load_v2_split()
    print(f"train_all={len(train_all):,} core={len(core):,} val={len(val):,} test={len(test):,}")

    booster = fit_final_lgbm(train_all, best, FULL8)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(ARTIFACT_DIR / "model.txt"))

    train_f = build_lgbm_frame(train_all, train_all, exclude_self=True)
    ref = train_f[REF_COLUMNS].reset_index(drop=True)
    ref.to_parquet(ARTIFACT_DIR / "ref.parquet")

    # 5단계 재현 확인: 방금 메모리에 있던 booster/train_all이 아니라
    # 디스크에 저장한 model.txt·ref.parquet만 다시 읽어 테스트 7,561건을 재예측한다.
    reloaded_booster = lgb.Booster(model_file=str(ARTIFACT_DIR / "model.txt"))
    reloaded_ref = pd.read_parquet(ARTIFACT_DIR / "ref.parquet")
    test_f = build_lgbm_frame(test, reloaded_ref, exclude_self=False)
    pred_log = reloaded_booster.predict(test_f[FULL8])
    pred_price = np.exp(pred_log) * test_f["excluUseAr"].to_numpy()
    test_mdape, test_pct10, _, test_n_na = mdape_price(test["dealAmount"].to_numpy(), pred_price)

    ref_test_mdape = model_choice["test_mdape"]["G2"]
    diff = abs(test_mdape - ref_test_mdape)
    print(f"[재현 확인] artifact 재예측 MdAPE={test_mdape:.6f}%, EXP-008 등록값={ref_test_mdape:.6f}%, 차이={diff:.6f}%p")
    if diff > REPRO_TOLERANCE:
        raise SystemExit(
            f"재현 확인 실패: 차이 {diff:.6f}%p > {REPRO_TOLERANCE}%p — 멈춤 (오류로 멈춤, 직접 고쳐 재시작하지 않음)"
        )
    print(f"[재현 확인] 통과 (허용 오차 {REPRO_TOLERANCE}%p)")

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "model": "G2(8변수, EXP-008 채택)",
        "feature_cols": FULL8,
        "lgbm_params": model_choice["lgbm_params"],
        "best_iteration": best["best_iteration"],
        "train_period": {"start": str(train_all["contract_date"].min().date()),
                          "end": str(train_all["contract_date"].max().date())},
        "n_train": len(train_all),
        "n_ref": len(ref),
        "test_mdape": test_mdape,
        "test_mdape_ref_exp008": ref_test_mdape,
        "test_mdape_n": len(test),
        "test_mdape_n_na": test_n_na,
    }
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {ARTIFACT_DIR / 'model.txt'}, {ARTIFACT_DIR / 'ref.parquet'}, {ARTIFACT_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
