"""EXP-012 7단계: build_artifact.py와 같은 방법으로 G2를 학습·저장하고, MLflow에 기록·등록한다.
EXP-014 13단계 수정: --skip-build를 주면 재학습하지 않고 기존 artifact/(model.txt·ref.parquet·
meta.json, EXP-008 G2 그대로)만 다시 로그한다 — 코드(pyfunc_model.py의 KnnIndex 벡터화 경로)만 바뀐
새 버전을 등록할 때 쓴다. 등록 뒤에는 방금 등록한 모델을 실제로 불러와 테스트 7,561건 MdAPE를
다시 계산해 기준(11.5635%, ±0.0001%p)을 확인한 뒤에만 champion 별칭을 옮긴다.

MLFLOW_TRACKING_URI 환경변수로 가리키는 MLflow 서버(기본 http://localhost:5000, docker compose의
mlflow 서비스를 host 포트로 연 것)에 run을 남기고, npl-el-price 이름으로 등록해 champion 별칭을 붙인다.
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema

SERVING_DIR = Path(__file__).resolve().parent
ROOT = SERVING_DIR.parent.parent
sys.path.insert(0, str(SERVING_DIR))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "v2"))

from build_artifact import main as build_artifact_main, ARTIFACT_DIR  # noqa: E402

REGISTERED_MODEL_NAME = "npl-el-price"
ALIAS = "champion"
PYFUNC_MODEL_PATH = str(SERVING_DIR / "pyfunc_model.py")
MDAPE_REF = 11.5635
MDAPE_TOLERANCE = 0.0001

CODE_PATHS = [
    str(ROOT / "src" / "features.py"),
    str(ROOT / "src" / "split.py"),
    str(ROOT / "src" / "knn_comps.py"),
    str(ROOT / "v2" / "price_model.py"),
    str(SERVING_DIR / "valuation_core.py"),
    str(SERVING_DIR / "fast_features.py"),
]


def verify_mdape(model_uri):
    """EXP-014 13단계: 방금 등록한 모델을 실제로 불러와 테스트 7,561건으로 MdAPE를 다시 잰다
    (meta.json에 저장된 값을 그냥 믿지 않는다 — 코드 경로가 바뀌었으므로 실제로 다시 확인한다)."""
    from price_model import load_v2_split

    _, _, _, test = load_v2_split()
    model = mlflow.pyfunc.load_model(model_uri)
    cols = ["lat", "lng", "excluUseAr", "floor", "buildYear", "dealYear", "dealMonth",
            "gu_name", "umdNm", "jibun"]
    model_input = test[cols].copy()
    model_input["floor"] = model_input["floor"].astype(float)  # 모델 시그니처가 floor를 double로 둠(INPUT_EXAMPLE 3.0 기준)
    for c in ("buildYear", "dealYear", "dealMonth"):
        model_input[c] = model_input[c].astype(int)  # load_v2_split()은 이 컬럼들을 문자열로 준다
    pred = model.predict(model_input)
    actual = test["dealAmount"].to_numpy()
    ape = np.abs(actual - pred["market_price"].to_numpy()) / actual
    mdape = float(np.median(ape) * 100)
    return mdape, len(test)

INPUT_EXAMPLE = pd.DataFrame([{
    "lat": 37.6036, "lng": 126.9589, "excluUseAr": 36.5, "floor": 3.0,
    "buildYear": 2005, "dealYear": 2026, "dealMonth": 9,
    "gu_name": "종로구", "umdNm": "신영동", "jibun": "213-1",
}])

# floor는 실데이터(floor_feat)에서 float64라 double로 둔다. 구·umdNm·jibun은 /valuation처럼
# 선택 입력이라 required=False로 둔다(자동 추론은 예시 1행만 보고 전부 필수·int로 잘못 고정한다).
SIGNATURE = ModelSignature(
    inputs=Schema([
        ColSpec(DataType.double, "lat"),
        ColSpec(DataType.double, "lng"),
        ColSpec(DataType.double, "excluUseAr"),
        ColSpec(DataType.double, "floor"),
        ColSpec(DataType.long, "buildYear"),
        ColSpec(DataType.long, "dealYear"),
        ColSpec(DataType.long, "dealMonth"),
        ColSpec(DataType.string, "gu_name", required=False),
        ColSpec(DataType.string, "umdNm", required=False),
        ColSpec(DataType.string, "jibun", required=False),
    ]),
    outputs=Schema([
        ColSpec(DataType.double, "market_price"),
        ColSpec(DataType.double, "price_per_m2"),
        ColSpec(DataType.double, "knn_ppm2_5"),
        ColSpec(DataType.long, "same_bldg"),
    ]),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true",
                         help="재학습하지 않는다. artifact/의 model.txt·ref.parquet·meta.json을 그대로 쓴다"
                              "(EXP-014: 코드만 바뀐 새 버전을 등록할 때).")
    args = parser.parse_args()

    if not args.skip_build:
        build_artifact_main()
    else:
        for f in ("model.txt", "ref.parquet", "meta.json"):
            assert (ARTIFACT_DIR / f).exists(), f"--skip-build인데 {ARTIFACT_DIR / f} 없음 — 멈춤"
    meta = json.loads((ARTIFACT_DIR / "meta.json").read_text(encoding="utf-8"))

    print(f"[MLflow] tracking_uri={mlflow.get_tracking_uri()}")

    run_name = "EXP-014 G2(fast_features, 재학습 없음)" if args.skip_build else "EXP-012 G2"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "num_leaves": meta["lgbm_params"]["num_leaves"],
            "min_child_samples": meta["lgbm_params"]["min_child_samples"],
            "learning_rate": meta["lgbm_params"]["learning_rate"],
            "num_boost_round": meta["best_iteration"],
            "feature_cols": ",".join(meta["feature_cols"]),
            "train_period_start": meta["train_period"]["start"],
            "train_period_end": meta["train_period"]["end"],
            "test_period_start": meta["test_period"]["start"],
            "test_period_end": meta["test_period"]["end"],
            "git_commit": meta["git_commit"],
            "exp014_skip_build": args.skip_build,
        })
        mlflow.log_metrics({
            "test_mdape": meta["test_mdape"],
            "test_within_10pct": meta["test_pct10"],
            "n_train": meta["n_train"],
            "n_test": meta["test_mdape_n"],
        })

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=PYFUNC_MODEL_PATH,
            artifacts={
                "model_txt": str(ARTIFACT_DIR / "model.txt"),
                "ref_parquet": str(ARTIFACT_DIR / "ref.parquet"),
                "meta_json": str(ARTIFACT_DIR / "meta.json"),
            },
            code_paths=CODE_PATHS,
            input_example=INPUT_EXAMPLE,
            signature=SIGNATURE,
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        run_id = run.info.run_id

    version = model_info.registered_model_version
    client = mlflow.MlflowClient()

    mdape, n_test = verify_mdape(f"models:/{REGISTERED_MODEL_NAME}/{version}")
    diff = abs(mdape - MDAPE_REF)
    print(f"[검증] v{version} 재계산 MdAPE={mdape:.6f}%(n={n_test}), 기준={MDAPE_REF}%, 차이={diff:.6f}%p")
    if diff > MDAPE_TOLERANCE:
        print(f"[중단] 차이 {diff:.6f}%p > {MDAPE_TOLERANCE}%p — champion을 옮기지 않는다. 멈추고 보고한다.")
        raise SystemExit(1)

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS, version)
    print(f"[MLflow] 등록 완료: {REGISTERED_MODEL_NAME} v{version} run_id={run_id} alias={ALIAS}")


if __name__ == "__main__":
    main()
