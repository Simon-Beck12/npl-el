"""EXP-012 7단계: build_artifact.py와 같은 방법으로 G2를 학습·저장하고, MLflow에 기록·등록한다.

MLFLOW_TRACKING_URI 환경변수로 가리키는 MLflow 서버(기본 http://localhost:5000, docker compose의
mlflow 서비스를 host 포트로 연 것)에 run을 남기고, npl-el-price 이름으로 등록해 champion 별칭을 붙인다.
"""
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema

SERVING_DIR = Path(__file__).resolve().parent
ROOT = SERVING_DIR.parent.parent
sys.path.insert(0, str(SERVING_DIR))

from build_artifact import main as build_artifact_main, ARTIFACT_DIR  # noqa: E402

REGISTERED_MODEL_NAME = "npl-el-price"
ALIAS = "champion"
PYFUNC_MODEL_PATH = str(SERVING_DIR / "pyfunc_model.py")

CODE_PATHS = [
    str(ROOT / "src" / "features.py"),
    str(ROOT / "src" / "split.py"),
    str(ROOT / "src" / "knn_comps.py"),
    str(ROOT / "v2" / "price_model.py"),
    str(SERVING_DIR / "valuation_core.py"),
]

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
    build_artifact_main()
    meta = json.loads((ARTIFACT_DIR / "meta.json").read_text(encoding="utf-8"))

    print(f"[MLflow] tracking_uri={mlflow.get_tracking_uri()}")

    with mlflow.start_run(run_name="EXP-012 G2") as run:
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
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS, version)
    print(f"[MLflow] 등록 완료: {REGISTERED_MODEL_NAME} v{version} run_id={run_id} alias={ALIAS}")


if __name__ == "__main__":
    main()
