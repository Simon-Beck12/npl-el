"""EXP-011 6단계(1단계)/EXP-012 9단계(2단계 추가): artifact/ 또는 MLflow 레지스트리에서
모델·기준 데이터·메타데이터를 읽는다.

MODEL_URI가 있으면 mlflow.pyfunc.load_model로 불러온다(예: models:/npl-el-price@champion).
없으면 1단계처럼 MODEL_DIR(기본 artifact/)에서 model.txt·ref.parquet·meta.json을 직접 읽는다.
두 경로 모두 (booster, ref, meta) 같은 모양으로 돌려줘서 app.py·batch_score.py는 경로를 몰라도 된다.
"""
import json
import os
from pathlib import Path

import lightgbm as lgb
import pandas as pd

# EXP-014 9단계: app.py·batch_score.py가 model_loader를 통해서만 KnnIndex를 쓰도록
# 여기서 다시 내보낸다(ref로 cKDTree·건물별 개수를 1번만 만들어 재사용하는 경로).
from fast_features import KnnIndex  # noqa: F401

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "artifact"


def load_model(model_dir=None, model_uri=None):
    model_uri = model_uri or os.environ.get("MODEL_URI")
    if model_uri:
        return _load_from_mlflow(model_uri)
    return _load_from_dir(model_dir)


def _load_from_dir(model_dir):
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR") or DEFAULT_MODEL_DIR)
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    ref = pd.read_parquet(model_dir / "ref.parquet")
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    meta = {**meta, "source": "MODEL_DIR", "model_dir": str(model_dir),
            "registered_model_name": None, "alias": None, "version": None, "run_id": None}
    return booster, ref, meta


def _load_from_mlflow(model_uri):
    import mlflow
    from mlflow import MlflowClient

    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    inner = pyfunc_model.unwrap_python_model()

    meta = dict(inner.meta)
    meta["source"] = "MLFLOW"
    meta["model_uri"] = model_uri
    meta["run_id"] = pyfunc_model.metadata.run_id
    meta["registered_model_name"], meta["alias"], meta["version"] = None, None, None

    name, alias = _parse_alias_uri(model_uri)
    if name and alias:
        mv = MlflowClient().get_model_version_by_alias(name, alias)
        meta["registered_model_name"] = name
        meta["alias"] = alias
        meta["version"] = mv.version

    return inner.booster, inner.ref, meta


def _parse_alias_uri(model_uri):
    """models:/<name>@<alias> 형태만 파싱한다. 그 외 형태(버전 번호 등)는 (None, None)."""
    if model_uri.startswith("models:/") and "@" in model_uri:
        name, alias = model_uri[len("models:/"):].split("@", 1)
        return name, alias
    return None, None
