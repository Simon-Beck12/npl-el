"""EXP-011 6단계: artifact/에서 모델·기준 데이터·메타데이터를 읽는다."""
import json
import os
from pathlib import Path

import lightgbm as lgb
import pandas as pd

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "artifact"


def load_model(model_dir=None):
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR") or DEFAULT_MODEL_DIR)
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    ref = pd.read_parquet(model_dir / "ref.parquet")
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    return booster, ref, meta
