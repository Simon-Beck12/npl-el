"""EXP-012 6단계(EXP-014 8단계 수정): MLflow pyfunc 래퍼.

계산은 valuation_core.py·fast_features.py를 그대로 쓴다. 이 파일은 code_paths로 모델과
함께 로그되므로 features.py/split.py/knn_comps.py/price_model.py/valuation_core.py/
fast_features.py가 모두 같은 code 디렉터리에 평평하게(flat) 놓인다고 가정한다(모듈 이름으로만
import한다).
"""
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd

from fast_features import KnnIndex
from valuation_core import build_target_frame, predict_valuation_batch


class NplPriceModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.booster = lgb.Booster(model_file=context.artifacts["model_txt"])
        self.ref = pd.read_parquet(context.artifacts["ref_parquet"])
        self.meta = json.loads(Path(context.artifacts["meta_json"]).read_text(encoding="utf-8"))
        # EXP-014: cKDTree·건물별 개수를 ref로 1번만 만든다(행마다 새로 만들지 않는다).
        self.knn_index = KnnIndex(self.ref)

    def predict(self, context, model_input, params=None):
        targets = build_target_frame(
            lat=model_input["lat"], lng=model_input["lng"], excluUseAr=model_input["excluUseAr"],
            floor=model_input["floor"], buildYear=model_input["buildYear"],
            dealYear=model_input["dealYear"], dealMonth=model_input["dealMonth"],
            gu_name=model_input.get("gu_name"), umdNm=model_input.get("umdNm"), jibun=model_input.get("jibun"),
        )
        return predict_valuation_batch(targets, self.knn_index, self.booster)


# "models from code" 방식으로 로그한다(cloudpickle로 객체를 그대로 직렬화하는 대신 이 파일을
# 서빙 환경에서 다시 실행해 모델을 만든다) — API 컨테이너처럼 학습 때와 패키지 구성이 조금 달라도
# 더 안정적으로 로드된다.
mlflow.models.set_model(NplPriceModel())
