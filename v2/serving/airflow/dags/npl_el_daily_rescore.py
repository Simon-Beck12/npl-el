"""EXP-013 6단계: champion 모델(models:/npl-el-price@champion)로 매일 02:00(Asia/Seoul)
배치 재채점을 하고 결과를 MLflow npl-el-batch 실험에 기록하는 DAG.

check_input → score → validate_output → log_to_mlflow 순서로 4개 작업을 잇는다.
score·log_to_mlflow는 /opt/npl-venv(v2/serving/requirements.txt 전용 가상환경)에서 돌린다
(ExternalPythonOperator). v2/price_model.py, v2/recovery.py, src/ 는 수정하지 않고
volume으로 마운트된 코드를 그대로 import한다.
"""
import csv
import os

import pendulum
from airflow.exceptions import AirflowException
from airflow.sdk import dag, task

V2_SERVING_DIR = "/opt/npl-el/v2/serving"
OUTPUT_DIR = "/opt/npl-el/output"
NPL_VENV_PYTHON = "/opt/npl-venv/bin/python"
DEFAULT_INPUT_CSV = f"{V2_SERVING_DIR}/examples/sample_input.csv"
REQUIRED_COLUMNS = ["lat", "lng", "excluUseAr", "floor", "buildYear", "dealYear", "dealMonth",
                    "구", "umdNm", "jibun", "deposit", "ead", "months", "rho_scenario"]


def _input_csv_path():
    """입력 경로: NPL_EL_INPUT_CSV 환경변수, 없으면 examples/sample_input.csv."""
    return os.environ.get("NPL_EL_INPUT_CSV", DEFAULT_INPUT_CSV)


@dag(
    dag_id="npl_el_daily_rescore",
    schedule="0 2 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Seoul"),
    catchup=False,
    default_args={"retries": 1},
    tags=["npl-el", "v2", "EXP-013"],
    # EXP-013 수정: 기본값(True)이면 수동 트리거한 run도 unpause 전까지 queued에서 멈춘다.
    is_paused_upon_creation=False,
)
def npl_el_daily_rescore():

    @task
    def check_input() -> str:
        path = _input_csv_path()
        if not os.path.exists(path):
            raise AirflowException(f"입력 파일이 없음: {path}")
        with open(path, encoding="utf-8") as f:
            header = next(csv.reader(f))
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise AirflowException(f"필수 컬럼 없음: {missing} (파일: {path}, 헤더: {header})")
        return path

    @task.external_python(python=NPL_VENV_PYTHON)
    def score(input_path: str, output_dir: str, serving_dir: str) -> dict:
        import os as _os
        import sys

        import pandas as pd
        import pendulum as _pendulum

        sys.path.insert(0, serving_dir)
        from batch_score import DEFAULT_MODEL_URI, score_row  # noqa: E402
        from model_loader import load_model  # noqa: E402

        booster, ref, meta = load_model(model_uri=DEFAULT_MODEL_URI)
        df = pd.read_csv(input_path)
        scored = df.apply(lambda row: score_row(row, ref, booster), axis=1)
        out = pd.concat([df, scored], axis=1)

        _os.makedirs(output_dir, exist_ok=True)
        today = _pendulum.now("Asia/Seoul").format("YYYY-MM-DD")
        output_path = f"{output_dir}/rescore_{today}.csv"
        out.to_csv(output_path, index=False)

        n_error = int((out["오류"] != "").sum())
        return {
            "output_path": output_path,
            "input_path": input_path,
            "n_rows": int(len(out)),
            "n_error": n_error,
            "model_version": meta.get("version"),
            "run_id": meta.get("run_id"),
        }

    @task
    def validate_output(score_result: dict) -> dict:
        input_path = score_result["input_path"]
        output_path = score_result["output_path"]

        with open(input_path, encoding="utf-8") as f:
            n_input = sum(1 for _ in csv.reader(f)) - 1  # 헤더 제외

        with open(output_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if len(rows) != n_input:
            raise AirflowException(f"입력·출력 행 수가 다름: 입력 {n_input}건, 출력 {len(rows)}건")

        for r in rows:
            if r.get("오류"):
                continue
            if r.get("시세") in (None, ""):
                raise AirflowException(f"시세 결측(오류 표시 없는 행): {r}")
            lgd = float(r["LGD"])
            if not (0.0 <= lgd <= 1.0):
                raise AirflowException(f"LGD가 0~1 범위 밖: {lgd} (행: {r})")

        return {"n_input": n_input, "n_output": len(rows)}

    @task.external_python(python=NPL_VENV_PYTHON)
    def log_to_mlflow(score_result: dict) -> None:
        import csv as _csv

        import mlflow

        mlflow.set_experiment("npl-el-batch")
        with open(score_result["output_path"], encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))

        valid = [r for r in rows if not r.get("오류")]
        lgds = [float(r["LGD"]) for r in valid]
        pvs = [float(r["PV"]) for r in valid]
        eads = [float(r["ead"]) for r in valid]
        avg_lgd = sum(lgds) / len(lgds) if lgds else float("nan")
        portfolio_lgd = (1 - sum(pvs) / sum(eads)) if eads and sum(eads) > 0 else float("nan")

        with mlflow.start_run(run_name="npl_el_daily_rescore"):
            mlflow.log_params({
                "model_version": score_result.get("model_version"),
                "input_file": score_result["input_path"].rsplit("/", 1)[-1],
            })
            mlflow.log_metrics({
                "n_rows": score_result["n_rows"],
                "n_error": score_result["n_error"],
                "avg_lgd": avg_lgd,
                "portfolio_lgd": portfolio_lgd,
            })

    input_path = check_input()
    score_result = score(input_path=input_path, output_dir=OUTPUT_DIR, serving_dir=V2_SERVING_DIR)
    validate_output(score_result=score_result) >> log_to_mlflow(score_result=score_result)


npl_el_daily_rescore()
