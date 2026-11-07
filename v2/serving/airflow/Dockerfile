# EXP-013 5단계. apache/airflow 공식 이미지 기반. 쓴 버전: apache/airflow:3.4.0 (Airflow 3.4.0, Python 3.13).
FROM apache/airflow:3.4.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# v2/serving/ 코드를 돌리는 전용 가상환경(Airflow 자체 패키지와 섞지 않는다).
RUN python3 -m venv /opt/npl-venv && chown -R airflow:root /opt/npl-venv
COPY --chown=airflow:root v2/serving/requirements.txt /opt/npl-venv-requirements.txt

USER airflow
RUN /opt/npl-venv/bin/pip install --no-cache-dir -r /opt/npl-venv-requirements.txt
# EXP-013 수정: score 태스크가 external_python 서브프로세스에서 pendulum을 직접 import한다.
# npl-venv는 requirements.txt(API용)만 설치해 pendulum이 없었다. Airflow 이미지에 이미 설치된
# 버전(pendulum 3.3.0, lazy_object_proxy 1.13.0, docker exec python -c "import importlib.metadata as m; print(m.version('pendulum'), m.version('lazy_object_proxy'))" 로 확인)과 맞춰 설치한다.
RUN /opt/npl-venv/bin/pip install --no-cache-dir pendulum==3.3.0 lazy_object_proxy==1.13.0
