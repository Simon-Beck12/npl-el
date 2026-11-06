FROM python:3.12-slim

# LightGBM 실행에 OpenMP 런타임이 필요하다
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recovery_core.py predictor.py app.py ./
# export_artifacts.py로 만든 artifact(model.txt, reference.csv, meta.json)를 이미지에 넣는다
COPY artifacts/model.txt artifacts/reference.csv artifacts/meta.json ./artifacts/

ENV NPL_ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
