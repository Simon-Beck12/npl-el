# 빌드 컨텍스트는 저장소 루트다(docker-compose.yml의 build.context 참고).
# src/, v2/price_model.py·recovery.py는 v2/serving/ 코드가 import하는 의존 모듈이라 함께 복사한다.
# 모델 파일(artifact/)과 데이터(data/)는 이미지에 넣지 않고 compose volume으로 연결한다.
FROM python:3.12-slim

WORKDIR /app

COPY v2/serving/requirements.txt v2/serving/requirements.txt
RUN pip install --no-cache-dir -r v2/serving/requirements.txt

COPY src/features.py src/split.py src/knn_comps.py src/
COPY v2/price_model.py v2/recovery.py v2/
COPY v2/serving/model_loader.py v2/serving/valuation_core.py v2/serving/app.py \
     v2/serving/batch_score.py v2/serving/build_artifact.py v2/serving/

ENV MODEL_DIR=/app/artifact
WORKDIR /app/v2/serving
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
