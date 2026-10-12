FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch keeps the image small; swap the index URL for a CUDA build if you have a GPU host.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[postgres,mlflow,catboost]"

ENV DATABASE_URL=sqlite:////data/clinicalfusion.db CF_ARTIFACT_DIR=/data/artifacts
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "clinicalfusion.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
