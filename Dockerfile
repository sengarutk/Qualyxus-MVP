FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONFIG_PATH=/app/config.yaml \
    CAMERA_STATUS_PATH=/app/data/live/camera_status.json \
    MODEL_ASSET_PATH=/app/models/best.pt \
    PYTHONPATH=/app \
    LOW_MEMORY_MODE=0 \
    MAX_CAMERA_WORKERS=0 \
    LOW_MEMORY_MAX_CAMERAS=1 \
    TORCH_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg libsm6 libxext6 libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better Docker layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /app/requirements.txt \
  && pip install --no-cache-dir fastapi uvicorn plotly

# Copy only required runtime source/configs.
COPY src /app/src
COPY scripts /app/scripts
COPY config.yaml /app/config.yaml

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -m src.healthcheck || exit 1

CMD ["python", "-m", "src.inference.infer_live", "--config", "/app/config.yaml"]
