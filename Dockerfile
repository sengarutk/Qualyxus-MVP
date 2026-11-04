FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONFIG_PATH=/app/config.yaml \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CAMERA_STATUS_PATH=/app/data/live/camera_status.json \
    MODEL_ASSET_PATH=/app/models/best.pt \
    PYTHONPATH=/app \
    LOW_MEMORY_MODE=0 \
    MAX_CAMERA_WORKERS=0 \
    LOW_MEMORY_MAX_CAMERAS=1 \
    TORCH_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# curl is required by the docker-compose HTTP healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg libsm6 libxext6 libgl1 curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better Docker layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /app/requirements.txt

# Copy only required runtime source/configs.
COPY main.py /app/main.py
COPY src /app/src
COPY scripts /app/scripts
COPY config.yaml config.360.yaml /app/

# Run without root. Bind-mounted data/, models/ and secrets/ must be writable by uid 1000.
RUN useradd --uid 1000 --create-home qualyxus \
  && mkdir -p /app/data /app/models /app/secrets \
  && chown -R qualyxus /app
USER qualyxus

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -m src.healthcheck || exit 1

# main.py is the full edge runtime: camera ingestion, batched inference,
# event persistence, the Pulse SSE API (port 8000) and the MQTT uplink.
CMD ["python", "main.py"]
