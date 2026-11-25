# One file, two targets:
#   x86 PC with an NVIDIA GPU:  docker build -t qualyxus .
#   Jetson (JetPack 6, arm64):  docker build -t qualyxus --target jetson .
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app \
    CONFIG_PATH=/app/config.yaml HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

FROM base AS app-files
COPY main.py config.yaml config.360.yaml /app/
COPY src /app/src

# NVIDIA's own image brings the CUDA and PyTorch builds for Jetson; PyPI has none for it.
# Not yet run on a Jetson. Pin the tag to the JetPack on the box.
ARG JETSON_BASE=dustynv/l4t-pytorch:r36.4.0
FROM ${JETSON_BASE} AS jetson
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    CONFIG_PATH=/app/config.yaml HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /app
COPY requirements.in /tmp/
RUN grep -vE '^(#|torch|torchvision|onnxruntime|opencv-python-headless)' /tmp/requirements.in > /tmp/jetson.txt \
  && python3 -m pip install --no-cache-dir -r /tmp/jetson.txt \
  && python3 -c "import torch, cv2; assert torch.version.cuda, 'this base image has no CUDA build of PyTorch'"
COPY --from=app-files /app /app
RUN useradd --uid 1000 --create-home qualyxus && mkdir -p data models secrets && chown -R qualyxus /app
USER qualyxus
HEALTHCHECK --interval=30s --timeout=20s --start-period=90s --retries=3 CMD ["python3", "-m", "src.healthcheck"]
STOPSIGNAL SIGINT
CMD ["python3", "main.py"]

FROM base AS pc
COPY requirements.txt /app/
RUN pip install --require-hashes -r requirements.txt
COPY --from=app-files /app /app
# Without root. Bind-mounted data/, models/ and secrets/ must be writable by uid 1000.
RUN useradd --uid 1000 --create-home qualyxus && mkdir -p data models secrets && chown -R qualyxus data models secrets
USER qualyxus
HEALTHCHECK --interval=30s --timeout=20s --start-period=90s --retries=3 CMD ["python", "-m", "src.healthcheck"]
STOPSIGNAL SIGINT
CMD ["python", "main.py"]
