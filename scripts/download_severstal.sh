#!/usr/bin/env bash
set -euo pipefail

echo "=============================="
echo " QualForge AI - Severstal Setup"
echo "=============================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"

mkdir -p "${RAW_DIR}"
mkdir -p "${PROJECT_ROOT}/scripts"

echo "[1/6] Project root: ${PROJECT_ROOT}"
echo "[2/6] Raw dataset dir: ${RAW_DIR}"

# --- check python
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ python3 not found. Install python3 first."
  exit 1
fi

# --- check kaggle
echo "[3/6] Checking Kaggle CLI..."
if ! python3 -c "import kaggle" >/dev/null 2>&1; then
  echo "Kaggle CLI not found in this environment. Installing..."
  python3 -m pip install --upgrade pip >/dev/null
  python3 -m pip install kaggle >/dev/null
fi
echo "✅ Kaggle CLI ready."

# --- check kaggle.json
KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
if [ ! -f "${KAGGLE_JSON}" ]; then
  echo "❌ Kaggle API key not found at: ${KAGGLE_JSON}"
  echo ""
  echo "Fix:"
  echo "1) Go to Kaggle -> Account -> API -> Create New Token"
  echo "2) It downloads kaggle.json"
  echo "3) Run:"
  echo "   mkdir -p ~/.kaggle"
  echo "   mv /path/to/kaggle.json ~/.kaggle/"
  echo "   chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

chmod 600 "${KAGGLE_JSON}"
echo "✅ Kaggle key found."

# --- download dataset
ZIP_PATH="${RAW_DIR}/severstal-steel-defect-detection.zip"

echo "[4/6] Downloading Severstal dataset via Kaggle API..."
cd "${PROJECT_ROOT}"

# If already downloaded, skip
if [ -f "${ZIP_PATH}" ]; then
  echo "✅ Zip already exists: ${ZIP_PATH}"
else
  kaggle competitions download -c severstal-steel-defect-detection -p "${RAW_DIR}"
  echo "✅ Download complete."
fi

# --- unzip dataset
echo "[5/6] Unzipping dataset..."
cd "${RAW_DIR}"

# Unzip only if needed
if [ -f "${RAW_DIR}/train.csv" ] && [ -d "${RAW_DIR}/train_images" ]; then
  echo "✅ Dataset already extracted: train.csv + train_images/ exists."
else
  unzip -o "${ZIP_PATH}" >/dev/null
  echo "✅ Unzip complete."
fi

# sanity check
if [ ! -f "${RAW_DIR}/train.csv" ]; then
  echo "❌ train.csv not found in ${RAW_DIR}"
  exit 1
fi

if [ ! -d "${RAW_DIR}/train_images" ]; then
  echo "❌ train_images folder not found in ${RAW_DIR}"
  exit 1
fi

echo "✅ Dataset structure verified:"
echo "   - data/raw/train.csv"
echo "   - data/raw/train_images/"

# --- run preparation
echo "[6/6] Running dataset preparation (Severstal -> YOLOv8-seg)..."
cd "${PROJECT_ROOT}"

python3 -m src.dataset.severstal_prepare

echo "=============================="
echo "✅ DONE!"
echo "Next:"
echo "python -m src.training.train_yolo_seg --model yolov8s-seg.pt --epochs 30 --imgsz 640"
echo "=============================="
