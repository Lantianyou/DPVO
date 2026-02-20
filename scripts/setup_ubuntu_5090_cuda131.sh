#!/usr/bin/env bash
set -euo pipefail

# Quick setup for Ubuntu + RTX 5090 + CUDA 13.1 (no viewer).
#
# Optional environment variables:
#   ENV_NAME=dpvo5090
#   PYTHON_VERSION=3.10
#   CUDA_HOME=/usr/local/cuda-13.1
#   TORCH_CUDA_ARCH_LIST=12.0
#   SKIP_APT=0                # set to 1 to skip apt dependencies
#   SKIP_DATA=0               # set to 1 to skip model/data download
#   RUN_DEMO=0                # set to 1 to run demo after setup

ENV_NAME="${ENV_NAME:-dpvo5090}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_DATA="${SKIP_DATA:-0}"
RUN_DEMO="${RUN_DEMO:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[1/7] System checks"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | sed -n '1,20p'
else
  echo "ERROR: nvidia-smi not found. Run this script on the Ubuntu 5090 machine."
  exit 1
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | sed -n '1,20p'
else
  echo "WARNING: nvcc not found in PATH. Will rely on CUDA_HOME=${CUDA_HOME}."
fi

if [[ "${SKIP_APT}" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  echo "[2/7] Installing system dependencies"
  APT_INSTALL="apt-get install -y build-essential git wget unzip ninja-build"
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo bash -lc "${APT_INSTALL}"
  else
    apt-get update
    bash -lc "${APT_INSTALL}"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required but not found."
  exit 1
fi

echo "[3/7] Creating conda env: ${ENV_NAME}"
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

echo "[4/7] Installing Python dependencies"
pip install -U pip setuptools wheel
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
pip install numpy==1.26.4 numba tqdm einops yacs opencv-python matplotlib scipy plyfile evo

echo "[5/7] Building DPVO extensions"
mkdir -p thirdparty
if [[ ! -d thirdparty/eigen-3.4.0 ]]; then
  wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
  unzip -q -o eigen-3.4.0.zip -d thirdparty
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST
pip install -v .

if [[ "${SKIP_DATA}" != "1" ]]; then
  echo "[6/7] Downloading model and sample data"
  ./download_models_and_data.sh
fi

echo "[7/7] Smoke test"
python - <<'PY'
import torch
import cuda_corr
import cuda_ba
import lietorch_backends

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("extensions: ok")
PY

if [[ "${RUN_DEMO}" == "1" ]]; then
  if [[ ! -f movies/IMG_0492.MOV ]]; then
    echo "ERROR: movies/IMG_0492.MOV not found. Re-run with SKIP_DATA=0."
    exit 1
  fi
  python demo.py --imagedir=movies/IMG_0492.MOV --calib=calib/iphone.txt --stride=5 --plot
fi

echo "Setup complete."
