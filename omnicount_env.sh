#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# setup_omnicount_env.sh
#
# Usage examples:
#   bash setup_omnicount_env.sh
#   bash setup_omnicount_env.sh --repo-dir OmniCount --venv .venv --torch cu121
#   bash setup_omnicount_env.sh --torch cpu
#
# Notes:
# - For Llama-3.1-8B-Instruct, make sure you have access on Hugging Face.
# - If using gated models, run `huggingface-cli login` after activation.
# -----------------------------------------------------------------------------

REPO_DIR="OmniCount"
VENV_DIR=".venv"
PYTHON_BIN="python3"
TORCH_VARIANT="cu121"   # options: cu121, cu118, cpu
SKIP_CHECKPOINTS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --torch) TORCH_VARIANT="$2"; shift 2 ;;
    --skip-checkpoints) SKIP_CHECKPOINTS="1"; shift 1 ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "[1/8] Clone repo (if missing) + init submodules"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --recursive https://github.com/mondalanindya/OmniCount.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git submodule update --init --recursive

echo "[2/8] Create virtual environment"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[3/8] Upgrade pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[4/8] Install PyTorch (${TORCH_VARIANT})"
if [[ "$TORCH_VARIANT" == "cu121" ]]; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$TORCH_VARIANT" == "cu118" ]]; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "$TORCH_VARIANT" == "cpu" ]]; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  echo "Invalid --torch value: $TORCH_VARIANT (expected: cu121|cu118|cpu)"
  exit 1
fi

echo "[5/8] Install OmniCount requirements"
pip install -r requirements.txt

# Install submodule requirements if present
[[ -f external/SAN/requirements.txt ]] && pip install -r external/SAN/requirements.txt || true
[[ -f external/GroundingDINO/requirements.txt ]] && pip install -r external/GroundingDINO/requirements.txt || true

echo "[6/8] Install extras for LLM decomposition + pipeline"
pip install \
  "transformers>=4.43.0" \
  accelerate \
  bitsandbytes \
  sentencepiece \
  safetensors \
  scipy \
  scikit-image \
  opencv-python \
  pillow \
  requests \
  tqdm

echo "[7/8] Download checkpoints (optional)"
if [[ "$SKIP_CHECKPOINTS" == "0" ]]; then
  if [[ -f scripts/download_checkpoints.sh ]]; then
    bash scripts/download_checkpoints.sh || {
      echo "Checkpoint script returned non-zero. You can re-run it manually later."
    }
  else
    echo "scripts/download_checkpoints.sh not found, skipping."
  fi
else
  echo "Skipping checkpoint download by request."
fi

echo "[8/8] Smoke test"
python - << 'PY'
import torch, sys
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
try:
    import transformers
    print("Transformers:", transformers.__version__)
except Exception as e:
    print("Transformers import failed:", e)
print("Environment setup looks OK.")
PY

cat << 'MSG'

Done ✅

Activate environment:
  source .venv/bin/activate

If you use Llama-3.1-8B-Instruct:
  huggingface-cli login
(ensure your account has model access)

Then run your pipeline script as usual.
MSG