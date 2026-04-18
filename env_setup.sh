#!/bin/bash

# Define environment name
ENV_NAME="vision_pipeline_env"

echo "[1/7] Checking for Python 3.13..."
# Check if python3.13 is available
if ! command -v python3.13 &> /dev/null
then
    echo "[!] Python 3.13 not found. Please install it using: sudo apt install python3.13 (on Ubuntu) or brew install python@3.13 (on Mac)"
    exit 1
fi

echo "[2/7] Creating Python venv: $ENV_NAME with Python 3.13..."
python3.13 -m venv $ENV_NAME

echo "[3/7] Activating environment..."
source $ENV_NAME/bin/activate

echo "[4/7] Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "[5/7] Installing PyTorch 2.11 with CUDA 12.8 support..."
# Using cu128 for PyTorch 2.11
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "[6/7] Installing core CV and utility dependencies..."
pip install opencv-python matplotlib pillow numpy scipy tqdm requests

echo "[7/7] Installing Hugging Face Transformers & Accelerate..."
pip install transformers accelerate bitsandbytes

echo "[8/8] Installing Segment Anything 2 (SAM 2)..."
if [ ! -d "sam2" ]; then
    echo "Cloning SAM 2 repository..."
    git clone https://github.com/facebookresearch/segment-anything-2.git
    cd segment-anything-2
    pip install -e .
    cd ..
else
    echo "SAM 2 directory already exists, skipping clone."
fi

# Ensure pycocotools is installed for GroundingDINO
pip install pycocotools

echo "--------------------------------------------------------"
echo "Setup Complete!"
echo "Python Version: $(python --version)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Activate your environment with: source $ENV_NAME/bin/activate"
echo "--------------------------------------------------------"