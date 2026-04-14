#!/bin/bash

# Define environment name
ENV_NAME="vision_pipeline_env"

echo "[1/6] Creating Python venv: $ENV_NAME with Python 3.10..."
python3.10 -m venv $ENV_NAME
source $ENV_NAME/bin/activate

echo "[2/6] Installing PyTorch with CUDA support..."
# Adjust 'cu121' to your specific CUDA version if necessary
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[3/6] Installing core CV and utility dependencies..."
pip install opencv-python matplotlib pillow numpy scipy tqdm requests

echo "[4/6] Installing Hugging Face Transformers & Accelerate..."
# Accelerate is required for 'device_map="auto"' or 'device_map="cpu"' in Llama models
pip install transformers accelerate bitsandbytes

echo "[5/6] Installing Segment Anything 2 (SAM 2)..."
# SAM 2 requires a manual build from the repository for the latest predictor features
if [ ! -d "sam2" ]; then
    git clone https://github.com/facebookresearch/segment-anything-2.git
    cd segment-anything-2
    pip install -e .
    cd ..
else
    echo "SAM 2 directory already exists, skipping clone."
fi

echo "[6/6] Finalizing installation..."
# Ensure all dependencies for GroundingDINO (via Transformers) are met
pip install pycocotools

echo "--------------------------------------------------------"
echo "Setup Complete!"
echo "Activate your environment with: source $ENV_NAME/bin/activate"
echo "Note: Ensure you have access to meta-llama/Llama-3.2-8B-Instruct on Hugging Face."
echo "--------------------------------------------------------"