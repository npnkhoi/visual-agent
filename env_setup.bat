@echo off
setlocal

:: Define environment name
set ENV_NAME=vision_pipeline_env

echo [1/9] Checking for Python 3.13...
py -3.13 -m venv %ENV_NAME%

if %errorlevel% neq 0 (
    echo [!] Python 3.13 not found. Please install it from python.org first.
    pause
    exit /b
)

echo [2/9] Activating environment: %ENV_NAME%...
call %ENV_NAME%\Scripts\activate

echo [3/9] Upgrading pip and installer tools...
python -m pip install --upgrade pip "setuptools<82" wheel

echo [4/9] Installing PyTorch 2.11 with CUDA 12.8 support...
:: Correct index for stable CUDA 12.8 builds
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo [5/9] Installing core CV and utility dependencies...
pip install opencv-python matplotlib pillow numpy scipy tqdm requests

echo [6/9] Installing Hugging Face Transformers ^& Accelerate...
pip install transformers accelerate bitsandbytes

echo [7/9] Installing Segment Anything 2.1 (SAM 2.1)...
if not exist "segment-anything-2" (
    echo Cloning SAM 2 repository...
    git clone https://github.com/facebookresearch/segment-anything-2.git
    cd segment-anything-2
    pip install -e .
    cd ..
) else (
    echo SAM 2 directory already exists, skipping clone.
)

echo [8/9] Downloading SAM 2.1 Checkpoints (Windows PowerShell)...
:: Create checkpoints folder if it doesn't exist
if not exist "segment-anything-2\checkpoints" mkdir "segment-anything-2\checkpoints"

:: Use PowerShell to download the specific 2.1 Hiera-Large weights
powershell -Command "& {Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt' -OutFile 'segment-anything-2\checkpoints\sam2.1_hiera_large.pt'}"

echo [9/9] Finalizing installation...
pip install pycocotools

echo --------------------------------------------------------
echo Setup Complete!
echo Python Version: 3.13
echo PyTorch Version: 2.11 (CUDA 12.8)
echo SAM 2.1 Model: Hiera-Large downloaded.
echo.
echo Activate your environment with: call %ENV_NAME%\Scripts\activate
echo --------------------------------------------------------

pause