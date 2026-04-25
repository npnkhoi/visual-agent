@echo off
setlocal enabledelayedexpansion

REM -----------------------------------------------------------------------------
REM setup_omnicount_env.bat
REM
REM Usage:
REM   setup_omnicount_env.bat
REM   setup_omnicount_env.bat --repo-dir OmniCount --venv .venv --python python --torch cu121 --skip-checkpoints
REM
REM Torch options: cu121 | cu118 | cpu
REM -----------------------------------------------------------------------------

set "REPO_DIR=OmniCount"
set "VENV_DIR=.venv"
set "PYTHON_BIN=python"
set "TORCH_VARIANT=cu121"
set "SKIP_CHECKPOINTS=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--repo-dir" (
  set "REPO_DIR=%~2"
  shift & shift
  goto parse_args
)
if /I "%~1"=="--venv" (
  set "VENV_DIR=%~2"
  shift & shift
  goto parse_args
)
if /I "%~1"=="--python" (
  set "PYTHON_BIN=%~2"
  shift & shift
  goto parse_args
)
if /I "%~1"=="--torch" (
  set "TORCH_VARIANT=%~2"
  shift & shift
  goto parse_args
)
if /I "%~1"=="--skip-checkpoints" (
  set "SKIP_CHECKPOINTS=1"
  shift
  goto parse_args
)

echo Unknown argument: %~1
exit /b 1

:args_done

echo [1/8] Clone repo (if missing) + init submodules
if not exist "%REPO_DIR%\.git" (
  git clone --recursive https://github.com/mondalanindya/OmniCount.git "%REPO_DIR%"
  if errorlevel 1 exit /b 1
)

cd /d "%REPO_DIR%"
if errorlevel 1 exit /b 1

git submodule update --init --recursive
if errorlevel 1 exit /b 1

echo [2/8] Create virtual environment
if not exist "%VENV_DIR%\Scripts\activate.bat" (
  %PYTHON_BIN% -m venv "%VENV_DIR%"
  if errorlevel 1 exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

echo [3/8] Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

echo [4/8] Install PyTorch (%TORCH_VARIANT%)
if /I "%TORCH_VARIANT%"=="cu121" (
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if /I "%TORCH_VARIANT%"=="cu118" (
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if /I "%TORCH_VARIANT%"=="cpu" (
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
  echo Invalid --torch value: %TORCH_VARIANT% ^(expected: cu121^|cu118^|cpu^)
  exit /b 1
)
if errorlevel 1 exit /b 1

echo [5/8] Install OmniCount requirements
pip install -r requirements.txt
if errorlevel 1 exit /b 1

if exist "external\SAN\requirements.txt" (
  pip install -r external\SAN\requirements.txt
)
if exist "external\GroundingDINO\requirements.txt" (
  pip install -r external\GroundingDINO\requirements.txt
)

echo [6/8] Install extras for LLM decomposition + pipeline
pip install transformers>=4.43.0 accelerate bitsandbytes sentencepiece safetensors scipy scikit-image opencv-python pillow requests tqdm
if errorlevel 1 exit /b 1

echo [7/8] Download checkpoints (optional)
if "%SKIP_CHECKPOINTS%"=="0" (
  if exist "scripts\download_checkpoints.sh" (
    where bash >nul 2>nul
    if errorlevel 1 (
      echo bash not found. Skipping checkpoint shell script.
      echo You can run scripts\download_checkpoints.sh manually in Git Bash/WSL.
    ) else (
      bash scripts/download_checkpoints.sh
      if errorlevel 1 (
        echo Checkpoint script returned non-zero. You can re-run it manually later.
      )
    )
  ) else (
    echo scripts\download_checkpoints.sh not found, skipping.
  )
) else (
  echo Skipping checkpoint download by request.
)

echo [8/8] Smoke test
python -c "import torch,sys; print('Python:',sys.version.split()[0]); print('Torch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); import transformers; print('Transformers:',transformers.__version__); print('Environment setup looks OK.')"
if errorlevel 1 exit /b 1

echo.
echo Done
echo.
echo To activate later:
echo   call %REPO_DIR%\%VENV_DIR%\Scripts\activate.bat
echo.
echo If using Llama-3.1-8B-Instruct:
echo   huggingface-cli login
echo (ensure your account has model access)
echo.

endlocal
exit /b 0