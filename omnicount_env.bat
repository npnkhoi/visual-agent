@echo off
setlocal enabledelayedexpansion

REM -----------------------------------------------------------------------------
REM setup_omnicount_env.bat  -  Python 3.10 + CUDA 12.1 + Torch 2.1.2
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
goto parse_args

:args_done

echo [1/9] Clone repo (if missing) + init submodules
if not exist "%REPO_DIR%\.git" (
  git clone --recursive https://github.com/mondalanindya/OmniCount.git "%REPO_DIR%"
  if errorlevel 1 call :error_handler "Git clone failed"
)

cd /d "%REPO_DIR%"
if errorlevel 1 call :error_handler "Failed to enter directory %REPO_DIR%"

git submodule update --init --recursive
if errorlevel 1 call :error_handler "Submodule update failed"

echo [2/9] Create virtual environment (Python 3.10)
if not exist "%VENV_DIR%\Scripts\activate.bat" (
  "%PYTHON_BIN%" -m venv "%VENV_DIR%"
  if errorlevel 1 call :error_handler "Venv creation failed. Make sure Python 3.10 is installed and on PATH."
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 call :error_handler "Failed to activate venv"

echo [3/9] Upgrade pip + build tools
python -m pip install --upgrade pip wheel
:: Pin setuptools to <70 - setuptools 70+ dropped pkg_resources which torch's cpp_extension needs
python -m pip install "setuptools==69.5.1"
if errorlevel 1 call :error_handler "Pip/Setuptools upgrade failed"

echo [4/9] Install PyTorch 2.1.2 (%TORCH_VARIANT%)
if /I "%TORCH_VARIANT%"=="cu121" (
  pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
) else if /I "%TORCH_VARIANT%"=="cu118" (
  pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
) else if /I "%TORCH_VARIANT%"=="cpu" (
  pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
) else (
  call :error_handler "Invalid Torch variant: %TORCH_VARIANT%. Use cu121, cu118, or cpu."
)
if errorlevel 1 call :error_handler "Torch installation failed"

echo [5/9] Install OmniCount requirements
:: Pin numpy - 1.26.4 is compatible with Python 3.10 + detectron2 + SAN
pip install "numpy==1.26.4"
if errorlevel 1 call :error_handler "numpy install failed"

:: Install MMCV - open-mmlab has cp310 wheels for torch2.1/cu121
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
if errorlevel 1 (
  echo [!] Pre-built MMCV wheel not found. Trying PyPI...
  pip install mmcv==2.1.0
  if errorlevel 1 echo [!] MMCV install failed. Continuing - may cause issues if SAN needs it.
)

:: Install main OmniCount requirements
pip install -r requirements.txt
if errorlevel 1 call :error_handler "Main requirements.txt failed"

:: SAN requirements - filter out pinned numpy==1.22.4 (incompatible with Py3.10 build env)
if exist "external\SAN\requirements.txt" (
  :: Exclude numpy and mmcv; install mmcv separately with no-build-isolation
  findstr /v /i "numpy mmcv" external\SAN\requirements.txt > external\SAN\requirements_no_numpy_no_mmcv.txt
  pip install -r external\SAN\requirements_no_numpy_no_mmcv.txt
  if errorlevel 1 echo [!] SAN requirements (without mmcv) failed. Check manually.

  pip install --force-reinstall "setuptools==69.5.1"
  if errorlevel 1 call :error_handler "setuptools pin failed before SAN mmcv install"

  pip install --no-build-isolation "mmcv==1.3.14"
  if errorlevel 1 call :error_handler "SAN mmcv==1.3.14 install failed"
)

if exist "external\GroundingDINO\requirements.txt" (
  pip install -r external\GroundingDINO\requirements.txt
  if errorlevel 1 echo [!] GroundingDINO requirements failed.
)

echo [6/9] Install detectron2 from source
if not exist "external\detectron2\.git" (
  git clone https://github.com/facebookresearch/detectron2.git external\detectron2
  if errorlevel 1 call :error_handler "detectron2 git clone failed"
) else (
  echo detectron2 already cloned, skipping.
)

pip install ninja
if errorlevel 1 echo [!] ninja install failed. Build may be slower.

:: Toolchain precheck (must be available for detectron2 C++ extension build)
where cl >nul 2>nul
if errorlevel 1 call :error_handler "MSVC compiler (cl.exe) not found. Open 'x64 Native Tools Command Prompt for VS' or install VS Build Tools C++ workload + Windows SDK."
where rc >nul 2>nul
if errorlevel 1 call :error_handler "Windows SDK resource compiler (rc.exe) not found. Install Windows 10/11 SDK in VS Build Tools."
where mt >nul 2>nul
if errorlevel 1 call :error_handler "Windows SDK manifest tool (mt.exe) not found. Install Windows 10/11 SDK in VS Build Tools."

:: Re-pin build-sensitive deps right before detectron2 build
pip install --force-reinstall "numpy==1.26.4" "setuptools==69.5.1"
if errorlevel 1 call :error_handler "Failed to pin numpy/setuptools before detectron2 build"

:: --no-build-isolation lets the build process see the already-installed torch + setuptools
pip install --no-build-isolation -e external\detectron2
if errorlevel 1 call :error_handler "detectron2 install failed. Ensure Visual Studio Build Tools (C++ workload) are installed."

echo [7/9] Install LLM + pipeline extras
pip install "transformers>=4.43.0" accelerate bitsandbytes sentencepiece safetensors scipy scikit-image opencv-python pillow requests tqdm
if errorlevel 1 call :error_handler "Extras installation failed"

echo [8/9] Download checkpoints (optional)
if "%SKIP_CHECKPOINTS%"=="0" (
  if exist "scripts\download_checkpoints.sh" (
    where bash >nul 2>nul
    if errorlevel 1 (
      echo bash not found. Skipping checkpoint download script.
    ) else (
      bash scripts/download_checkpoints.sh
      if errorlevel 1 echo [!] Checkpoint script error. Continuing...
    )
  )
)

echo [9/9] Smoke test
python -c "import torch, detectron2; print('torch:', torch.__version__); print('detectron2:', detectron2.__version__); print('CUDA available:', torch.cuda.is_available())"
if errorlevel 1 call :error_handler "Smoke test failed"

echo.
echo ============================================================
echo  Setup complete! Activate your venv with:
echo    call %REPO_DIR%\%VENV_DIR%\Scripts\activate.bat
echo ============================================================
pause
endlocal
exit /b 0

:error_handler
echo.
echo ---------------------------------------------------------
echo [ERROR] %~1
echo The script will pause. Press any key to try continuing...
echo ---------------------------------------------------------
pause
goto :EOF