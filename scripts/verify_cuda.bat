@echo off
REM Script to verify CUDA and PyTorch installation

echo =========================================
echo CUDA and PyTorch Verification
echo =========================================
echo.

REM Check NVIDIA driver
echo [1/4] Checking NVIDIA Driver...
nvidia-smi --query-gpu=driver_version,cuda_version,name,memory.total --format=csv
echo.

REM Check if Python is available
echo [2/4] Checking Python...
python --version
echo.

REM Check if PyTorch is installed
echo [3/4] Checking PyTorch installation...
python -c "import torch; print(f'PyTorch Version: {torch.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyTorch is not installed!
    echo Run: .\scripts\install_windows.bat
    echo.
    pause
    exit /b 1
)
echo.

REM Check CUDA availability in PyTorch
echo [4/4] Checking CUDA availability in PyTorch...
python -c "import torch; cuda_available = torch.cuda.is_available(); print(f'CUDA Available: {cuda_available}'); print(f'CUDA Version (PyTorch): {torch.version.cuda if cuda_available else \"N/A\"}'); print(f'GPU Count: {torch.cuda.device_count() if cuda_available else 0}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if cuda_available else \"N/A\"}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if cuda_available else \"N/A\")"
echo.

REM Check if CUDA is available
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =========================================
    echo WARNING: CUDA is NOT available in PyTorch!
    echo =========================================
    echo.
    echo This means PyTorch is using CPU-only version.
    echo You need to reinstall PyTorch with CUDA support:
    echo.
    echo   pip uninstall torch torchvision torchaudio
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo Or run: .\scripts\install_windows.bat
    echo.
    pause
    exit /b 1
)

echo =========================================
echo ✓ All checks passed!
echo =========================================
echo.
echo Your system is ready for GPU training:
echo - NVIDIA Driver: Installed
echo - CUDA: Available
echo - PyTorch: Installed with CUDA support
echo - GPU: Detected and ready
echo.
echo Next steps:
echo 1. Login to HuggingFace: huggingface-cli login
echo 2. Prepare data: python scripts\prepare_data.py --create-sample
echo 3. Start training: .\scripts\run_prototype.bat
echo.
pause
