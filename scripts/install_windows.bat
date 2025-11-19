@echo off
REM Windows installation script with proper dependency order

echo =========================================
echo Installing Nemotron CPT Dependencies
echo =========================================
echo.

REM Upgrade pip first
echo [1/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch with CUDA support (required by other packages)
echo [2/5] Installing PyTorch with CUDA 12.1...
echo This may take a few minutes to download ~2GB...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

REM Verify PyTorch CUDA installation
echo Verifying PyTorch CUDA installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

REM Install core dependencies
echo [3/5] Installing core dependencies...
pip install transformers>=4.46.0 datasets>=2.16.0 accelerate>=0.25.0 tokenizers>=0.15.0
echo.

REM Install other dependencies
echo [4/5] Installing additional packages...
pip install sentencepiece pyarrow pandas numpy wandb tensorboard tqdm pyyaml scipy scikit-learn gputil psutil peft bitsandbytes
echo.

REM Optional: DeepSpeed (Windows support limited)
echo [5/5] DeepSpeed installation (optional)...
set /p install_deepspeed="Install DeepSpeed? (May fail on Windows, y/n): "
if /i "%install_deepspeed%"=="y" (
    echo Attempting to install DeepSpeed...
    pip install deepspeed --no-build-isolation
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo WARNING: DeepSpeed installation failed.
        echo This is common on Windows. Training will work without it,
        echo but with slightly less memory efficiency.
        echo.
    ) else (
        echo DeepSpeed installed successfully!
    )
) else (
    echo Skipping DeepSpeed installation
)

echo.
echo =========================================
echo Installation Complete!
echo =========================================
echo.
echo Optional: Install Flash Attention for 2x speedup
echo.
set /p install_flash="Install Flash Attention? (May require long paths enabled, y/n): "
if /i "%install_flash%"=="y" (
    echo.
    echo Attempting to install Flash Attention...
    echo This may fail due to Windows long path limitations.
    echo.
    pip install flash-attn --no-build-isolation
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo WARNING: Flash Attention installation failed.
        echo This is common on Windows due to long file path limitations.
        echo.
        echo Training will work without Flash Attention, just ~30%% slower.
        echo.
        echo To enable Flash Attention later:
        echo 1. Run PowerShell as Administrator
        echo 2. Run: .\scripts\enable_long_paths.ps1
        echo 3. Restart computer
        echo 4. Try: pip install flash-attn --no-build-isolation
        echo.
        echo Or simply train without it - it's optional!
        echo.
    ) else (
        echo.
        echo ✓ Flash Attention installed successfully!
        echo You can now use use_flash_attention: true in configs
        echo.
    )
) else (
    echo Skipping Flash Attention installation
    echo Training will work fine without it!
)
echo.
pause
