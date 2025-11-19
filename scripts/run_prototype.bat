@echo off
REM Script to run prototype training on RTX 4090 (Windows)

echo =========================================
echo Nemotron CPT - Prototype Training
echo =========================================

REM Set environment variables for optimization
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set TOKENIZERS_PARALLELISM=false

REM Check if CUDA is available
where nvidia-smi >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: nvidia-smi not found. Please ensure CUDA is installed.
    exit /b 1
)

echo GPU Information:
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

REM Detect GPU VRAM and select appropriate config
echo.
echo Detecting GPU configuration...
for /f "tokens=*" %%a in ('nvidia-smi --query-gpu^=memory.total --format^=csv,noheader,nounits') do set GPU_MEM=%%a

set CONFIG_FILE=configs\prototype_rtx4090.yaml

REM Check if GPU has less than 20GB (likely 16GB laptop variant)
if %GPU_MEM% LSS 20000 (
    echo Detected RTX 4090 Laptop with 16GB VRAM
    set CONFIG_FILE=configs\prototype_rtx4090_16gb.yaml
) else (
    echo Detected RTX 4090 Desktop with 24GB VRAM
    set CONFIG_FILE=configs\prototype_rtx4090.yaml
)

REM Run training
echo.
echo Starting prototype training with %CONFIG_FILE%...
python src\training\train_cpt.py --config %CONFIG_FILE%

echo.
echo Training complete!
pause
