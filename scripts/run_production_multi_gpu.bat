@echo off
REM Script to run production training on multi-GPU setup (Windows)

echo =========================================
echo Nemotron CPT - Production Training
echo =========================================

REM Set number of GPUs (default to 4)
set NUM_GPUS=%1
if "%NUM_GPUS%"=="" set NUM_GPUS=4

echo Training with %NUM_GPUS% GPUs

REM Set environment variables
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

REM Run training with DeepSpeed
echo.
echo Starting production training with DeepSpeed...

deepspeed --num_gpus=%NUM_GPUS% src\training\train_cpt.py --config configs\production_multi_gpu.yaml

echo.
echo Training complete!
pause
