#!/bin/bash
# Script to run production training on multi-GPU setup

echo "========================================="
echo "Nemotron CPT - Production Training"
echo "========================================="

# Set number of GPUs
NUM_GPUS=${1:-4}  # Default to 4 GPUs, override with first argument

echo "Training with $NUM_GPUS GPUs"

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null
then
    echo "ERROR: nvidia-smi not found. Please ensure CUDA is installed."
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run training with DeepSpeed
echo ""
echo "Starting production training with DeepSpeed..."

deepspeed --num_gpus=$NUM_GPUS \
    src/training/train_cpt.py \
    --config configs/production_multi_gpu.yaml

echo ""
echo "Training complete!"
