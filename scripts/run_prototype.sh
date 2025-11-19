#!/bin/bash
# Script to run prototype training on RTX 4090

echo "========================================="
echo "Nemotron CPT - Prototype Training"
echo "========================================="

# Set environment variables for optimization
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

# Run training
echo ""
echo "Starting prototype training..."
python src/training/train_cpt.py \
    --config configs/prototype_rtx4090.yaml

echo ""
echo "Training complete!"
