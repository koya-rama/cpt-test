#!/bin/bash
# Installation script for CPT framework dependencies
# Python 3.10.12 + CUDA 12.7 + NVIDIA L40S

set -e  # Exit on error

echo "=========================================="
echo "CPT Framework Dependency Installation"
echo "Python 3.10.12 + CUDA 12.7 + NVIDIA L40S"
echo "=========================================="
echo ""

# Step 1: Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.7)
echo "Step 1/3: Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install other dependencies from requirements.txt
echo ""
echo "Step 2/3: Installing other dependencies..."
pip install -r requirements.txt

# Step 3: Install flash-attention (requires torch to be installed first)
echo ""
echo "Step 3/3: Installing flash-attention (this may take 5-10 minutes)..."
pip install flash-attn --no-build-isolation

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}');"
