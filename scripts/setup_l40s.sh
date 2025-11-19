#!/bin/bash
# Quick Setup Script for L40S with Nemotron-Nano-9B-v2
# Python 3.10.12, CUDA 12.7, NVIDIA L40S 46GB

set -e  # Exit on error

echo "=========================================="
echo "L40S Nemotron-Nano-9B-v2 Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check CUDA
echo "Checking CUDA availability..."
nvidia-smi
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "env_cpt" ]; then
    echo "Creating virtual environment..."
    python -m venv env_cpt
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env_cpt/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.7)
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Try to install flash-attn
echo "Installing Flash Attention..."
echo "Attempting method 1: Direct installation..."
if pip install flash-attn --no-build-isolation --no-cache-dir; then
    echo "Flash Attention installed successfully!"
else
    echo "Method 1 failed. Trying method 2: Download wheel..."
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    if pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl; then
        echo "Flash Attention installed successfully!"
        rm flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    else
        echo "Warning: Flash Attention installation failed. Training will work but slower."
    fi
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Try to import flash_attn
python -c "try:
    import flash_attn
    print('Flash Attention: Installed ✓')
except ImportError:
    print('Flash Attention: Not installed (training will work but slower)')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Login to HuggingFace: huggingface-cli login"
echo "2. Prepare data: python scripts/prepare_data.py --create-sample"
echo "3. Verify setup: python scripts/prepare_data.py --verify configs/l40s_nemotron_nano_9b.yaml"
echo "4. Start training: python src/training/train_cpt.py --config configs/l40s_nemotron_nano_9b.yaml"
echo ""
echo "For optimal performance, make sure Flash Attention is installed!"
