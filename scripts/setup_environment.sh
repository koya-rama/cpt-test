#!/bin/bash
# Automated setup script for Nemotron CPT Pipeline

set -e  # Exit on error

echo "========================================="
echo "Nemotron CPT Pipeline - Environment Setup"
echo "========================================="
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "ERROR: Python 3.10+ is required"
    exit 1
fi

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "[4/6] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "[5/6] Installing dependencies..."
pip install -r requirements.txt

# Install Flash Attention (optional)
echo ""
echo "[6/6] Installing Flash Attention (optional)..."
read -p "Install Flash Attention 2? (requires compilation, y/n): " install_flash
if [ "$install_flash" = "y" ] || [ "$install_flash" = "Y" ]; then
    pip install flash-attn --no-build-isolation
    echo "Flash Attention installed"
else
    echo "Skipping Flash Attention installation"
fi

# Create sample data
echo ""
echo "Creating sample data..."
python scripts/prepare_data.py --create-sample

# HuggingFace login
echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Login to HuggingFace: huggingface-cli login"
echo "3. Add your data to: data/custom_corpus/"
echo "4. Run training: ./scripts/run_prototype.sh"
echo ""
echo "For more information, see README.md and QUICKSTART.md"
echo ""
