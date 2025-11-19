# CUDA-Enabled PyTorch Installation Guide

Your system has **CUDA 13.0** (Driver 581.29). We'll install PyTorch with **CUDA 12.1** support, which is fully compatible.

## Quick Install (Recommended)

### Method 1: Use the Install Script

```powershell
.\scripts\install_windows.bat
```

This automatically installs PyTorch with CUDA 12.1 support.

### Method 2: Manual Install

```powershell
# Step 1: Install PyTorch with CUDA 12.1 (Compatible with your CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2: Verify CUDA is working
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Step 3: Install other dependencies
pip install transformers datasets accelerate tokenizers sentencepiece pyarrow pandas numpy wandb tensorboard tqdm pyyaml scipy scikit-learn gputil psutil peft bitsandbytes

# Step 4: (Optional) Try DeepSpeed
pip install deepspeed --no-build-isolation
```

## Verify CUDA Installation

```powershell
# Check PyTorch can see your GPU
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True

# Check GPU details
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should print: NVIDIA GeForce RTX 4090 Laptop GPU

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"
# Should print: 12.1

# Full verification
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
PyTorch: 2.9.1+cu121
CUDA Available: True
CUDA Version: 12.1
GPU Count: 1
GPU: NVIDIA GeForce RTX 4090 Laptop GPU
```

## CUDA Version Compatibility

| Your System | PyTorch Version | Compatible |
|-------------|-----------------|------------|
| CUDA 13.0 (Driver 581.29) | PyTorch with CUDA 12.1 | ✅ Yes |
| CUDA 13.0 (Driver 581.29) | PyTorch with CUDA 11.8 | ✅ Yes |
| CUDA 13.0 (Driver 581.29) | PyTorch CPU-only | ❌ No (won't use GPU) |

**Note:** CUDA drivers are backward compatible. Your CUDA 13.0 driver supports PyTorch built for CUDA 12.1 or 11.8.

## Alternative: CUDA 11.8

If you prefer CUDA 11.8:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

### Issue: "CUDA Available: False"

**Check 1: Verify PyTorch CUDA version**
```powershell
pip show torch
# Look for: torch 2.9.1+cu121 (should have +cu121, not +cpu)
```

**Fix: Reinstall with CUDA**
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "RuntimeError: CUDA out of memory"

This is normal during training. Solutions:
1. Close other GPU applications (browsers, etc.)
2. Reduce batch size in config
3. Reduce sequence length in config

### Issue: Wrong PyTorch version installed

**Check current version:**
```powershell
python -c "import torch; print(torch.__version__)"
```

If it shows `2.9.1+cpu` instead of `2.9.1+cu121`:

**Fix:**
```powershell
# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Flash Attention with CUDA

Flash Attention requires compilation against CUDA:

```powershell
# Requires Visual Studio Build Tools
pip install flash-attn --no-build-isolation
```

If compilation fails:
1. Install Visual Studio Build Tools 2019 or later
2. Or disable Flash Attention in config: `use_flash_attention: false`

## Performance Test

Test GPU performance:

```powershell
python -c "import torch; x = torch.randn(1000, 1000).cuda(); y = torch.randn(1000, 1000).cuda(); z = torch.matmul(x, y); print('GPU computation successful!')"
```

Should print: `GPU computation successful!`

## PyTorch Installation Options

### CUDA 12.1 (Recommended for your system)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8 (Alternative)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CPU-only (NOT recommended - won't use GPU)
```powershell
pip install torch torchvision torchaudio
# Don't use this - your training will be 100x slower
```

## Next Steps After Installation

1. **Verify CUDA works:**
   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Login to HuggingFace:**
   ```powershell
   huggingface-cli login
   ```

3. **Create sample data:**
   ```powershell
   python scripts\prepare_data.py --create-sample
   ```

4. **Run training:**
   ```powershell
   .\scripts\run_prototype.bat
   ```

## System Requirements Met ✅

- ✅ Python 3.12
- ✅ NVIDIA RTX 4090 Laptop (16GB)
- ✅ CUDA 13.0 (Driver 581.29)
- ✅ Windows 11

You're all set for GPU-accelerated training! 🚀
