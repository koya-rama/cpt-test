# Windows Setup Guide - Nemotron CPT on RTX 4090

Complete guide for running Nemotron Continuous Pre-Training on Windows with RTX 4090.

## Prerequisites

### Required Software

1. **Python 3.10 or 3.11**
   - Download from: https://www.python.org/downloads/
   - ✓ During installation, check "Add Python to PATH"

2. **CUDA Toolkit 11.8 or 12.1**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → 11 → exe (local)

3. **Visual Studio Build Tools** (for Flash Attention compilation)
   - Download: https://visualstudio.microsoft.com/downloads/
   - Install "Desktop development with C++"

4. **Git for Windows** (optional, for version control)
   - Download from: https://git-scm.com/download/win

### Verify Installation

Open PowerShell or Command Prompt:

```powershell
# Check Python
python --version
# Should show: Python 3.10.x or 3.11.x

# Check CUDA
nvidia-smi
# Should show your RTX 4090 with driver info

# Check pip
pip --version
```

## Installation Steps

### Step 1: Navigate to Project Directory

```powershell
cd C:\Makerslab\Projects\IndiaAI\Indus\CPT
```

### Step 2: Run Automated Setup

**Option A: Using the setup script (Recommended)**

```powershell
.\scripts\setup_environment.bat
```

This will automatically:
- Create virtual environment
- Install all dependencies
- Create sample data
- Prompt for Flash Attention installation

**Option B: Manual setup**

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention (requires VS Build Tools)
pip install flash-attn --no-build-isolation
```

### Step 3: Login to HuggingFace

```powershell
# Activate environment if not already active
venv\Scripts\activate

# Login to HuggingFace
huggingface-cli login
```

You'll need a HuggingFace token from: https://huggingface.co/settings/tokens

### Step 4: Prepare Data

**Create sample data for testing:**

```powershell
python scripts\prepare_data.py --create-sample
```

**Or add your own data:**

```powershell
# Copy your data to the custom corpus folder
copy your_data.txt data\custom_corpus\
copy your_data.jsonl data\custom_corpus\
```

**Verify data loading:**

```powershell
python scripts\prepare_data.py --verify configs\prototype_rtx4090.yaml
```

## Running Training

### Prototype Training (RTX 4090)

**Start training:**

```powershell
# Make sure environment is activated
venv\Scripts\activate

# Run training
.\scripts\run_prototype.bat
```

**Or run directly:**

```powershell
python src\training\train_cpt.py --config configs\prototype_rtx4090.yaml
```

### Monitor Training

**Open a NEW Command Prompt or PowerShell window:**

```powershell
# Navigate to project
cd C:\Makerslab\Projects\IndiaAI\Indus\CPT

# Activate environment
venv\Scripts\activate

# Monitor GPU
python scripts\monitor_training.py --mode gpu
```

**Or use nvidia-smi:**

```powershell
# Real-time monitoring (updates every 1 second)
nvidia-smi -l 1
```

**Monitor checkpoints:**

```powershell
python scripts\monitor_training.py --mode checkpoint --checkpoint-dir checkpoints\prototype
```

## Configuration for Windows

The default config `configs\prototype_rtx4090.yaml` is already optimized for Windows + RTX 4090:

```yaml
model:
  name: "nvidia/nemotron-3-8b-base-4k"
  use_flash_attention: true  # Set to false if Flash Attention install fails

data:
  custom_corpus_paths:
    - "./data/custom_corpus"  # Works on Windows
  max_length: 4096
  streaming: true

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  bf16: true
  gradient_checkpointing: true
  output_dir: "./checkpoints/prototype"  # Windows path
```

## Multi-GPU Training on Windows

If you have multiple GPUs:

```powershell
# For 2 GPUs
.\scripts\run_production_multi_gpu.bat 2

# For 4 GPUs
.\scripts\run_production_multi_gpu.bat 4
```

## Common Windows Issues & Solutions

### Issue 1: PowerShell Execution Policy

**Error:** "running scripts is disabled on this system"

**Solution:**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 2: Long Path Support

**Error:** Path too long errors

**Solution:**
```powershell
# Enable long paths (Run as Administrator)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Or use shorter paths for checkpoints:
```yaml
training:
  output_dir: "C:\CPT_checkpoints"  # Shorter path
```

### Issue 3: Flash Attention Installation Fails

**Error:** Compilation errors

**Solution 1:** Install Visual Studio Build Tools
```powershell
# Download and install from:
# https://visualstudio.microsoft.com/downloads/
# Select "Desktop development with C++"
```

**Solution 2:** Disable Flash Attention
```yaml
# In your config file
model:
  use_flash_attention: false
```

### Issue 4: Out of Memory on RTX 4090

**Solution:** Edit `configs\prototype_rtx4090.yaml`:

```yaml
data:
  max_length: 2048  # Reduce from 4096

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # Increase from 8
```

### Issue 5: DeepSpeed Installation Issues

**Error:** DeepSpeed not installing properly

**Solution:**
```powershell
# Install pre-built wheel
pip install deepspeed --no-build-isolation

# Or disable DeepSpeed in config
# Comment out:
# deepspeed_config: "./configs/ds_config_prototype.json"
```

### Issue 6: Slow Data Loading

**Solution:**
```yaml
# In your config file
data:
  streaming: true  # Enable streaming
  num_proc: 2  # Reduce workers on Windows
```

## Windows Performance Tips

### 1. Disable Windows Defender Real-Time Scanning

Temporarily for the data folder:
```powershell
# Add exclusion (Run as Administrator)
Add-MpPreference -ExclusionPath "C:\Makerslab\Projects\IndiaAI\Indus\CPT\data"
Add-MpPreference -ExclusionPath "C:\Makerslab\Projects\IndiaAI\Indus\CPT\checkpoints"
```

### 2. Set High Performance Power Plan

```powershell
# Set to High Performance
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

### 3. Close Background Applications

- Close browsers, Discord, etc.
- Keep Task Manager open to monitor

### 4. Monitor Temperatures

```powershell
# Use nvidia-smi to check GPU temp
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1
```

## File Paths on Windows

Always use forward slashes `/` or double backslashes `\\` in config files:

**✓ Correct:**
```yaml
output_dir: "./checkpoints/prototype"
output_dir: ".\\checkpoints\\prototype"
output_dir: "C:/CPT/checkpoints"
```

**✗ Incorrect:**
```yaml
output_dir: ".\checkpoints\prototype"  # Single backslash can cause issues
```

## Expected Performance on Windows RTX 4090

- **First run:** Model download ~15GB (one time)
- **First batch:** 2-3 minutes (compilation)
- **Training speed:** 2-3 tokens/second
- **GPU usage:** 20-22GB / 24GB
- **Checkpoint save:** ~30 seconds per checkpoint

## Stopping Training

**Graceful shutdown:**
- Press `Ctrl+C` in the training window
- Wait for current checkpoint to save
- Training can be resumed later

**Resume training:**
```yaml
# In your config file, add:
training:
  resume_from_checkpoint: "./checkpoints/prototype/checkpoint-1000"
```

## Directory Structure on Windows

```
C:\Makerslab\Projects\IndiaAI\Indus\CPT\
├── venv\                           # Virtual environment
├── data\
│   ├── custom_corpus\              # Your data here
│   └── cache\                      # HuggingFace cache
├── checkpoints\
│   └── prototype\                  # Saved checkpoints
├── logs\                           # Training logs
├── configs\
│   └── prototype_rtx4090.yaml     # Your config
└── scripts\
    ├── setup_environment.bat       # Setup script
    ├── run_prototype.bat           # Training script
    └── monitor_training.py         # Monitoring script
```

## Useful Windows Commands

```powershell
# Check GPU usage
nvidia-smi

# Check disk space
dir "checkpoints" | measure-object -property length -sum

# Find Python processes
tasklist | findstr python

# Kill Python if frozen
taskkill /F /IM python.exe

# Check port usage (for TensorBoard)
netstat -ano | findstr :6006
```

## Using TensorBoard on Windows

```powershell
# Activate environment
venv\Scripts\activate

# Start TensorBoard
tensorboard --logdir checkpoints\prototype

# Open browser to: http://localhost:6006
```

## Weights & Biases on Windows

```powershell
# Login
wandb login

# Your API key from: https://wandb.ai/authorize
```

Enable in config:
```yaml
training:
  use_wandb: true
  wandb_project: "nemotron-cpt"
```

## Next Steps

1. ✓ Run `.\scripts\setup_environment.bat`
2. ✓ Login: `huggingface-cli login`
3. ✓ Add data to `data\custom_corpus\`
4. ✓ Test: `.\scripts\run_prototype.bat`
5. ✓ Monitor: `python scripts\monitor_training.py --mode gpu`

## Support

For Windows-specific issues:
1. Check Windows Event Viewer for system errors
2. Verify CUDA installation: `nvidia-smi`
3. Check Python path: `where python`
4. Review logs in the terminal

Happy training on Windows! 🚀
