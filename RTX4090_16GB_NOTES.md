# RTX 4090 Laptop (16GB) - Important Notes

Your RTX 4090 **Laptop** has **16GB VRAM** (not 24GB like the desktop variant). This requires specific optimizations.

## Auto-Detection

The training script (`scripts\run_prototype.bat`) now **automatically detects** your GPU memory and uses the correct config:
- **16GB detected** → Uses `configs/prototype_rtx4090_16gb.yaml`
- **24GB detected** → Uses `configs/prototype_rtx4090.yaml`

## Key Differences for 16GB Configuration

### Memory Optimizations

| Setting | 24GB Desktop | 16GB Laptop | Why |
|---------|--------------|-------------|-----|
| Sequence Length | 4096 | 2048 | 50% memory reduction |
| Batch Size | 1 | 1 | Same |
| Grad Accumulation | 8 | 16 | Maintain effective batch size |
| DeepSpeed | Optional | **Enabled** | CPU offload essential |
| Gradient Checkpointing | Yes | Yes | Both need it |

### Expected Performance

**16GB Configuration:**
- **Memory usage**: ~14-15GB / 16GB
- **Training speed**: ~1.5-2 tokens/second (slightly slower due to shorter sequences)
- **Sequence length**: 2048 tokens (vs 4096 on desktop)
- **Effective batch size**: 16 (same as desktop config)

### Running Training

**Automatic (Recommended):**
```powershell
.\scripts\run_prototype.bat
# Will auto-detect 16GB and use correct config
```

**Manual:**
```powershell
# Explicitly use 16GB config
python src\training\train_cpt.py --config configs\prototype_rtx4090_16gb.yaml
```

## If You Get OOM (Out of Memory) Errors

If you still run out of memory, further reduce settings:

### Option 1: Reduce Sequence Length
```yaml
data:
  max_length: 1024  # Even shorter sequences
```

### Option 2: Increase Gradient Accumulation
```yaml
training:
  gradient_accumulation_steps: 32  # More accumulation
```

### Option 3: Use Smaller Model
```yaml
model:
  name: "microsoft/phi-2"  # 2.7B model (much smaller)
  # or
  name: "stabilityai/stablelm-2-1_6b"  # 1.6B model
```

### Option 4: Enable More Aggressive DeepSpeed Offload

Edit `configs/ds_config_prototype.json`:
```json
{
  "zero_optimization": {
    "stage": 3,  // Change from 2 to 3
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",  // Change from "cpu" to enable param offload
      "pin_memory": true
    }
  }
}
```

## Monitoring 16GB VRAM

```powershell
# Watch memory usage in real-time
nvidia-smi -l 1

# Or use the monitoring script
python scripts\monitor_training.py --mode gpu
```

**Safe memory levels:**
- **Idle**: 0.3GB (normal)
- **Model loading**: 8-10GB
- **Training**: 14-15GB
- **Peak**: Should not exceed 15.5GB

## Tips for 16GB

### 1. Close Background Apps
Your `nvidia-smi` shows Brave browser using GPU memory:
```powershell
# Close browsers before training
# Or restart to clear GPU memory
```

### 2. Clear GPU Memory
```powershell
# Kill all GPU processes
taskkill /F /IM brave.exe
# Then check
nvidia-smi
```

### 3. Start Small
```powershell
# Create small test dataset first
python scripts\prepare_data.py --create-sample

# Run for just 100 steps to test
# Edit config: max_steps: 100
```

### 4. Monitor Temperature
Laptop GPUs can thermal throttle:
```powershell
# Monitor temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1
```

Keep below 80°C for sustained performance.

## Memory Breakdown (16GB Config)

Estimated usage with `prototype_rtx4090_16gb.yaml`:

```
Model weights (BF16):        ~16GB → 8GB (with ZeRO-2)
Optimizer states:            ~6GB → 3GB (CPU offload)
Gradients:                   ~2GB → 1GB (with ZeRO-2)
Activations (2048 seq):      ~2GB → 1GB (grad checkpoint)
Buffer/overhead:             ~1GB
─────────────────────────────────────────────
Total VRAM usage:            ~14-15GB / 16GB ✓
```

## Comparison: Desktop vs Laptop

| Aspect | Desktop (24GB) | Laptop (16GB) |
|--------|----------------|---------------|
| Max Sequence | 4096 | 2048 |
| Speed | 2-3 tok/s | 1.5-2 tok/s |
| DeepSpeed | Optional | Required |
| Max Batch | 1 | 1 |
| Thermal | Better | Watch temps |
| Cost | Higher | Lower |

## Upgrading Options

If you need more memory:
1. **Reduce sequence length** to 1024 (fastest)
2. **Use smaller model** like Phi-2 (2.7B)
3. **Use LoRA/QLoRA** for parameter-efficient training
4. **Cloud GPUs** for larger experiments (A100, H100)

## Current GPU Status

Your current GPU state:
```
GPU: RTX 4090 Laptop
VRAM: 16,376 MB (16GB)
Used: 361 MB (Brave browsers)
Available: ~16GB
Temperature: 55°C (good)
```

**Recommendation**: Close Brave before training to free up VRAM.

## Quick Start (16GB Optimized)

```powershell
# 1. Close browsers
taskkill /F /IM brave.exe

# 2. Check GPU is clear
nvidia-smi

# 3. Run setup
.\scripts\setup_environment.bat

# 4. Login to HuggingFace
venv\Scripts\activate
huggingface-cli login

# 5. Create sample data
python scripts\prepare_data.py --create-sample

# 6. Start training (auto-detects 16GB)
.\scripts\run_prototype.bat

# 7. Monitor in new window
python scripts\monitor_training.py --mode gpu
```

Your pipeline is now optimized for 16GB! 🚀
