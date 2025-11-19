# Quick Start Guide - Nemotron CPT on RTX 4090

This guide will get you started with continuous pre-training on your RTX 4090 laptop in under 10 minutes.

## Model Selection

This pipeline supports multiple models. Choose based on your hardware:

**For NVIDIA L40S 46GB:** ⭐ **RECOMMENDED**
- **nvidia/NVIDIA-Nemotron-Nano-9B-v2** (Latest & Best) - 9B model with advanced architecture
  - Config: `configs/l40s_nemotron_nano_9b.yaml`
  - Memory: ~35-40GB usage with 4096 context
  - Performance: ~5-8 tokens/sec with Flash Attention
- **nvidia/Llama-3.1-Nemotron-Nano-8B-v1** - 8B model with 128K context support
  - Config: `configs/prototype_rtx4090_16gb_nemotron_nano.yaml` (can use larger batch sizes)

**For RTX 4090 16GB:**
- **nvidia/Llama-3.1-Nemotron-Nano-8B-v1** (Recommended) - 8B model with 128K context support
  - Config: `configs/prototype_rtx4090_16gb_nemotron_nano.yaml`
- **meta-llama/Llama-3.2-3B** - Smaller, faster for testing
  - Config: `configs/prototype_rtx4090_16gb_llama.yaml`

**For RTX 4090 24GB:**
- **nvidia/nemotron-3-8b-base-4k** - Full Nemotron model
  - Config: `configs/prototype_rtx4090.yaml`

## Step 1: Install Dependencies (5 minutes)

### System Requirements
- **Python**: 3.10.12+ (tested on 3.10.12)
- **CUDA**: 12.1+ (tested with 12.7 on NVIDIA L40S)
- **GPU**: RTX 4090 (16GB/24GB), RTX 3090, L40S, or equivalent

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Ubuntu)
source venv/bin/activate

# Step 1: Install PyTorch with CUDA 12.1 (compatible with CUDA 12.7)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install other requirements
pip install -r requirements.txt

# Step 3: Optional - Install Flash Attention for 2x speed boost
# Method 1: Try direct installation (may fail due to filesystem issues)
pip install flash-attn --no-build-isolation --no-cache-dir

# Method 2: If above fails, download and install wheel directly
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Note: Flash Attention is optional. Training will work without it, just slower.
```

## Step 2: Login to HuggingFace (1 minute)

```bash
# Get your token from https://huggingface.co/settings/tokens
huggingface-cli login
```

## Step 3: Prepare Your Data (2 minutes)

**Option A: Use Sample Data (Recommended for testing)**
```bash
python scripts/prepare_data.py --create-sample
```

**Option B: Add Your Own Data**
```bash
# Place your text files in data/custom_corpus/
cp your_text_data.txt data/custom_corpus/

# Or JSONL format
echo '{"text": "Your training text here..."}' > data/custom_corpus/data.jsonl
```

**Option C: Use NVIDIA Nemotron Datasets (For L40S)**
Edit `configs/l40s_nemotron_nano_9b.yaml` and uncomment:
```yaml
data:
  hf_datasets:
    - "nvidia/Nemotron-CC"  # 6.3T high-quality tokens
```

## Step 4: Verify Setup (1 minute)

**For L40S with Nemotron-Nano-9B-v2 (46GB VRAM):** ⭐
```bash
python scripts/prepare_data.py --verify configs/l40s_nemotron_nano_9b.yaml
```

**For Nemotron Nano (16GB VRAM):**
```bash
python scripts/prepare_data.py --verify configs/prototype_rtx4090_16gb_nemotron_nano.yaml
```

**For Llama 3.2 3B (16GB VRAM):**
```bash
python scripts/prepare_data.py --verify configs/prototype_rtx4090_16gb_llama.yaml
```

**For standard RTX 4090 (24GB VRAM):**
```bash
python scripts/prepare_data.py --verify configs/prototype_rtx4090.yaml
```

## Step 5: Start Training (1 minute to start)

**Using Nemotron-Nano-9B-v2 on L40S (46GB - BEST):** ⭐
```bash
# Linux/Ubuntu (L40S is typically server-side)
python src/training/train_cpt.py --config configs/l40s_nemotron_nano_9b.yaml

# With DeepSpeed for further optimization
python src/training/train_cpt.py --config configs/l40s_nemotron_nano_9b.yaml --deepspeed configs/ds_config_l40s.json
```

**Using Nemotron Nano 8B (Recommended for 16GB):**
```bash
# Windows
python src/training/train_cpt.py --config configs/prototype_rtx4090_16gb_nemotron_nano.yaml

# Linux/Mac
python src/training/train_cpt.py --config configs/prototype_rtx4090_16gb_nemotron_nano.yaml
```

**Using Llama 3.2 3B (Fastest for 16GB):**
```bash
python src/training/train_cpt.py --config configs/prototype_rtx4090_16gb_llama.yaml
```

**Using Default Script (24GB VRAM):**
```bash
# Windows
scripts\run_prototype.bat

# Linux/Mac
chmod +x scripts/run_prototype.sh
./scripts/run_prototype.sh
```

## Step 6: Monitor Training

Open a new terminal:
```bash
# Activate environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Monitor GPU
python scripts/monitor_training.py --mode gpu
```

## What to Expect

### First Run - L40S with 9B Model
- Model download: ~18GB (one time, cached)
- Initial compilation: 3-5 minutes (first batch)
- Training speed: ~5-8 tokens/sec with Flash Attention
- Memory usage: ~35-40GB / 46GB

### First Run - RTX 4090 with 8B Model
- Model download: ~15GB (one time, cached)
- Initial compilation: 2-3 minutes (first batch)
- Training speed: ~2-3 tokens/sec
- Memory usage: ~20-22GB / 24GB

### Memory Usage
- **L40S**: Expected 35-40GB / 46GB (plenty of headroom)
- **RTX 4090 24GB**: Expected 20-22GB / 24GB
- **RTX 4090 16GB**: Expected 14-15GB / 16GB
- If OOM occurs: Reduce batch size or max_length in config

### Checkpoints
- L40S: Saved every 500 steps to `checkpoints/l40s_nemotron_9b/`
- Other configs: Saved to respective checkpoint directories
- Each checkpoint: ~15-18GB for 8-9B models
- Keep last 2-3 checkpoints (auto-cleanup)

## Configuration Tips

### L40S with Nemotron-Nano-9B-v2 ⭐

**Advantages:**
- 46GB VRAM allows large batch sizes and long context
- Can train with max_length up to 8192 tokens
- Supports batch_size=2-4 with gradient accumulation
- Flash Attention provides 2-3x speedup

**Memory Profile:**
- Expected GPU usage: ~35-40GB / 46GB
- Sequence length: 4096 (default), can increase to 8192
- Batch size: 2 with gradient accumulation of 8

**To use longer context:**
```yaml
data:
  max_length: 8192  # Or even higher if needed

training:
  per_device_train_batch_size: 1  # Reduce if using very long context
  gradient_accumulation_steps: 16
```

**To increase throughput:**
```yaml
training:
  per_device_train_batch_size: 4  # Increase for faster training
  gradient_accumulation_steps: 4  # Reduce accordingly
```

### RTX 4090 - Nemotron Nano 8B (nvidia/Llama-3.1-Nemotron-Nano-8B-v1)

**Advantages:**
- 128K context length support (can be extended beyond 2048)
- Optimized architecture for efficiency
- Better performance on long-context tasks

**Memory Profile:**
- Expected GPU usage: ~14-15GB / 16GB
- Sequence length: 2048 (default), can increase to 4096 or 8192
- Batch size: 1 with gradient accumulation of 16

**To use longer context:**
```yaml
data:
  max_length: 4096  # Or 8192 for very long context
```

### Standard Configurations

The default config is optimized for RTX 4090, but you can adjust:

### For Faster Training (if you have memory headroom)
```yaml
training:
  per_device_train_batch_size: 2  # Instead of 1
  gradient_accumulation_steps: 4  # Instead of 8
```

### For Lower Memory Usage (if experiencing OOM)
```yaml
data:
  max_length: 2048  # Instead of 4096

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

## Using Real Nemotron Datasets

### With Nemotron Nano Model

To train the Nemotron Nano model on NVIDIA's official datasets, edit `configs/prototype_rtx4090_16gb_nemotron_nano.yaml`:

```yaml
data:
  hf_datasets:
    - "nvidia/Nemotron-CC"  # High-quality Common Crawl
    # - "nvidia/Nemotron-CC-Math-v1"  # Math-focused dataset (optional)
  
  custom_corpus_paths:
    - "./data/custom_corpus"  # Your custom data
  
  streaming: true  # Enable streaming for large datasets
  max_length: 2048  # Adjust based on your memory
```

### With Other Models

To train on NVIDIA's official datasets with standard models, edit `configs/prototype_rtx4090.yaml`:

```yaml
data:
  hf_datasets:
    - "nvidia/Nemotron-CC"  # Uncomment this line

  # For prototype, you may want to limit the dataset
  # The streaming mode will help manage memory
```

**Note:** Nemotron-CC is 6.3T tokens. For prototype testing, consider using a smaller dataset first or limiting max_steps.

**Recommended for 16GB VRAM:**
- Start with custom corpus only
- Add datasets gradually
- Use streaming mode: `streaming: true`
- Limit training steps: `max_steps: 5000`

## Next Steps

1. **Monitor training progress** - Check `checkpoints/prototype/` for saved models
2. **Adjust hyperparameters** - Edit `configs/prototype_rtx4090.yaml`
3. **Scale to production** - See README.md for multi-GPU setup
4. **Evaluate your model** - Load checkpoint and test on your use case

## Troubleshooting

### "CUDA out of memory"
- Reduce `max_length` to 2048
- Keep `per_device_train_batch_size: 1`
- Increase `gradient_accumulation_steps` to 16

### "Flash attention not available"
- Training will work without it, just slower
- Try: `pip install flash-attn --no-build-isolation --no-cache-dir`
- If installation fails due to filesystem issues, download wheel directly:
  ```bash
  wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  ```
- Or set `use_flash_attention: false` in config

### Slow download speeds
- HuggingFace datasets are large, first download takes time
- Data is cached, subsequent runs are faster
- Consider using `streaming: true` (already default)

### Model not found
- Ensure you're logged in: `huggingface-cli whoami`
- Check internet connection
- Verify model name in config

## Training on Limited Data

For quick prototyping with minimal data:

```yaml
training:
  max_steps: 1000  # Just 1000 steps
  save_steps: 100  # Save frequently
  logging_steps: 10  # Monitor closely
```

This will:
- Train for only 1000 steps (~20 minutes on RTX 4090)
- Create checkpoints every 100 steps
- Let you verify the pipeline works

## Support

If you encounter issues:
1. Check GPU is available: `nvidia-smi`
2. Verify CUDA version: Should be 12.1+ (tested with 12.7)
3. Check Python version: Should be 3.10.12+
4. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
5. Review logs in terminal for specific errors

### Tested Environment
- **Python**: 3.10.12
- **CUDA**: 12.7 (Driver 565.57.01)
- **GPU**: NVIDIA L40S (46GB VRAM)
- **PyTorch**: 2.5.1+cu121

Happy training! 🚀
