# Nemotron Continuous Pre-Training (CPT) Pipeline

A comprehensive pipeline for continuous pre-training of NVIDIA Nemotron 8B models using HuggingFace datasets and custom corpus data. Optimized for both prototype development on consumer GPUs (RTX 4090) and production-scale training on multi-GPU setups.

## Overview

This pipeline enables you to:
- **Continue pre-training** Nemotron models on domain-specific data
- **Use NVIDIA's Nemotron datasets** from HuggingFace (6.6T tokens of high-quality data)
- **Add custom corpus data** to adapt models to your specific use case
- **Optimize for different hardware** - from single RTX 4090 to multi-GPU clusters
- **Leverage modern optimizations** - Flash Attention, DeepSpeed ZeRO, gradient checkpointing

## Features

- **Data Loading**
  - HuggingFace datasets integration (Nemotron-CC, Nemotron-CC-Math, etc.)
  - Custom corpus support (txt, jsonl, parquet, csv)
  - Streaming mode for large datasets
  - Efficient tokenization and batching

- **Training Optimizations**
  - DeepSpeed ZeRO-2 and ZeRO-3 for memory efficiency
  - Flash Attention 2 for faster attention computation
  - Gradient checkpointing to reduce memory usage
  - Mixed precision training (BF16/FP16)
  - Gradient accumulation for larger effective batch sizes

- **Monitoring & Logging**
  - Weights & Biases integration
  - TensorBoard support
  - GPU monitoring utilities
  - Checkpoint tracking

## Project Structure

```
CPT/
├── configs/                          # Configuration files
│   ├── prototype_rtx4090.yaml       # RTX 4090 optimized config
│   ├── prototype_rtx4090_16gb_nemotron_nano.yaml  # Nemotron Nano 8B config
│   ├── prototype_rtx4090_16gb_llama.yaml          # Llama 3.2 3B config
│   ├── production_multi_gpu.yaml    # Multi-GPU production config
│   ├── ds_config_prototype.json     # DeepSpeed config for prototype
│   └── ds_config_production.json    # DeepSpeed config for production
│
├── data/                             # Data directory
│   ├── custom_corpus/               # Place your custom data here
│   └── cache/                       # HuggingFace cache
│
├── src/                              # Source code
│   ├── data/
│   │   └── data_loader.py           # Data loading and preprocessing
│   ├── training/
│   │   └── train_cpt.py             # Main training script
│   └── utils/                        # Utility functions
│
├── scripts/                          # Execution scripts
│   ├── run_prototype.sh             # Linux/Mac prototype training
│   ├── run_prototype.bat            # Windows prototype training
│   ├── run_production_multi_gpu.sh  # Multi-GPU training
│   ├── prepare_data.py              # Data preparation utilities
│   └── monitor_training.py          # Training monitoring
│
├── checkpoints/                      # Model checkpoints
├── logs/                             # Training logs
└── requirements.txt                  # Python dependencies
```

## Installation

### Prerequisites

- **Python**: 3.10.12+ (tested on 3.10.12)
- **CUDA**: 12.1+ (tested with 12.7 on NVIDIA L40S)
- **GPU Memory**:
  - Prototype: 16GB+ (RTX 4090, RTX 3090)
  - Production: 4x GPUs with 40GB+ each recommended (A100, H100, L40S)
- **Tested Hardware**: NVIDIA L40S (46GB VRAM), RTX 4090 (24GB)

### Setup

1. **Clone or navigate to the repository**
   ```bash
   cd C:\Makerslab\Projects\IndiaAI\Indus\CPT
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On Linux/Ubuntu
   source venv/bin/activate
   ```

3. **Install PyTorch with CUDA 12.1 (compatible with CUDA 12.7)**
   ```bash
   # PyTorch with CUDA 12.1 (works with CUDA 12.7 via backward compatibility)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install other dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install Flash Attention (optional but recommended)**
   ```bash
   # Method 1: Try direct installation
   pip install flash-attn --no-build-isolation --no-cache-dir

   # Method 2: If above fails due to filesystem issues, download wheel directly
   # For Python 3.10 + CUDA 12 + PyTorch 2.5:
   wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

   # Note: Flash Attention is optional. Training will work without it, just slower.
   ```

6. **Login to HuggingFace (required for Nemotron datasets)**
   ```bash
   huggingface-cli login
   ```

### Tested Environment

This pipeline has been tested on:
- **Python**: 3.10.12
- **CUDA**: 12.7 (Driver Version 565.57.01)
- **GPU**: NVIDIA L40S (46GB VRAM)
- **PyTorch**: 2.5.1+cu121
- **OS**: Ubuntu 22.04 LTS

## Quick Start - Prototype on RTX 4090

### Model Selection Quick Reference

| Model | Config File | VRAM | Context | Best For |
|-------|------------|------|---------|----------|
| **Nemotron-Nano-9B-v2** ⭐ | `l40s_nemotron_nano_9b.yaml` | 46GB | 4K-8K | **L40S, A100** (Latest & Best) |
| **Llama 3.2 3B** ✅ | `prototype_rtx4090_16gb_llama.yaml` | 16GB | 4K | **16GB systems** (Fast testing) |
| **Nemotron Nano 8B** | `prototype_rtx4090_16gb_nemotron_nano.yaml` | 20-24GB | 128K | Long context, 20GB+ VRAM |
| **Nemotron 8B** | `prototype_rtx4090.yaml` | 24GB | 4K | Full model training |

**Recommended:**
- **L40S / A100 (40GB+)**: Use **Nemotron-Nano-9B-v2** (latest, most capable)
- **RTX 4090 16GB**: Start with **Llama 3.2 3B**
- **RTX 4090 24GB**: Use **Nemotron Nano 8B** or **Nemotron 8B**

### Step 1: Prepare Sample Data

Create sample data for testing:
```bash
python scripts/prepare_data.py --create-sample
```

Or add your own data to `data/custom_corpus/`:
```bash
# Add your text files, jsonl, parquet, or csv files
cp your_data.txt data/custom_corpus/
```

### Step 2: Verify Data Loading

```bash
python scripts/prepare_data.py --verify configs/prototype_rtx4090.yaml
```

### Step 3: Run Training

**Using Nemotron-Nano-9B-v2 on L40S (46GB VRAM - RECOMMENDED):** ⭐
```bash
# Standard training
python src/training/train_cpt.py --config configs/l40s_nemotron_nano_9b.yaml

# With DeepSpeed for further optimization
python src/training/train_cpt.py --config configs/l40s_nemotron_nano_9b.yaml --deepspeed configs/ds_config_l40s.json
```

**Using Nemotron Nano 8B (Recommended for 16GB VRAM):**
```bash
python src/training/train_cpt.py --config configs/prototype_rtx4090_16gb_nemotron_nano.yaml
```

**Using Llama 3.2 3B (16GB VRAM, faster testing):**
```bash
python src/training/train_cpt.py --config configs/prototype_rtx4090_16gb_llama.yaml
```

**Using default script (24GB VRAM):**
```bash
# On Windows:
scripts\run_prototype.bat

# On Linux/Mac:
chmod +x scripts/run_prototype.sh
./scripts/run_prototype.sh

# Or run directly:
python src/training/train_cpt.py --config configs/prototype_rtx4090.yaml
```

### Step 4: Monitor Training

In a separate terminal:
```bash
# Monitor GPU usage
python scripts/monitor_training.py --mode gpu

# Monitor checkpoints
python scripts/monitor_training.py --mode checkpoint --checkpoint-dir checkpoints/prototype
```

## Production Training - Multi-GPU

### Configuration

Edit `configs/production_multi_gpu.yaml` to:
1. Enable desired HuggingFace datasets (Nemotron-CC, etc.)
2. Add your custom corpus paths
3. Adjust batch size based on available GPU memory
4. Configure Weights & Biases logging

### Running Multi-GPU Training

```bash
# For 4 GPUs (default)
./scripts/run_production_multi_gpu.sh

# For 8 GPUs
./scripts/run_production_multi_gpu.sh 8

# Or with DeepSpeed directly
deepspeed --num_gpus=4 src/training/train_cpt.py --config configs/production_multi_gpu.yaml
```

## Available Nemotron Datasets

The pipeline supports loading these NVIDIA datasets from HuggingFace:

| Dataset | Tokens | Description |
|---------|--------|-------------|
| `nvidia/Nemotron-CC` | 6.3T | High-quality Common Crawl data |
| `nvidia/Nemotron-CC-v2` | Extended | Extended Common Crawl (2024-2025) |
| `nvidia/Nemotron-CC-Math-v1` | 133B | Math-focused dataset with LaTeX |
| `nvidia/Nemotron-Pretraining-Code-v1` | - | Code datasets from GitHub |

To use these datasets, uncomment them in your config file:

```yaml
data:
  hf_datasets:
    - "nvidia/Nemotron-CC"
    - "nvidia/Nemotron-CC-Math-v1"
```

## Nemotron Models

### Available Models

- **nvidia/NVIDIA-Nemotron-Nano-9B-v2** ⭐ - **LATEST** 9B parameters, advanced architecture (L40S/A100)
- **nvidia/Llama-3.1-Nemotron-Nano-8B-v1** - Llama-based, 128K context, optimized efficiency
- **nvidia/nemotron-3-8b-base-4k** - 8B parameters, 4K context (standard)
- **nvidia/Nemotron-H-8B-Base-8K** - Hybrid architecture, 8K context
- **meta-llama/Llama-3.2-3B** - Smaller Llama model, 3B parameters (good for testing)

### Recommended Configurations

**For 40-48GB VRAM (L40S, A100 40GB/80GB):** ⭐ **BEST PERFORMANCE**
- ✅ Use `configs/l40s_nemotron_nano_9b.yaml` for **Nemotron-Nano-9B-v2** (Recommended)
- Can handle batch_size=2-4, max_length=4096-8192
- Expected throughput: ~5-8 tokens/sec with Flash Attention
- Memory usage: ~35-40GB / 46GB (plenty of headroom)

**For 16GB VRAM (RTX 4090 Laptop):**
- ✅ Use `configs/prototype_rtx4090_16gb_llama.yaml` for **Llama 3.2 3B** (Recommended)
- ⚠️ Nemotron Nano 8B requires 20GB+ VRAM for full training
- Note: 8B+ models are too large for 16GB VRAM without DeepSpeed ZeRO-3 with CPU offloading

**For 20-24GB VRAM (RTX 4090 Desktop, RTX 3090):**
- Use `configs/prototype_rtx4090_16gb_nemotron_nano.yaml` for Nemotron Nano 8B (with optimizations)
- Use `configs/prototype_rtx4090.yaml` for standard Nemotron models
- Can handle larger batch sizes and longer sequences

### Changing the Model

Edit your config file:
```yaml
model:
  name: "nvidia/NVIDIA-Nemotron-Nano-9B-v2"  # Change to your desired model
  use_flash_attention: true  # Highly recommended for performance
```

Or use a pre-configured file:
```bash
# For L40S with 9B model (Best)
python src/training/train_cpt.py --config configs/l40s_nemotron_nano_9b.yaml

# For RTX 4090 with 8B model
python src/training/train_cpt.py --config configs/prototype_rtx4090_16gb_nemotron_nano.yaml
```

## Configuration Guide

### Key Configuration Parameters

**Model Settings:**
```yaml
model:
  name: "nvidia/nemotron-3-8b-base-4k"
  use_flash_attention: true  # Requires flash-attn package
```

**Data Settings:**
```yaml
data:
  hf_datasets:
    - "nvidia/Nemotron-CC"  # HuggingFace datasets
  custom_corpus_paths:
    - "./data/custom_corpus"  # Your custom data
  max_length: 4096  # Sequence length
  streaming: true  # Stream large datasets
  num_proc: 4  # Parallel processing
```

**Training Settings:**
```yaml
training:
  per_device_train_batch_size: 1  # Batch size per GPU
  gradient_accumulation_steps: 8  # Effective batch = 1 * 8 = 8
  learning_rate: 2.0e-5
  max_steps: 10000  # Or set num_epochs
  bf16: true  # Use BF16 mixed precision
  gradient_checkpointing: true  # Save memory
  deepspeed_config: "./configs/ds_config_prototype.json"
```

### Memory Optimization Tips

**For L40S (46GB):** ⭐
- Batch size: 2-4
- Gradient accumulation: 4-8
- Max length: 4096-8192
- Enable gradient checkpointing
- Flash Attention highly recommended
- Optional: Use DeepSpeed ZeRO-2 for marginal gains

**For RTX 4090 (24GB):**
- Batch size: 1
- Gradient accumulation: 8-16
- Max length: 4096
- Enable gradient checkpointing
- Use DeepSpeed ZeRO-2 with CPU offload

**For Multi-GPU (80GB each):**
- Batch size: 4-8 per GPU
- Gradient accumulation: 4
- Max length: 8192
- Enable gradient checkpointing
- Use DeepSpeed ZeRO-3

## Monitoring Training

### Weights & Biases

Enable W&B in your config:
```yaml
training:
  use_wandb: true
  wandb_project: "nemotron-cpt"
  run_name: "my-training-run"
```

Then login:
```bash
wandb login
```

### TensorBoard

```bash
tensorboard --logdir checkpoints/prototype
```

### GPU Monitoring

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use the monitoring script
python scripts/monitor_training.py --mode gpu
```

## Resume Training

To resume from a checkpoint, edit your config:
```yaml
training:
  resume_from_checkpoint: "./checkpoints/prototype/checkpoint-1000"
```

## Troubleshooting

### Out of Memory (OOM)

1. **Reduce batch size**: Set `per_device_train_batch_size: 1`
2. **Increase gradient accumulation**: `gradient_accumulation_steps: 16`
3. **Reduce sequence length**: `max_length: 2048`
4. **Enable DeepSpeed**: Uncomment `deepspeed_config` in config
5. **Use ZeRO-3**: Switch to `ds_config_production.json`

### Slow Training

1. **Enable Flash Attention**: `use_flash_attention: true`
   - Install: `pip install flash-attn --no-build-isolation --no-cache-dir`
   - If fails, download wheel: See Installation section above
2. **Increase batch size** if memory allows
3. **Reduce gradient checkpointing** if memory is not an issue
4. **Use multiple GPUs** with DeepSpeed
5. **Increase dataloader workers**: `dataloader_num_workers: 8`

### Data Loading Issues

1. **Verify data format**: Use `scripts/prepare_data.py --verify`
2. **Check HuggingFace login**: `huggingface-cli whoami`
3. **Enable streaming**: `streaming: true` for large datasets
4. **Check disk space**: Ensure sufficient space for cache

## Performance Benchmarks

### L40S (46GB) - Nemotron-Nano-9B-v2 ⭐
- Model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
- Batch size: 2
- Gradient accumulation: 8
- Sequence length: 4096
- Flash Attention: Enabled
- **Expected**: ~5-8 tokens/sec/GPU
- **Memory**: ~35-40GB / 46GB

### RTX 4090 (24GB)
- Model: nemotron-3-8b-base-4k
- Batch size: 1
- Gradient accumulation: 8
- Sequence length: 4096
- **Expected**: ~2-3 tokens/sec/GPU
- **Memory**: ~20-22GB / 24GB

### 4x A100 (80GB each)
- Model: Nemotron-H-8B-Base-8K
- Batch size: 4 per GPU
- Gradient accumulation: 4
- Sequence length: 8192
- **Expected**: ~40-50 tokens/sec total

## Advanced Usage

### Custom Data Format

The pipeline supports multiple data formats:

**Text files (.txt):**
```
data/custom_corpus/document1.txt
data/custom_corpus/document2.txt
```

**JSONL (.jsonl):**
```json
{"text": "Your text here..."}
{"text": "More text..."}
```

**Parquet (.parquet):**
```python
import pandas as pd
df = pd.DataFrame({"text": ["Text 1", "Text 2"]})
df.to_parquet("data/custom_corpus/data.parquet")
```

### Using LoRA/QLoRA

For parameter-efficient fine-tuning, you can modify the training script to use PEFT:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
```

## Citation

If you use Nemotron models or datasets, please cite:

```bibtex
@article{nemotron2024,
  title={Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset},
  author={NVIDIA},
  year={2024}
}
```

## Resources

- [Nemotron Models on HuggingFace](https://huggingface.co/nvidia)
- [Nemotron-CC Paper](https://arxiv.org/abs/2412.02595)
- [NVIDIA Blog: Building Nemotron-CC](https://developer.nvidia.com/blog/building-nemotron-cc-a-high-quality-trillion-token-dataset-for-llm-pretraining-from-common-crawl-using-nvidia-nemo-curator/)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## License

This project is provided as-is. Please check the licenses of individual models and datasets:
- Nemotron models: [NVIDIA Open Model License](https://developer.nvidia.com/downloads/nemo-open-model-license-agreement)
- Datasets: Check individual dataset cards on HuggingFace

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify your environment:
   - `nvidia-smi` - Check GPU availability
   - `python --version` - Should be 3.10.12+
   - `python -c "import torch; print(torch.cuda.is_available())"` - Should print True
   - `python -c "import torch; print(torch.version.cuda)"` - Should show CUDA version
3. Review HuggingFace model/dataset documentation
4. Check NVIDIA NeMo documentation
5. Open an issue in your repository

## Acknowledgments

- NVIDIA for Nemotron models and datasets
- HuggingFace for the Transformers library
- Microsoft for DeepSpeed
- The open-source AI community

---

**Happy Training!** 🚀
