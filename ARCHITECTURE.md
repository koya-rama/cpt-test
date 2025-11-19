# Architecture Overview

## Pipeline Components

### 1. Data Loading (`src/data/data_loader.py`)

**CPTDataLoader** - Main class for loading and preprocessing data

Key features:
- **Multi-source loading**: HuggingFace datasets + custom corpus
- **Streaming support**: Handle datasets larger than memory
- **Efficient tokenization**: Batched processing with multiprocessing
- **Format support**: txt, jsonl, parquet, csv

Flow:
```
Raw Data → Load → Tokenize → Chunk → DataLoader
```

### 2. Training (`src/training/train_cpt.py`)

**CPTTrainer** - Orchestrates the training process

Components:
- **Model loading**: AutoModelForCausalLM with optimizations
- **Data preparation**: Dataset loading and preprocessing
- **Training loop**: HuggingFace Trainer with DeepSpeed
- **Checkpointing**: Automatic saving and resuming

Optimizations:
- Flash Attention 2
- Gradient checkpointing
- Mixed precision (BF16)
- DeepSpeed ZeRO

### 3. Configuration System

**YAML-based configs** for different scenarios:
- `prototype_rtx4090.yaml` - Single GPU, limited resources
- `production_multi_gpu.yaml` - Multi-GPU, full scale

**DeepSpeed configs**:
- `ds_config_prototype.json` - ZeRO-2 with CPU offload
- `ds_config_production.json` - ZeRO-3 for maximum scale

## Training Process

### Initialization

```
1. Load config → Parse YAML
2. Setup model → Download/cache Nemotron
3. Setup tokenizer → Load and configure
4. Load datasets → HF + custom corpus
5. Initialize trainer → Configure optimizations
```

### Training Loop

```
For each batch:
  1. Load batch from dataloader
  2. Forward pass (with mixed precision)
  3. Compute loss (causal LM objective)
  4. Backward pass (with gradient accumulation)
  5. Optimizer step (DeepSpeed)
  6. Log metrics (W&B, TensorBoard)
  7. Save checkpoint (periodic)
```

### Memory Management

**Gradient Checkpointing**:
- Trades compute for memory
- Recomputes activations during backward pass
- ~30% slower, ~50% less memory

**DeepSpeed ZeRO**:
- **Stage 1**: Optimizer state partitioning
- **Stage 2**: + Gradient partitioning
- **Stage 3**: + Parameter partitioning

**CPU Offload**:
- Offload optimizer states to CPU RAM
- Slower but enables larger models

## Data Flow

```
┌─────────────────┐
│  HuggingFace    │
│  Datasets       │
└────────┬────────┘
         │
         ↓
    ┌────────────┐      ┌─────────────┐
    │   Custom   │  →   │   Combine   │
    │   Corpus   │      │  Datasets   │
    └────────────┘      └──────┬──────┘
                               │
                               ↓
                        ┌──────────────┐
                        │   Tokenize   │
                        │  (batched)   │
                        └──────┬───────┘
                               │
                               ↓
                        ┌──────────────┐
                        │   Dataloader │
                        │  (streaming) │
                        └──────┬───────┘
                               │
                               ↓
                        ┌──────────────┐
                        │   Training   │
                        │     Loop     │
                        └──────────────┘
```

## Model Architecture

### Nemotron-3-8B-Base-4K
- **Parameters**: 8B
- **Architecture**: Transformer decoder
- **Context**: 4096 tokens
- **Vocab size**: 32k tokens
- **Layers**: 32
- **Attention heads**: 32
- **Hidden size**: 4096

### Nemotron-H-8B-Base-8K
- **Parameters**: 8B
- **Architecture**: Hybrid (Mamba-2 + Attention)
- **Mamba layers**: 28
- **Attention layers**: 4
- **Context**: 8192 tokens

## Optimization Strategies

### For RTX 4090 (24GB)

```yaml
Memory Budget:
- Model (BF16): ~16GB
- Optimizer: ~4GB (with CPU offload)
- Activations: ~4GB (with grad checkpoint)
- Buffer: ~2GB
Total: ~22GB / 24GB ✓

Settings:
- Batch size: 1
- Grad accumulation: 8
- Sequence length: 4096
- DeepSpeed ZeRO-2
- CPU offload: ON
```

### For 4x A100 (80GB each)

```yaml
Memory Budget per GPU:
- Model shard: ~4GB (ZeRO-3)
- Optimizer: ~8GB (ZeRO-3)
- Gradients: ~8GB (ZeRO-3)
- Activations: ~50GB
- Buffer: ~10GB
Total: ~70GB / 80GB ✓

Settings:
- Batch size: 4-8 per GPU
- Grad accumulation: 4
- Sequence length: 8192
- DeepSpeed ZeRO-3
- CPU offload: OFF
```

## Checkpointing Strategy

### Checkpoint Structure
```
checkpoint-1000/
├── config.json              # Model config
├── model.safetensors        # Model weights
├── optimizer.pt             # Optimizer state
├── scheduler.pt             # LR scheduler state
├── trainer_state.json       # Training state
├── training_args.bin        # Training arguments
└── rng_state.pth           # Random state
```

### Resume Logic
1. Load model from checkpoint
2. Restore optimizer state
3. Restore LR scheduler
4. Resume from exact step
5. Continue training

## Distributed Training

### Data Parallel
- Same model on each GPU
- Different data batches
- Gradient all-reduce

### DeepSpeed ZeRO
- Model sharded across GPUs
- Reduced memory per GPU
- Enables larger models/batches

### Communication
- NCCL backend for GPU-GPU
- Overlapped communication
- Gradient bucketing

## Performance Considerations

### Bottlenecks
1. **Data loading**: Use streaming + multiple workers
2. **Attention**: Use Flash Attention 2
3. **Memory**: Use gradient checkpointing + DeepSpeed
4. **I/O**: Cache datasets, use fast storage for checkpoints

### Profiling
```python
# Add to trainer
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    trainer.train()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Extension Points

### Custom Data Processing
- Extend `CPTDataLoader`
- Add custom tokenization
- Implement data filtering

### Custom Training Loop
- Inherit from `CPTTrainer`
- Override `train()` method
- Add custom callbacks

### Model Modifications
- Load with PEFT (LoRA)
- Custom attention patterns
- Modified loss functions

## Security Considerations

### Data Privacy
- Local caching of datasets
- No data sent to external services (unless W&B enabled)
- HuggingFace authentication required for some datasets

### Model Integrity
- Use safetensors format
- Verify model checksums
- Trust remote code cautiously

## Monitoring

### Metrics Tracked
- Loss (training, validation)
- Learning rate
- Gradient norm
- GPU memory usage
- Throughput (tokens/sec)

### Logging Backends
- Console (stdout)
- TensorBoard
- Weights & Biases
- Custom loggers

## Future Enhancements

Potential improvements:
1. **Data augmentation**: Back-translation, paraphrasing
2. **Curriculum learning**: Start with easier examples
3. **Dynamic batching**: Variable sequence lengths
4. **Model parallelism**: Tensor/pipeline parallelism for larger models
5. **Mixed dataset sampling**: Weighted sampling from different sources
6. **Evaluation during training**: Automatic benchmark evaluation
7. **Hyperparameter tuning**: Ray Tune integration
8. **Model compression**: Quantization, pruning

---

For implementation details, see the source code in `src/` directory.
