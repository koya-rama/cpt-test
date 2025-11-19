# Quick CPT Test on RTX 4090 Laptop

This is a minimal test to prove the CPT pipeline works on your laptop.

## What This Test Does

- ✅ Loads Phi-3 Mini 3.8B model
- ✅ Uses 3 sample text files from `data/custom_corpus/`
- ✅ Runs only **50 training steps** (~2-3 minutes)
- ✅ Uses short sequences (512 tokens) for speed
- ✅ Saves 1 checkpoint at step 25
- ✅ Proves the entire pipeline works

## Run the Quick Test

```powershell
# Single command - that's it!
python src\training\train_cpt.py --config configs\quick_test_phi.yaml
```

## What to Expect

### Phase 1: Model Loading (30 seconds)
```
INFO:__main__:Loading model: microsoft/Phi-3-mini-4k-instruct
Loading checkpoint shards: 100%|████████| 2/2 [00:00<00:00]
INFO:__main__:Model loaded: phi3
INFO:__main__:Model parameters: 3.82B
```

### Phase 2: Data Loading (5 seconds)
```
INFO:data.data_loader:Found 3 files to load
INFO:data.data_loader:Loaded 3 examples from custom corpus
Map: 100%|███████████| 3/3 [00:00<00:00]
INFO:data.data_loader:Dataset tokenization complete
```

### Phase 3: Training (2-3 minutes)
```
INFO:__main__:Starting training loop...
  0%|          | 0/50 [00:00<?, ?it/s]
{'loss': 2.345, 'grad_norm': 1.234, 'learning_rate': 1.5e-05, 'epoch': 0.1}
  10%|█         | 5/50 [00:20<03:00, 4.00s/it]
{'loss': 2.123, 'grad_norm': 0.987, 'learning_rate': 1.8e-05, 'epoch': 0.2}
...
{'train_runtime': 180.5, 'train_samples_per_second': 0.83}
```

### Phase 4: Checkpoint Saved
```
Saving model checkpoint to ./checkpoints/quick_test/checkpoint-25
Saving model checkpoint to ./checkpoints/quick_test/checkpoint-50
Training complete!
```

## Expected Performance

- **Total time**: 2-3 minutes
- **GPU memory**: ~10-11GB / 16GB
- **Speed**: ~4-5 seconds per step
- **Loss**: Should decrease from ~2.5 to ~2.0

## Monitor GPU (Optional)

In a separate terminal:

```powershell
# Activate environment
venv\Scripts\activate

# Watch GPU usage
python scripts\monitor_training.py --mode gpu
```

Or simple monitoring:

```powershell
# Watch GPU every 2 seconds
nvidia-smi -l 2
```

## After the Test

If successful, you'll have:

1. ✅ **Proven the pipeline works**
2. ✅ **Checkpoint saved** in `checkpoints/quick_test/`
3. ✅ **Training logs** showing loss decreasing
4. ✅ **Confidence** to run full CPT

## Next Steps After Success

### Option 1: Add More Data
```powershell
# Copy your real data
copy your_data.txt data\custom_corpus\
copy your_data.jsonl data\custom_corpus\
```

### Option 2: Run Longer Training
Edit `configs\quick_test_phi.yaml`:
```yaml
training:
  max_steps: 1000  # Change from 50 to 1000
```

### Option 3: Use Full Config
```powershell
# Use the full prototype config
python src\training\train_cpt.py --config configs\prototype_rtx4090_16gb_phi.yaml
```

### Option 4: Switch to Nemotron (After Approval)
```powershell
# After getting Nemotron access
python src\training\train_cpt.py --config configs\prototype_rtx4090_16gb_no_deepspeed.yaml
```

## Troubleshooting

### If you get OOM (Out of Memory):
```yaml
# Edit configs\quick_test_phi.yaml
data:
  max_length: 256  # Reduce from 512
```

### If training is slow:
- Close other GPU applications
- Check GPU temp isn't thermal throttling
- Reduce `dataloader_num_workers` to 0

### If you see warnings about Flash Attention:
- **Ignore them** - the config has `use_flash_attention: false`
- Training works fine without it

## Verify Checkpoint

After training completes:

```powershell
# List checkpoints
dir checkpoints\quick_test

# Should show:
# checkpoint-25/
# checkpoint-50/
```

Each checkpoint contains:
- `model.safetensors` - Model weights
- `config.json` - Model config
- `optimizer.pt` - Optimizer state
- `trainer_state.json` - Training state

## Performance Metrics

**Expected on RTX 4090 Laptop (16GB):**
- First step: ~15-20 seconds (compilation)
- Subsequent steps: ~4-5 seconds
- Total for 50 steps: ~2-3 minutes
- GPU usage: 10-11GB / 16GB
- GPU utilization: 80-95%

## Summary

This quick test proves:
- ✅ Model loads correctly
- ✅ Data processing works
- ✅ Tokenization works
- ✅ Training loop works
- ✅ Gradient updates work
- ✅ Checkpointing works
- ✅ Your GPU can handle CPT

**Just run:**
```powershell
python src\training\train_cpt.py --config configs\quick_test_phi.yaml
```

And you'll have a working CPT pipeline in 3 minutes! 🚀
