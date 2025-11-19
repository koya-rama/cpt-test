# Flash Attention on Windows - Installation Guide

Flash Attention has issues on Windows due to:
1. **Long file paths** (exceeds Windows 260-char limit)
2. **Compilation complexity** (requires CUDA toolkit + VS Build Tools)
3. **Limited Windows support**

## Solutions (Choose One)

### Solution 1: Enable Long Paths (Recommended)

**Step 1: Enable Long Paths in Windows**

Run PowerShell **as Administrator**:

```powershell
# Enable long paths in Windows Registry
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Enable long paths in Git (if using)
git config --system core.longpaths true
```

**Step 2: Restart your computer** (required for changes to take effect)

**Step 3: Try installing again**

```powershell
pip install flash-attn --no-build-isolation
```

### Solution 2: Use Pre-built Wheel (Easiest)

Check if pre-built wheels are available:

```powershell
# Try installing pre-built version
pip install flash-attn --prefer-binary
```

If not available, use **Solution 3**.

### Solution 3: Skip Flash Attention (Simplest)

**Flash Attention is OPTIONAL.** Your training will work fine without it, just ~30-40% slower.

**Disable in your config:**

Edit `configs\prototype_rtx4090_16gb_no_deepspeed.yaml`:

```yaml
model:
  name: "nvidia/nemotron-3-8b-base-4k"
  use_flash_attention: false  # Changed from true
```

**Training will still work!** You'll just use standard PyTorch attention.

### Solution 4: Use xFormers (Alternative)

xFormers is easier to install on Windows and provides similar speedups:

```powershell
pip install xformers
```

Then modify the training script to use xFormers instead of Flash Attention.

### Solution 5: Use Shorter Temp Path

Change Windows temp directory to shorter path:

```powershell
# Set temp to shorter path
$env:TMP = "C:\Temp"
$env:TEMP = "C:\Temp"
New-Item -ItemType Directory -Force -Path C:\Temp

# Try installing
pip install flash-attn --no-build-isolation
```

## Comparison: With vs Without Flash Attention

| Metric | Without Flash Attn | With Flash Attn |
|--------|-------------------|-----------------|
| **Speed** | 1.5-2 tok/s | 2.5-3.5 tok/s |
| **Memory** | ~15GB | ~13GB |
| **Installation** | ✅ Easy | ⚠️ Complex |
| **Windows Support** | ✅ Full | ⚠️ Limited |

## Recommended Approach for Windows

1. **Try Solution 1** (Enable long paths + restart)
2. **If fails**: Use **Solution 3** (Skip Flash Attention)
3. Training works great either way!

## Enable Flash Attention Later

You can always install Flash Attention later. For now, train without it:

```yaml
# configs/prototype_rtx4090_16gb_no_deepspeed.yaml
model:
  use_flash_attention: false
```

Start training immediately:

```powershell
python src\training\train_cpt.py --config configs\prototype_rtx4090_16gb_no_deepspeed.yaml
```

## Performance Impact

**Without Flash Attention on RTX 4090 (16GB):**
- Training speed: 1.5-2 tokens/second
- Memory usage: ~15GB / 16GB
- **Still very usable!**

**With Flash Attention:**
- Training speed: 2.5-3.5 tokens/second
- Memory usage: ~13GB / 16GB
- Nice to have, but not critical

## Alternative: Use Smaller Sequences

If you want faster training without Flash Attention:

```yaml
data:
  max_length: 1024  # Reduce from 2048
```

Shorter sequences train faster and use less memory.

## Summary

**For immediate training:**
- Set `use_flash_attention: false`
- Start training right away
- Revisit Flash Attention later if needed

**For maximum performance:**
- Enable Windows long paths (requires admin + restart)
- Install Flash Attention
- ~30% speed improvement

Choose what works best for you! 🚀
