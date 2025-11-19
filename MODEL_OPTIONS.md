# Model Options for CPT Pipeline

## Available Models (16GB VRAM)

### 🔓 Open Access (No Gating - Use Immediately)

| Model | Params | Context | Config File | Access |
|-------|--------|---------|-------------|--------|
| **Llama 3.2 3B** | 3B | 128K | `prototype_rtx4090_16gb_llama.yaml` | ✅ Open |
| **Phi-3 Mini** | 3.8B | 4K | `prototype_rtx4090_16gb_phi.yaml` | ✅ Open |
| **StableLM 2** | 1.6B | 4K | - | ✅ Open |
| **Qwen 2.5** | 3B | 32K | - | ✅ Open |

### 🔐 Gated Access (Requires Approval)

| Model | Params | Context | Config File | Status |
|-------|--------|---------|-------------|--------|
| **Nemotron 3 8B** | 8B | 4K | `prototype_rtx4090_16gb_no_deepspeed.yaml` | ⚠️ Need approval |
| **Llama 3.1 8B** | 8B | 128K | - | ⚠️ Need approval |

## Quick Start Options

### Option 1: Llama 3.2 3B (Recommended for Testing)

**Best for:** Quick start, testing pipeline

```powershell
# Create sample data
python scripts\prepare_data.py --create-sample

# Train with Llama 3.2 3B
python src\training\train_cpt.py --config configs\prototype_rtx4090_16gb_llama.yaml
```

**Advantages:**
- ✅ No gating - works immediately
- ✅ 128K context window
- ✅ High quality (Meta)
- ✅ Fits easily in 16GB

### Option 2: Phi-3 Mini 3.8B

**Best for:** Microsoft ecosystem, instruction following

```powershell
# Train with Phi-3 Mini
python src\training\train_cpt.py --config configs\prototype_rtx4090_16gb_phi.yaml
```

**Advantages:**
- ✅ No gating
- ✅ Strong performance for size
- ✅ Microsoft support
- ✅ Good for reasoning tasks

### Option 3: Nemotron 8B (After Approval)

**Best for:** Maximum performance, production use

**Steps:**
1. Request access: https://huggingface.co/nvidia/nemotron-3-8b-base-4k
2. Wait for approval (usually instant)
3. Re-login to HuggingFace
4. Use original config

```powershell
# After getting access
python src\training\train_cpt.py --config configs\prototype_rtx4090_16gb_no_deepspeed.yaml
```

## Request Access to Nemotron

### Step 1: Visit Model Page
https://huggingface.co/nvidia/nemotron-3-8b-base-4k

### Step 2: Accept License
Click "Request Access" or "Accept License"

### Step 3: Wait for Approval
- Usually instant
- Check your email for confirmation

### Step 4: Re-login
```powershell
huggingface-cli logout
huggingface-cli login
# Use same token, but now with Nemotron access
```

### Step 5: Verify Access
```powershell
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('nvidia/nemotron-3-8b-base-4k'); print('✓ Access granted!')"
```

## Model Comparison

### Memory Usage (16GB VRAM)

| Model | Size | VRAM Used | Speed | Available |
|-------|------|-----------|-------|-----------|
| Llama 3.2 3B | 3B | ~10GB | Fast | ✅ Now |
| Phi-3 Mini | 3.8B | ~11GB | Fast | ✅ Now |
| Nemotron 8B | 8B | ~15GB | Medium | ⏳ After approval |

### Performance Characteristics

**Llama 3.2 3B:**
- 🟢 Excellent general knowledge
- 🟢 128K context (great for long documents)
- 🟢 Meta's latest architecture
- 🟡 Smaller than Nemotron

**Phi-3 Mini 3.8B:**
- 🟢 Strong reasoning capabilities
- 🟢 Instruction following
- 🟢 Code generation
- 🟡 Only 4K context

**Nemotron 8B:**
- 🟢 Largest model (best quality)
- 🟢 NVIDIA optimized
- 🟢 Great for CPT
- 🔴 Requires approval

## Recommended Workflow

### Phase 1: Test Pipeline (Today)
Use **Llama 3.2 3B** or **Phi-3 Mini**
- ✅ Works immediately
- ✅ Test data pipeline
- ✅ Verify training works
- ✅ Quick iterations

### Phase 2: Production (After Approval)
Switch to **Nemotron 8B**
- ⏳ Request access today
- ⏳ Get approval (hours to days)
- ✅ Better quality
- ✅ Full 8B parameters

## Alternative: Other Nemotron Models

Some Nemotron models may not be gated:

```powershell
# Try Nemotron Instruct (might be open)
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('nvidia/Llama-3.1-Nemotron-Nano-8B-v1'); print('Model accessible!')"
```

Check these models:
- `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`
- `nvidia/Nemotron-H-8B-Base-8K` (if released as open)

## Quick Test Commands

**Test Llama 3.2 3B access:**
```powershell
python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B'); print('✓ Llama 3.2 3B accessible')"
```

**Test Phi-3 Mini access:**
```powershell
python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct'); print('✓ Phi-3 Mini accessible')"
```

**Test Nemotron access:**
```powershell
python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('nvidia/nemotron-3-8b-base-4k'); print('✓ Nemotron accessible')"
```

## Summary

**For immediate start:**
```powershell
# Use Llama 3.2 3B (no waiting)
python src\training\train_cpt.py --config configs\prototype_rtx4090_16gb_llama.yaml
```

**For best quality (after approval):**
```powershell
# Request access to Nemotron, then:
python src\training\train_cpt.py --config configs\prototype_rtx4090_16gb_no_deepspeed.yaml
```

Choose based on your timeline! 🚀
