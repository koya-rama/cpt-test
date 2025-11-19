"""
Helper utilities for the CPT pipeline
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import torch


def get_gpu_memory_info():
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3

        info[f"GPU {i}"] = {
            "name": props.name,
            "total_memory_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(total - reserved, 2),
        }

    return info


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "total_billions": round(total / 1e9, 2),
        "trainable_billions": round(trainable / 1e9, 2),
    }


def estimate_memory_usage(
    num_parameters: int,
    batch_size: int,
    sequence_length: int,
    precision: str = "bf16",
) -> Dict[str, float]:
    """
    Estimate GPU memory usage

    Args:
        num_parameters: Number of model parameters
        batch_size: Batch size
        sequence_length: Sequence length
        precision: bf16, fp16, or fp32

    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_param = {
        "fp32": 4,
        "bf16": 2,
        "fp16": 2,
    }[precision]

    # Model weights
    model_memory = num_parameters * bytes_per_param

    # Optimizer states (assuming AdamW)
    # AdamW stores: gradients + first moment + second moment
    optimizer_memory = num_parameters * 4 * 3  # Always fp32

    # Activations (rough estimate)
    # Depends on model architecture, this is approximate
    activation_memory = (
        batch_size * sequence_length * num_parameters * bytes_per_param * 0.01
    )

    # Convert to GB
    total_gb = (model_memory + optimizer_memory + activation_memory) / 1024**3

    return {
        "model_gb": round(model_memory / 1024**3, 2),
        "optimizer_gb": round(optimizer_memory / 1024**3, 2),
        "activation_gb": round(activation_memory / 1024**3, 2),
        "total_gb": round(total_gb, 2),
    }


def save_metrics(metrics: Dict[str, Any], output_path: str):
    """Save metrics to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file"""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_number(num: float) -> str:
    """Format large numbers with suffixes"""
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


def print_training_info(config: Dict, model, tokenizer):
    """Print comprehensive training information"""
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    print(f"\nModel: {config['model']['name']}")
    params = count_parameters(model)
    print(f"Parameters: {format_number(params['total'])} ({params['total_billions']}B)")
    print(f"Trainable: {format_number(params['trainable'])} ({params['trainable_billions']}B)")

    print(f"\nTokenizer:")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Max length: {config['data']['max_length']}")

    print(f"\nTraining:")
    print(f"  Batch size per device: {config['training']['per_device_train_batch_size']}")
    print(f"  Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    effective_batch = (
        config['training']['per_device_train_batch_size'] *
        config['training']['gradient_accumulation_steps']
    )
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Precision: {'BF16' if config['training'].get('bf16') else 'FP16'}")

    # Memory estimate
    mem = estimate_memory_usage(
        num_parameters=params['total'],
        batch_size=config['training']['per_device_train_batch_size'],
        sequence_length=config['data']['max_length'],
        precision='bf16' if config['training'].get('bf16') else 'fp16',
    )
    print(f"\nEstimated Memory Usage (per GPU):")
    print(f"  Model: {mem['model_gb']} GB")
    print(f"  Optimizer: {mem['optimizer_gb']} GB")
    print(f"  Activations: {mem['activation_gb']} GB")
    print(f"  Total: {mem['total_gb']} GB")

    # GPU info
    gpu_info = get_gpu_memory_info()
    if "error" not in gpu_info:
        print(f"\nAvailable GPUs:")
        for gpu_name, info in gpu_info.items():
            print(f"  {gpu_name}: {info['name']}")
            print(f"    Total: {info['total_memory_gb']} GB")
            print(f"    Free: {info['free_gb']} GB")

    print("\n" + "=" * 70)
