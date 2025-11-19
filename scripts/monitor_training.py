#!/usr/bin/env python3
"""
Script to monitor training progress
"""

import os
import sys
import argparse
import time
from pathlib import Path
import json

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def monitor_gpu():
    """Monitor GPU usage"""
    if not GPU_AVAILABLE:
        print("GPUtil not installed. Install with: pip install gputil")
        return

    while True:
        gpus = GPUtil.getGPUs()

        print("\n" + "="*60)
        print(f"GPU Monitoring - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        for gpu in gpus:
            print(f"\nGPU {gpu.id}: {gpu.name}")
            print(f"  Temperature: {gpu.temperature}°C")
            print(f"  GPU Load: {gpu.load * 100:.1f}%")
            print(f"  Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)")
            print(f"  Free Memory: {gpu.memoryFree}MB")

        time.sleep(5)


def monitor_checkpoints(checkpoint_dir: str):
    """Monitor checkpoint progress"""
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    while True:
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))

        print("\n" + "="*60)
        print(f"Checkpoint Monitoring - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Number of checkpoints: {len(checkpoints)}\n")

        for ckpt in checkpoints[-5:]:  # Show last 5 checkpoints
            trainer_state_file = ckpt / "trainer_state.json"
            if trainer_state_file.exists():
                with open(trainer_state_file) as f:
                    state = json.load(f)

                print(f"{ckpt.name}:")
                if 'log_history' in state and state['log_history']:
                    last_log = state['log_history'][-1]
                    for key, value in last_log.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")

        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument(
        "--mode",
        choices=["gpu", "checkpoint"],
        default="gpu",
        help="Monitoring mode"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/prototype",
        help="Checkpoint directory to monitor"
    )

    args = parser.parse_args()

    try:
        if args.mode == "gpu":
            monitor_gpu()
        elif args.mode == "checkpoint":
            monitor_checkpoints(args.checkpoint_dir)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()
