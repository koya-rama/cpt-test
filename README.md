# Multilingual Physics Education LLM - Continued Pretraining

This repository contains the training pipeline for a multilingual physics education LLM targeting Indian students (grades 6-12) with support for English, Hindi, and Bhojpuri.

## 🎯 Project Goals
- Develop specialized physics LLM for Indian curriculum
- Support multilingual education (English, Hindi, Bhojpuri)
- Implement efficient training with knowledge distillation

## 🚀 Quick Start

### Installation
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Training
\`\`\`bash
# For 7B model
python src/training/train_cpt.py --config configs/l40s_mistral_7b_minimal.yaml

# For 9B model with DeepSpeed
DEEPSPEED_NO_MPI=1 python src/training/train_cpt.py --config configs/l40s_nemotron_nano_9b.yaml
\`\`\`

## 📁 Project Structure
- `src/`: Source code for training pipeline
- `configs/`: YAML configuration files
- `data/`: Training data directory
- `checkpoints/`: Model checkpoints

## 🛠️ Tech Stack
- **Base Model**: nvidia/Mistral-NeMo-Minitron-8B-Base
- **Training**: DeepSpeed ZeRO-2, Gradient Checkpointing
- **Tracking**: Weights & Biases
- **Hardware**: L40S GPU (48GB)

## 📊 Training Configuration
- Sequence length: 2048 tokens
- Batch size: 1 per device
- Gradient accumulation: 16 steps
- Precision: BF16
- Optimizer: AdamW 8-bit / DeepSpeed

## 📝 License
[Add your license]

## 👥 Contributors
- [Your name]
