"""
Continuous Pre-Training (CPT) script for Nemotron models
Supports DeepSpeed, gradient accumulation, mixed precision, and distributed training
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict
import json

import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from data.data_loader import get_nemotron_data_loader, CPTDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CPTTrainer:
    """
    Trainer for Continuous Pre-Training of Nemotron models
    """

    def __init__(self, config: Dict):
        """
        Initialize the CPT trainer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None

    def setup_model(self):
        """Load and configure the model"""
        logger.info(f"Loading model: {self.config['model']['name']}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optimizations
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.config['training'].get('bf16', True) else torch.float16,
        }

        # Add flash attention if enabled
        if self.config['model'].get('use_flash_attention', True):
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            **model_kwargs
        )

        logger.info(f"Model loaded: {self.model.config.model_type}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")

        # Enable gradient checkpointing to save memory
        if self.config['training'].get('gradient_checkpointing', True):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    def setup_data(self):
        """Load and prepare datasets"""
        logger.info("Setting up datasets...")

        data_config = self.config['data']

        # Get HuggingFace datasets
        hf_datasets = data_config.get('hf_datasets', [])
        custom_paths = data_config.get('custom_corpus_paths', [])

        if not hf_datasets and not custom_paths:
            raise ValueError("No datasets specified in config")

        # Create data loader
        data_loader = CPTDataLoader(
            tokenizer_name=self.config['model']['name'],
            max_length=data_config.get('max_length', 4096),
            streaming=data_config.get('streaming', True),
        )

        datasets_to_combine = []

        # Load HuggingFace datasets
        if hf_datasets:
            logger.info(f"Loading HuggingFace datasets: {hf_datasets}")
            hf_dataset = data_loader.load_nemotron_datasets(
                dataset_names=hf_datasets,
                num_proc=data_config.get('num_proc', 4),
            )
            datasets_to_combine.append(hf_dataset)

        # Load custom corpus
        if custom_paths:
            logger.info(f"Loading custom corpus from: {custom_paths}")
            custom_dataset = data_loader.load_custom_corpus(
                data_paths=custom_paths,
                text_column=data_config.get('text_column', 'text'),
            )
            datasets_to_combine.append(custom_dataset)

        # Combine datasets
        if len(datasets_to_combine) > 1:
            if data_config.get('streaming', True):
                from datasets import interleave_datasets
                combined = interleave_datasets(datasets_to_combine)
            else:
                from datasets import concatenate_datasets
                combined = concatenate_datasets(datasets_to_combine)
        else:
            combined = datasets_to_combine[0]

        # Prepare dataset
        self.train_dataset = data_loader.prepare_dataset(
            dataset=combined,
            text_column=data_config.get('text_column', 'text'),
            num_proc=data_config.get('num_proc', 4),
        )

        logger.info("Dataset setup complete")

        # Setup eval dataset if specified
        if data_config.get('eval_dataset'):
            logger.info("Loading evaluation dataset...")
            eval_data = load_dataset(
                data_config['eval_dataset'],
                split='validation',
                streaming=data_config.get('streaming', True),
            )
            self.eval_dataset = data_loader.prepare_dataset(
                dataset=eval_data,
                text_column=data_config.get('text_column', 'text'),
                num_proc=data_config.get('num_proc', 4),
            )

    def train(self):
        """Run the training loop"""
        logger.info("Starting training...")

        training_config = self.config['training']

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],

            # Training hyperparameters
            num_train_epochs=training_config.get('num_epochs', 1),
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),

            # Optimization
            learning_rate=training_config.get('learning_rate', 2e-5),
            weight_decay=training_config.get('weight_decay', 0.01),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            warmup_steps=training_config.get('warmup_steps', 500),
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),

            # Mixed precision
            bf16=training_config.get('bf16', True),
            fp16=training_config.get('fp16', False),

            # Logging and checkpointing
            logging_steps=training_config.get('logging_steps', 10),
            save_steps=training_config.get('save_steps', 1000),
            save_total_limit=training_config.get('save_total_limit', 3),
            eval_strategy=training_config.get('evaluation_strategy', 'no'),  # Updated parameter name
            eval_steps=training_config.get('eval_steps', 1000) if self.eval_dataset else None,

            # Performance optimizations
            dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
            dataloader_pin_memory=True,
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),

            # DeepSpeed
            deepspeed=training_config.get('deepspeed_config'),

            # W&B logging
            report_to=["wandb"] if training_config.get('use_wandb', False) else [],
            run_name=training_config.get('run_name', 'nemotron-cpt'),

            # Checkpointing
            load_best_model_at_end=False,
            save_safetensors=True,

            # Misc
            remove_unused_columns=False,
            max_steps=training_config.get('max_steps', -1),
        )

        # Data collator for causal language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        # Initialize wandb if enabled
        if training_config.get('use_wandb', False):
            wandb.init(
                project=training_config.get('wandb_project', 'nemotron-cpt'),
                name=training_config.get('run_name', 'nemotron-cpt'),
                config=self.config,
            )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Train
        logger.info("Starting training loop...")
        train_result = trainer.train(
            resume_from_checkpoint=training_config.get('resume_from_checkpoint')
        )

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        trainer.save_state()

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        logger.info("Training complete!")

        # Cleanup
        if training_config.get('use_wandb', False):
            wandb.finish()

    def run(self):
        """Main execution flow"""
        self.setup_model()
        self.setup_data()
        self.train()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Continuous Pre-Training for Nemotron")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Log configuration
    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=2))

    # Create trainer and run
    trainer = CPTTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
