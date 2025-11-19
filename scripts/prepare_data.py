#!/usr/bin/env python3
"""
Script to prepare and verify data before training
"""

import os
import sys
from pathlib import Path
import argparse
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data.data_loader import CPTDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_sample_data(output_dir: str = "./data/custom_corpus"):
    """
    Create sample data for testing the pipeline
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create sample text files
    samples = [
        "This is a sample text for continuous pre-training. " * 100,
        "Nemotron is a family of language models developed by NVIDIA. " * 100,
        "Continuous pre-training helps adapt models to specific domains. " * 100,
    ]

    for i, text in enumerate(samples):
        file_path = os.path.join(output_dir, f"sample_{i}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)

    logger.info(f"Created {len(samples)} sample files in {output_dir}")


def verify_data(config_path: str):
    """
    Verify that data can be loaded correctly
    """
    import yaml

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Initializing data loader...")
    data_loader = CPTDataLoader(
        tokenizer_name=config['model']['name'],
        max_length=config['data']['max_length'],
        streaming=config['data']['streaming'],
    )

    # Try loading datasets
    datasets_to_load = []

    if config['data'].get('hf_datasets'):
        logger.info(f"HuggingFace datasets: {config['data']['hf_datasets']}")
        datasets_to_load.extend(config['data']['hf_datasets'])

    if config['data'].get('custom_corpus_paths'):
        logger.info(f"Custom corpus paths: {config['data']['custom_corpus_paths']}")

        custom_paths = config['data']['custom_corpus_paths']
        try:
            custom_dataset = data_loader.load_custom_corpus(
                data_paths=custom_paths,
                text_column=config['data'].get('text_column', 'text'),
            )
            logger.info(f"Successfully loaded custom corpus: {len(custom_dataset)} examples")

            # Show first example
            logger.info("First example:")
            logger.info(f"Text length: {len(custom_dataset[0]['text'])} characters")

        except Exception as e:
            logger.error(f"Failed to load custom corpus: {e}")

    logger.info("Data verification complete!")


def main():
    parser = argparse.ArgumentParser(description="Prepare and verify data")
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample data for testing"
    )
    parser.add_argument(
        "--verify",
        type=str,
        help="Verify data loading with config file"
    )

    args = parser.parse_args()

    if args.create_sample:
        logger.info("Creating sample data...")
        prepare_sample_data()

    if args.verify:
        logger.info(f"Verifying data with config: {args.verify}")
        verify_data(args.verify)

    if not args.create_sample and not args.verify:
        parser.print_help()


if __name__ == "__main__":
    main()
