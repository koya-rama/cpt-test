"""
Data loading and preprocessing for Continuous Pre-Training (CPT)
Supports HuggingFace datasets and custom corpus data
"""

import os
from typing import Optional, List, Dict, Union
from pathlib import Path
import logging

import torch
from datasets import load_dataset, concatenate_datasets, Dataset, IterableDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CPTDataLoader:
    """
    Data loader for Continuous Pre-Training that supports:
    - Nemotron pre-training datasets from HuggingFace
    - Custom corpus data (text files, jsonl, parquet)
    - Streaming for large datasets
    - Efficient tokenization with padding and chunking
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int = 4096,
        streaming: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the data loader

        Args:
            tokenizer_name: HuggingFace tokenizer name/path
            max_length: Maximum sequence length
            streaming: Whether to use streaming mode for large datasets
            cache_dir: Cache directory for datasets
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            use_fast=True
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.streaming = streaming
        self.cache_dir = cache_dir or "./data/cache"

        logger.info(f"Initialized tokenizer: {tokenizer_name}")
        logger.info(f"Max sequence length: {max_length}")
        logger.info(f"Streaming mode: {streaming}")

    def load_nemotron_datasets(
        self,
        dataset_names: List[str],
        split: str = "train",
        num_proc: int = 4,
    ) -> Union[Dataset, IterableDataset]:
        """
        Load Nemotron pre-training datasets from HuggingFace

        Available datasets:
        - nvidia/Nemotron-CC  # 6.3T tokens from Common Crawl
        - nvidia/Nemotron-CC-v2  # Extended version
        - nvidia/Nemotron-CC-Math-v1  # Math-focused dataset
        - nvidia/Nemotron-Pretraining-Code-v1  # Code dataset

        Args:
            dataset_names: List of dataset names to load
            split: Dataset split to load
            num_proc: Number of processes for data loading

        Returns:
            Combined dataset
        """
        datasets = []

        for dataset_name in dataset_names:
            logger.info(f"Loading dataset: {dataset_name}")
            try:
                ds = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=self.streaming,
                    cache_dir=self.cache_dir,
                    num_proc=None if self.streaming else num_proc,
                )
                datasets.append(ds)
                logger.info(f"Successfully loaded: {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
                continue

        if not datasets:
            raise ValueError("No datasets were successfully loaded")

        # Concatenate datasets
        if len(datasets) > 1:
            if self.streaming:
                from datasets import interleave_datasets
                combined = interleave_datasets(datasets)
            else:
                combined = concatenate_datasets(datasets)
        else:
            combined = datasets[0]

        return combined

    def load_custom_corpus(
        self,
        data_paths: List[str],
        text_column: str = "text",
    ) -> Dataset:
        """
        Load custom corpus data from local files

        Supports:
        - Plain text files (.txt)
        - JSONL files (.jsonl)
        - Parquet files (.parquet)
        - CSV files (.csv)

        Args:
            data_paths: List of file paths or directory paths
            text_column: Name of the column containing text

        Returns:
            Combined dataset
        """
        all_files = []

        for path in data_paths:
            path_obj = Path(path)
            if path_obj.is_dir():
                # Get all supported files in directory
                all_files.extend(path_obj.glob("**/*.txt"))
                all_files.extend(path_obj.glob("**/*.jsonl"))
                all_files.extend(path_obj.glob("**/*.parquet"))
                all_files.extend(path_obj.glob("**/*.csv"))
            else:
                all_files.append(path_obj)

        logger.info(f"Found {len(all_files)} files to load")

        datasets = []

        for file_path in all_files:
            file_path = str(file_path)
            logger.info(f"Loading: {file_path}")

            try:
                if file_path.endswith(".txt"):
                    # Load plain text files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    ds = Dataset.from_dict({text_column: [text]})

                elif file_path.endswith(".jsonl"):
                    ds = load_dataset("json", data_files=file_path, split="train")

                elif file_path.endswith(".parquet"):
                    ds = load_dataset("parquet", data_files=file_path, split="train")

                elif file_path.endswith(".csv"):
                    ds = load_dataset("csv", data_files=file_path, split="train")

                datasets.append(ds)

            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue

        if not datasets:
            raise ValueError("No custom corpus files were successfully loaded")

        # Concatenate all datasets
        combined = concatenate_datasets(datasets)
        logger.info(f"Loaded {len(combined)} examples from custom corpus")

        return combined

    def tokenize_function(self, examples: Dict, text_column: str = "text") -> Dict:
        """
        Tokenize examples with padding and truncation

        Args:
            examples: Batch of examples
            text_column: Name of the text column

        Returns:
            Tokenized examples
        """
        # Tokenize
        tokenized = self.tokenizer(
            examples[text_column],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )

        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        text_column: str = "text",
        num_proc: int = 4,
        remove_columns: Optional[List[str]] = None,
    ) -> Union[Dataset, IterableDataset]:
        """
        Prepare dataset by tokenizing and formatting

        Args:
            dataset: Raw dataset
            text_column: Name of the text column
            num_proc: Number of processes for mapping
            remove_columns: Columns to remove after tokenization

        Returns:
            Prepared dataset ready for training
        """
        logger.info("Tokenizing dataset...")

        # Get columns to remove
        if remove_columns is None:
            if hasattr(dataset, 'column_names'):
                remove_columns = dataset.column_names
            else:
                remove_columns = []

        # For non-streaming datasets, ensure we remove all original columns
        # This prevents the text column from being passed to the model
        tokenized = dataset.map(
            lambda x: self.tokenize_function(x, text_column),
            batched=True,
            num_proc=None if self.streaming else num_proc,
            remove_columns=remove_columns,  # Always remove columns for non-streaming
        )

        logger.info("Dataset tokenization complete")

        return tokenized

    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        """
        Create PyTorch DataLoader from dataset

        Args:
            dataset: Prepared dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes

        Returns:
            DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


def get_nemotron_data_loader(
    tokenizer_name: str,
    dataset_names: Optional[List[str]] = None,
    custom_corpus_paths: Optional[List[str]] = None,
    max_length: int = 4096,
    streaming: bool = True,
    text_column: str = "text",
    batch_size: int = 4,
    num_proc: int = 4,
) -> Union[Dataset, IterableDataset]:
    """
    Convenience function to get a complete data loader

    Args:
        tokenizer_name: HuggingFace tokenizer name
        dataset_names: List of HuggingFace dataset names
        custom_corpus_paths: List of custom corpus file/directory paths
        max_length: Maximum sequence length
        streaming: Whether to use streaming mode
        text_column: Name of the text column
        batch_size: Batch size for training
        num_proc: Number of processes

    Returns:
        Prepared dataset
    """
    loader = CPTDataLoader(
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        streaming=streaming,
    )

    datasets_to_combine = []

    # Load HuggingFace datasets
    if dataset_names:
        hf_dataset = loader.load_nemotron_datasets(
            dataset_names=dataset_names,
            num_proc=num_proc,
        )
        datasets_to_combine.append(hf_dataset)

    # Load custom corpus
    if custom_corpus_paths:
        custom_dataset = loader.load_custom_corpus(
            data_paths=custom_corpus_paths,
            text_column=text_column,
        )
        datasets_to_combine.append(custom_dataset)

    # Combine all datasets
    if len(datasets_to_combine) > 1:
        if streaming:
            from datasets import interleave_datasets
            combined = interleave_datasets(datasets_to_combine)
        else:
            combined = concatenate_datasets(datasets_to_combine)
    elif len(datasets_to_combine) == 1:
        combined = datasets_to_combine[0]
    else:
        raise ValueError("No datasets specified")

    # Tokenize and prepare
    prepared = loader.prepare_dataset(
        dataset=combined,
        text_column=text_column,
        num_proc=num_proc,
    )

    return prepared
