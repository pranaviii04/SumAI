"""
SummAI Data Loader
Handles loading and preprocessing of summarization datasets
"""

import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional
from app.config import Config


class SummarizationDataset(Dataset):
    """Custom dataset for summarization tasks"""
    
    def __init__(
        self,
        texts: List[str],
        summaries: List[str],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = Config.MAX_INPUT_LENGTH,
        max_target_length: int = Config.MAX_TARGET_LENGTH
    ):
        """
        Initialize the dataset
        
        Args:
            texts: List of input texts
            summaries: List of reference summaries
            tokenizer: HuggingFace tokenizer
            max_input_length: Maximum length for input sequences
            max_target_length: Maximum length for target sequences
        """
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx) -> Dict:
        """
        Get a single preprocessed example
        
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        text = self.texts[idx]
        summary = self.summaries[idx]
        
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target summary
        targets = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }


def load_huggingface_dataset(
    dataset_name: str = Config.DATASET_NAME,
    dataset_config: Optional[str] = Config.DATASET_VERSION,
    max_train_samples: Optional[int] = Config.MAX_TRAIN_SAMPLES,
    max_eval_samples: Optional[int] = Config.MAX_EVAL_SAMPLES
):
    """
    Load a dataset from HuggingFace
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        dataset_config: Configuration/version of the dataset
        max_train_samples: Maximum number of training samples
        max_eval_samples: Maximum number of evaluation samples
    
    Returns:
        Tuple of (train_texts, train_summaries, eval_texts, eval_summaries)
    """
    print(f"[Download] Loading {dataset_name} dataset from HuggingFace...")
    
    # Load dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)
    
    # Extract text and summary fields based on dataset
    if dataset_name == "cnn_dailymail":
        text_field = "article"
        summary_field = "highlights"
    elif dataset_name == "xsum":
        text_field = "document"
        summary_field = "summary"
    elif dataset_name == "samsum":
        text_field = "dialogue"
        summary_field = "summary"
    else:
        # Try to auto-detect
        text_field = "text" if "text" in dataset["train"].features else "article"
        summary_field = "summary"
    
    # Extract training data
    train_texts = dataset["train"][text_field]
    train_summaries = dataset["train"][summary_field]
    
    # Extract validation data
    if "validation" in dataset:
        eval_texts = dataset["validation"][text_field]
        eval_summaries = dataset["validation"][summary_field]
    elif "test" in dataset:
        eval_texts = dataset["test"][text_field]
        eval_summaries = dataset["test"][summary_field]
    else:
        # Split train into train/eval
        split_idx = int(len(train_texts) * 0.8)
        eval_texts = train_texts[split_idx:]
        eval_summaries = train_summaries[split_idx:]
        train_texts = train_texts[:split_idx]
        train_summaries = train_summaries[:split_idx]
    
    # Limit samples if specified
    if max_train_samples:
        train_texts = train_texts[:max_train_samples]
        train_summaries = train_summaries[:max_train_samples]
    
    if max_eval_samples:
        eval_texts = eval_texts[:max_eval_samples]
        eval_summaries = eval_summaries[:max_eval_samples]
    
    print(f"[Success] Loaded {len(train_texts)} training samples and {len(eval_texts)} evaluation samples")
    
    return train_texts, train_summaries, eval_texts, eval_summaries


def load_csv_dataset(csv_path: str, text_column: str = "text", summary_column: str = "summary"):
    """
    Load a custom CSV dataset
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of the text column
        summary_column: Name of the summary column
    
    Returns:
        Tuple of (texts, summaries)
    """
    print(f"[Loading] Loading dataset from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    if text_column not in df.columns or summary_column not in df.columns:
        raise ValueError(f"CSV must contain '{text_column}' and '{summary_column}' columns")
    
    texts = df[text_column].tolist()
    summaries = df[summary_column].tolist()
    
    # Remove any NaN values
    valid_pairs = [(t, s) for t, s in zip(texts, summaries) 
                   if pd.notna(t) and pd.notna(s)]
    
    texts, summaries = zip(*valid_pairs) if valid_pairs else ([], [])
    
    print(f"[Success] Loaded {len(texts)} samples from CSV")
    
    return list(texts), list(summaries)


def prepare_datasets(
    tokenizer: PreTrainedTokenizer,
    use_huggingface: bool = True,
    csv_path: Optional[str] = None
):
    """
    Prepare training and evaluation datasets
    
    Args:
        tokenizer: HuggingFace tokenizer
        use_huggingface: Whether to use HuggingFace dataset
        csv_path: Path to custom CSV file (if use_huggingface=False)
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    if use_huggingface:
        train_texts, train_summaries, eval_texts, eval_summaries = load_huggingface_dataset()
    elif csv_path:
        texts, summaries = load_csv_dataset(csv_path)
        # Split into train/eval
        split_idx = int(len(texts) * (1 - Config.TRAIN_TEST_SPLIT))
        train_texts = texts[:split_idx]
        train_summaries = summaries[:split_idx]
        eval_texts = texts[split_idx:]
        eval_summaries = summaries[split_idx:]
    else:
        raise ValueError("Must specify either use_huggingface=True or provide csv_path")
    
    # Create datasets
    train_dataset = SummarizationDataset(
        train_texts, train_summaries, tokenizer
    )
    
    eval_dataset = SummarizationDataset(
        eval_texts, eval_summaries, tokenizer
    )
    
    return train_dataset, eval_dataset
