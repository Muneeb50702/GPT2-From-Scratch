"""Dataset utilities for Tiny Shakespeare character-level language modeling.

This module is intentionally explicit about each data transformation:
text -> tokens -> train/val streams -> random contiguous batches.
This makes it easy to reason about how information reaches the model.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

from config import DataConfig
from tokenizer import CharTokenizer


@dataclass
class DatasetBundle:
    """Container for tokenized train/val streams and tokenizer."""

    train_tokens: torch.Tensor
    val_tokens: torch.Tensor
    tokenizer: CharTokenizer


def ensure_tiny_shakespeare(data_cfg: DataConfig) -> Path:
    """Ensure Tiny Shakespeare is present locally; download if missing.

    Returns:
        Path to local text file.
    """
    data_cfg.data_dir.mkdir(parents=True, exist_ok=True)
    text_path = data_cfg.data_dir / data_cfg.text_file

    if not text_path.exists():
        urllib.request.urlretrieve(data_cfg.dataset_url, text_path)

    return text_path


def load_text(path: Path) -> str:
    """Load UTF-8 text from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_dataset(data_cfg: DataConfig) -> DatasetBundle:
    """Download (if needed), tokenize, and split Tiny Shakespeare.

    Mechanistic note:
        We keep contiguous token streams rather than shuffled examples.
        A language model is trained on next-token prediction over local context,
        so preserving order is critical for real sequence statistics.
    """
    text_path = ensure_tiny_shakespeare(data_cfg)
    text = load_text(text_path)

    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    all_tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_idx = int(len(all_tokens) * data_cfg.train_split)

    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    if len(train_tokens) <= data_cfg.block_size + 1:
        raise ValueError("Training split is too small for selected block_size.")
    if len(val_tokens) <= data_cfg.block_size + 1:
        raise ValueError("Validation split is too small for selected block_size.")

    return DatasetBundle(train_tokens=train_tokens, val_tokens=val_tokens, tokenizer=tokenizer)


def get_batch(
    token_stream: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous token windows for causal LM training.

    Returns:
        x: (batch, block_size) input tokens
        y: (batch, block_size) next-token targets
    """
    if token_stream.ndim != 1:
        raise ValueError("token_stream must be a 1D tensor of token ids.")

    max_start = len(token_stream) - block_size - 1
    if max_start <= 0:
        raise ValueError("token_stream too short for requested block_size.")

    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([token_stream[s : s + block_size] for s in starts])
    y = torch.stack([token_stream[s + 1 : s + block_size + 1] for s in starts])

    return x.to(device), y.to(device)
