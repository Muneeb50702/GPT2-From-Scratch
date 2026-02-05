"""Global configuration objects for GPT-2-from-scratch training and analysis.

This module intentionally keeps configuration explicit and minimal so learners can see
exactly which hyperparameters control each behavior. We use dataclasses rather than a
heavier framework to preserve readability and make mechanistic experiments easy to edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset and tokenization configuration.

    Mechanistic note:
        Character-level tokenization produces a tiny and transparent vocabulary.
        This is useful for interpretability because every token corresponds to a
        directly understandable symbol (letter, punctuation, whitespace).
    """

    dataset_name: str = "tiny_shakespeare"
    dataset_url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt"
    )
    data_dir: Path = field(default_factory=lambda: Path("data"))
    text_file: str = "tiny_shakespeare.txt"
    train_split: float = 0.9
    block_size: int = 128


@dataclass
class ModelConfig:
    """Decoder-only transformer architecture hyperparameters."""

    vocab_size: int = 65  # overwritten after tokenizer fit
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_mlp: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    use_bias: bool = True
    tie_embeddings: bool = True


@dataclass
class TrainConfig:
    """Optimization and evaluation settings."""

    batch_size: int = 32
    max_steps: int = 5_000
    eval_interval: int = 200
    eval_steps: int = 50
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 200
    device: str = "cuda"
    seed: int = 42


@dataclass
class LoggingConfig:
    """File paths for checkpoints and logs."""

    out_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_every: int = 500
    run_name: str = "gpt2_from_scratch"


@dataclass
class Config:
    """Top-level configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


DEFAULT_CONFIG = Config()
