"""Embedding and positional encoding modules for decoder-only transformers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding lookup table."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.weight(tokens)


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute position embedding."""

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.weight(positions)


class RotaryEmbedding(nn.Module):
    """RoPE helper that rotates Q/K features in paired dimensions."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def scaled_init_(module: nn.Module, d_model: int) -> None:
    """Initialization matching transformer variance heuristics."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()
