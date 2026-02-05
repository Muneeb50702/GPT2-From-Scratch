"""Attention head ablation utilities."""

from __future__ import annotations

import torch


def zero_head(attn_pattern: torch.Tensor, head_idx: int) -> torch.Tensor:
    """Zeroes one attention head pattern. shape=(batch, heads, q, k)."""
    ablated = attn_pattern.clone()
    ablated[:, head_idx] = 0.0
    return ablated


def mean_replace_head(attn_pattern: torch.Tensor, head_idx: int) -> torch.Tensor:
    replaced = attn_pattern.clone()
    mean_head = attn_pattern.mean(dim=1, keepdim=True)
    replaced[:, head_idx : head_idx + 1] = mean_head
    return replaced
