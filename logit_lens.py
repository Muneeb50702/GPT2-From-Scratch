"""Logit lens utilities."""

from __future__ import annotations

import torch


def residual_to_logits(residual: torch.Tensor, unembed_weight: torch.Tensor) -> torch.Tensor:
    """Project residual stream state directly into vocab logit space."""
    return residual @ unembed_weight.T


def top_tokens(logits: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    vals, idx = torch.topk(logits, k=k, dim=-1)
    return idx, vals
