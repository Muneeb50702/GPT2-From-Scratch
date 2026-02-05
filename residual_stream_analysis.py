"""Residual stream analysis helpers."""

from __future__ import annotations

import torch


def component_norms(cache: dict[str, torch.Tensor], prefix: str = "blocks") -> dict[str, float]:
    norms = {}
    for k, v in cache.items():
        if k.startswith(prefix):
            norms[k] = float(v.norm().item())
    return norms


def residual_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)
