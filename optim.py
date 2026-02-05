"""Optimizer and learning-rate schedule utilities."""

from __future__ import annotations

import math

import torch


def build_adamw(model: torch.nn.Module, lr: float, weight_decay: float, betas: tuple[float, float]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)


def cosine_with_warmup(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (max_lr - min_lr)
