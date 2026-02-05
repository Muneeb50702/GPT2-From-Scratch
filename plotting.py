"""Plotting helpers for attention and training diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def plot_attention(attn: torch.Tensor, title: str = "Attention Pattern"):
    """attn shape: (seq, seq) or (heads, seq, seq) -> plots first head if 3D."""
    if attn.dim() == 3:
        attn = attn[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn.detach().cpu().numpy(), aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.colorbar(im, ax=ax)
    return fig


def plot_training_curve(steps: list[int], losses: list[float], title: str = "Training Loss"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    return fig
