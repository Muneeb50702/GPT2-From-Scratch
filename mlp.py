"""Transformer MLP block."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, dropout: float = 0.0, use_bias: bool = True) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_mlp, bias=use_bias)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(d_mlp, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.act(self.fc_in(x))))
