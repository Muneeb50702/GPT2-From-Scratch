"""Linear probing utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_probe(features: torch.Tensor, labels: torch.Tensor, num_classes: int, lr: float = 1e-2, steps: int = 300) -> LinearProbe:
    probe = LinearProbe(features.size(-1), num_classes).to(features.device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        logits = probe(features)
        loss = nn.functional.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return probe
