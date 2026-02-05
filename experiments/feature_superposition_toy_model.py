"""Toy superposition experiment with sparse features and a bottleneck."""

from __future__ import annotations

import torch
import torch.nn as nn


class ToySuperposition(nn.Module):
    def __init__(self, n_features: int = 64, hidden: int = 16) -> None:
        super().__init__()
        self.enc = nn.Linear(n_features, hidden, bias=False)
        self.dec = nn.Linear(hidden, n_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.relu(self.enc(x)))


def main() -> None:
    model = ToySuperposition()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(300):
        x = (torch.rand(128, 64) < 0.05).float() * torch.randn(128, 64)
        recon = model(x)
        loss = ((x - recon) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("final_loss", float(loss.item()))


if __name__ == "__main__":
    main()
