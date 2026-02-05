"""Mini grokking-style setup: modular addition classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class ModAddMLP(nn.Module):
    def __init__(self, p: int = 97, d: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(p, d)
        self.net = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, p))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.emb(a), self.emb(b)], dim=-1)
        return self.net(x)


def main() -> None:
    p = 97
    model = ModAddMLP(p=p)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for step in range(500):
        a = torch.randint(0, p, (256,))
        b = torch.randint(0, p, (256,))
        y = (a + b) % p
        logits = model(a, b)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("loss", float(loss.item()))


if __name__ == "__main__":
    main()
