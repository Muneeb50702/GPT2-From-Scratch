"""Head-importance scaffolding using ablation on cached attention patterns."""

from __future__ import annotations

import torch

from head_ablation import zero_head


def head_importance_proxy(attn: torch.Tensor) -> list[float]:
    """A toy proxy: total attention mass removed by zeroing each head."""
    base = float(attn.sum().item())
    scores = []
    for h in range(attn.size(1)):
        ab = zero_head(attn, h)
        scores.append(base - float(ab.sum().item()))
    return scores


if __name__ == "__main__":
    attn = torch.rand(2, 8, 16, 16)
    print(head_importance_proxy(attn))
