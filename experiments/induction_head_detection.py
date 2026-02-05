"""Simple heuristic induction head score on repeated token patterns."""

from __future__ import annotations

import torch


def induction_score(attn: torch.Tensor) -> float:
    """attn shape (heads, seq, seq). Heuristic: superdiagonal offset copy attention."""
    heads, seq, _ = attn.shape
    scores = []
    for h in range(heads):
        vals = []
        for q in range(2, seq):
            k = q - 1
            vals.append(float(attn[h, q, k].item()))
        scores.append(sum(vals) / max(1, len(vals)))
    return max(scores)


if __name__ == "__main__":
    a = torch.rand(8, 32, 32)
    a = a / a.sum(dim=-1, keepdim=True)
    print("induction_score", induction_score(a))
