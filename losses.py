"""Loss helpers for language modeling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
