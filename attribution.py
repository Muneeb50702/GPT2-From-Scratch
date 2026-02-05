"""Simple attribution and direct-logit-contribution utilities."""

from __future__ import annotations

import torch


def grad_x_input(logits: torch.Tensor, inputs: torch.Tensor, target_idx: int) -> torch.Tensor:
    """Compute grad*input attribution for a chosen logit index."""
    scalar = logits[..., target_idx].sum()
    grad = torch.autograd.grad(scalar, inputs, retain_graph=True, allow_unused=True)[0]
    if grad is None:
        raise RuntimeError("Gradient is None; ensure inputs require gradients.")
    return grad * inputs


def direct_logit_attribution(component: torch.Tensor, unembed_vec: torch.Tensor) -> torch.Tensor:
    """Contribution of a residual component to one token logit.

    component shape: (..., d_model)
    unembed_vec shape: (d_model,)
    """
    return (component * unembed_vec).sum(dim=-1)
