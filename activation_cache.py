"""Activation cache container and helper routines."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ActivationCache:
    store: dict[str, torch.Tensor] = field(default_factory=dict)

    def add(self, name: str, tensor: torch.Tensor) -> None:
        self.store[name] = tensor.detach()

    def get(self, name: str) -> torch.Tensor:
        return self.store[name]

    def keys(self) -> list[str]:
        return list(self.store.keys())

    def to(self, device: torch.device | str) -> "ActivationCache":
        return ActivationCache({k: v.to(device) for k, v in self.store.items()})
