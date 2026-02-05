"""Hook registration utilities for interpretability interventions."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Callable

import torch

HookFn = Callable[[torch.Tensor, str], torch.Tensor | None]


class HookManager:
    """Name-based forward hook manager used across interpretability tools."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookFn]] = defaultdict(list)

    def add_hook(self, name: str, fn: HookFn) -> None:
        self._hooks[name].append(fn)

    def clear(self) -> None:
        self._hooks.clear()

    def apply(self, name: str, x: torch.Tensor) -> torch.Tensor:
        for fn in self._hooks.get(name, []):
            out = fn(x, name)
            if out is not None:
                x = out
        return x

    @contextmanager
    def temporary_hook(self, name: str, fn: HookFn):
        self.add_hook(name, fn)
        try:
            yield
        finally:
            self._hooks[name].remove(fn)
