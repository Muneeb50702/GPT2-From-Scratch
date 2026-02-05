"""Activation and path patching helpers."""

from __future__ import annotations

import torch


def patch_activation(
    corrupt_cache: dict[str, torch.Tensor],
    clean_cache: dict[str, torch.Tensor],
    name: str,
) -> dict[str, torch.Tensor]:
    patched = dict(corrupt_cache)
    patched[name] = clean_cache[name]
    return patched


def patch_path(
    corrupt_cache: dict[str, torch.Tensor],
    clean_cache: dict[str, torch.Tensor],
    names: list[str],
) -> dict[str, torch.Tensor]:
    patched = dict(corrupt_cache)
    for n in names:
        patched[n] = clean_cache[n]
    return patched
