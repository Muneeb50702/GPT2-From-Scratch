"""Causal multi-head self-attention built from raw matrix operations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from embeddings import RotaryEmbedding, apply_rope


class MultiHeadSelfAttention(nn.Module):
    """Mechanistically transparent MHA with optional RoPE and attention return."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_bias: bool = True,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope(seq_len, x.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.o_proj(out)
        return out, (attn if return_attn else None)
