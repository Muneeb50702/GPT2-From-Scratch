"""Single pre-norm decoder transformer block with residual stream tracking."""

from __future__ import annotations

import torch
import torch.nn as nn

from attention import MultiHeadSelfAttention
from mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        dropout: float = 0.0,
        use_bias: bool = True,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, use_bias, use_rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp, dropout, use_bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        resid_pre = x
        attn_out, attn = self.attn(self.ln1(x), attn_mask=attn_mask, return_attn=return_attn)
        x = x + attn_out
        resid_mid = x
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
        resid_post = x
        block_cache = {
            "resid_pre": resid_pre,
            "resid_mid": resid_mid,
            "resid_post": resid_post,
            "attn_out": attn_out,
            "mlp_out": mlp_out,
        }
        return x, attn, block_cache
