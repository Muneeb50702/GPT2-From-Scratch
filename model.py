"""GPT-2 style decoder-only transformer with optional activation cache returns."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from config import ModelConfig
from embeddings import LearnedPositionalEmbedding, TokenEmbedding, causal_mask, scaled_init_
from transformer_block import TransformerBlock
from hooks import HookManager


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    cache: dict[str, torch.Tensor]


class GPT2FromScratch(nn.Module):
    def __init__(self, cfg: ModelConfig, use_rope: bool = False, hook_manager: HookManager | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_rope = use_rope
        self.hooks = hook_manager or HookManager()

        self.tok_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = None if use_rope else LearnedPositionalEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_mlp=cfg.d_mlp,
                    dropout=cfg.dropout,
                    use_bias=cfg.use_bias,
                    use_rope=use_rope,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.unembed.weight = self.tok_emb.weight.weight

        self.apply(lambda m: scaled_init_(m, cfg.d_model))

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_cache: bool = False,
        return_attn: bool = False,
    ) -> ForwardOutput:
        bsz, seq_len = tokens.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError("Input sequence length exceeds max_seq_len")

        tok = self.tok_emb(tokens)
        if self.pos_emb is not None:
            pos = self.pos_emb(seq_len, tokens.device)[None, :, :]
            x = tok + pos
        else:
            x = tok
        x = self.drop(x)
        x = self.hooks.apply("resid_0", x)

        mask = causal_mask(seq_len, tokens.device)
        cache: dict[str, torch.Tensor] = {"tok_emb": tok, "resid_0": x}

        for i, block in enumerate(self.blocks):
            x, attn, block_cache = block(x, attn_mask=mask, return_attn=return_attn)
            x = self.hooks.apply(f"blocks.{i}.resid_post", x)
            if return_cache:
                for k, v in block_cache.items():
                    cache[f"blocks.{i}.{k}"] = v
                if attn is not None:
                    cache[f"blocks.{i}.attn_pattern"] = attn

        x = self.ln_f(x)
        x = self.hooks.apply("ln_f", x)
        logits = self.unembed(x)
        logits = self.hooks.apply("logits", logits)
        if return_cache:
            cache["ln_f"] = x
            cache["logits"] = logits

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return ForwardOutput(logits=logits, loss=loss, cache=cache if return_cache else {})

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = tokens[:, -self.cfg.max_seq_len :]
            out = self(idx_cond)
            logits = out.logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
