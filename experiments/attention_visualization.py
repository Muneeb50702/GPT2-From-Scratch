"""Run a forward pass and visualize a head attention pattern."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from config import DEFAULT_CONFIG
from model import GPT2FromScratch
from plotting import plot_attention


def main() -> None:
    cfg = DEFAULT_CONFIG
    model = GPT2FromScratch(cfg.model)
    x = torch.randint(0, cfg.model.vocab_size, (1, 32))
    out = model(x, return_cache=True, return_attn=True)
    attn = out.cache["blocks.0.attn_pattern"][0, 0]
    fig = plot_attention(attn, title="Layer0 Head0")
    fig.savefig("attention_visualization.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
