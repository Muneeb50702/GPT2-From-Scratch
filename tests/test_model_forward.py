import torch

from config import ModelConfig
from model import GPT2FromScratch


def test_forward_shapes():
    cfg = ModelConfig(vocab_size=20, d_model=32, n_heads=4, n_layers=2, d_mlp=64, max_seq_len=16)
    model = GPT2FromScratch(cfg)
    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))
    out = model(x, y, return_cache=True, return_attn=True)
    assert out.logits.shape == (2, 8, 20)
    assert out.loss is not None
    assert "blocks.0.attn_pattern" in out.cache
