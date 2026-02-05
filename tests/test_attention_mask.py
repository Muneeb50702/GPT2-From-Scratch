import torch

from embeddings import causal_mask


def test_causal_mask_blocks_future():
    m = causal_mask(4, torch.device("cpu"))
    assert m[0, 1].item() is True
    assert m[2, 1].item() is False
