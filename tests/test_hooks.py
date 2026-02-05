import torch

from hooks import HookManager


def test_hook_manager_modifies_tensor():
    hm = HookManager()

    def plus_one(x: torch.Tensor, _name: str):
        return x + 1

    hm.add_hook("x", plus_one)
    y = hm.apply("x", torch.tensor([1.0]))
    assert torch.allclose(y, torch.tensor([2.0]))
