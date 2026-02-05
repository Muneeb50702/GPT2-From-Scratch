"""Text generation script using a trained checkpoint."""

from __future__ import annotations

import argparse

import torch

from model import GPT2FromScratch
from utils import choose_device, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="To be, or not to be")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = choose_device(args.device)
    ckpt = load_checkpoint(args.ckpt, map_location=str(device))
    cfg = ckpt["config"].model

    model = GPT2FromScratch(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    from tokenizer import CharTokenizer

    tok = CharTokenizer()
    tok.load_state_dict(ckpt["tokenizer"])

    x = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
