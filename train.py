"""Training entrypoint for GPT2FromScratch on Tiny Shakespeare."""

from __future__ import annotations

from pathlib import Path

import torch

from config import DEFAULT_CONFIG
from data import build_dataset, get_batch
from metrics import perplexity
from model import GPT2FromScratch
from optim import build_adamw, cosine_with_warmup
from utils import choose_device, save_checkpoint, set_seed


def evaluate(model: GPT2FromScratch, tokens: torch.Tensor, batch_size: int, block_size: int, device: torch.device, steps: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(steps):
            x, y = get_batch(tokens, batch_size, block_size, device)
            out = model(x, y)
            losses.append(float(out.loss.item()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    cfg = DEFAULT_CONFIG
    set_seed(cfg.train.seed)
    device = choose_device(cfg.train.device)

    bundle = build_dataset(cfg.data)
    cfg.model.vocab_size = bundle.tokenizer.vocab_size

    model = GPT2FromScratch(cfg.model).to(device)
    optimizer = build_adamw(model, cfg.train.learning_rate, cfg.train.weight_decay, cfg.train.betas)

    print(f"device={device} vocab={cfg.model.vocab_size} params={sum(p.numel() for p in model.parameters()):,}")

    for step in range(cfg.train.max_steps):
        lr = cosine_with_warmup(
            step=step,
            max_steps=cfg.train.max_steps,
            warmup_steps=cfg.train.warmup_steps,
            max_lr=cfg.train.learning_rate,
            min_lr=cfg.train.min_learning_rate,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(bundle.train_tokens, cfg.train.batch_size, cfg.data.block_size, device)
        out = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        if step % cfg.train.eval_interval == 0:
            val_loss = evaluate(model, bundle.val_tokens, cfg.train.batch_size, cfg.data.block_size, device, cfg.train.eval_steps)
            print(f"step={step:5d} train_loss={out.loss.item():.4f} val_loss={val_loss:.4f} val_ppl={perplexity(val_loss):.2f}")

        if step > 0 and step % cfg.logging.checkpoint_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "tokenizer": bundle.tokenizer.state_dict(),
                "step": step,
                "config": cfg,
            }
            save_checkpoint(cfg.logging.out_dir / f"{cfg.logging.run_name}_step{step}.pt", ckpt)

    final_path = cfg.logging.out_dir / f"{cfg.logging.run_name}_final.pt"
    save_checkpoint(
        final_path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tokenizer": bundle.tokenizer.state_dict(),
            "step": cfg.train.max_steps,
            "config": cfg,
        },
    )
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
