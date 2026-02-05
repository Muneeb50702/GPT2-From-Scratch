# GPT-2 From Scratch for Mechanistic Interpretability

Educational + research-friendly implementation of a decoder-only GPT-2 style transformer in pure PyTorch, with built-in mechanistic interpretability tooling and experiment scaffolding.

## What is implemented

### Core model
- `config.py`: structured dataclass configs.
- `tokenizer.py`: deterministic character-level tokenizer.
- `data.py`: Tiny Shakespeare download/load/split/batching.
- `embeddings.py`: token embeddings, learned positional embeddings, RoPE helpers, causal mask.
- `attention.py`: raw causal multi-head self-attention with optional RoPE.
- `mlp.py`: transformer feed-forward block.
- `transformer_block.py`: pre-norm residual attention+MLP block with cache outputs.
- `model.py`: full GPT-2 style decoder stack with generation and cache support.

### Training + inference
- `losses.py`: LM loss helpers.
- `optim.py`: AdamW setup + cosine warmup schedule.
- `train.py`: full training loop (eval, clipping, checkpointing).
- `generate.py`: checkpoint-based text generation.

### Mechanistic interpretability toolkit
- `hooks.py`: hook manager and temporary hook context.
- `activation_cache.py`: activation container.
- `logit_lens.py`: residual-to-logit projection helpers.
- `patching.py`: activation/path patch helpers.
- `attribution.py`: grad×input and direct logit attribution helpers.
- `head_ablation.py`: head ablation operators.
- `probing.py`: trainable linear probe utilities.
- `residual_stream_analysis.py`: residual norms/cosine analysis.

### Experiments
- `experiments/attention_visualization.py`
- `experiments/induction_head_detection.py`
- `experiments/head_importance.py`
- `experiments/feature_superposition_toy_model.py`
- `experiments/grokking_experiment.py`

### Tests
- `tests/test_tokenizer.py`
- `tests/test_attention_mask.py`
- `tests/test_model_forward.py`
- `tests/test_hooks.py`
- `tests/conftest.py` (ensures local imports work)

## Quickstart

1. Install dependencies (recommended):
```bash
pip install torch matplotlib pytest numpy
```

2. Train:
```bash
python train.py
```

3. Generate text:
```bash
python generate.py --ckpt outputs/gpt2_from_scratch_final.pt --prompt "To be, or not to be" --max_new_tokens 200
```

4. Run tests:
```bash
pytest -q
```

5. Run example experiments:
```bash
python experiments/attention_visualization.py
python experiments/induction_head_detection.py
python experiments/head_importance.py
python experiments/feature_superposition_toy_model.py
python experiments/grokking_experiment.py
```

## Notes
- This repository avoids prebuilt transformer model libraries for transparency.
- Activation caching + hooks are first-class to support circuit-level analysis workflows.
- The pedagogical roadmap remains in `COURSE_ROADMAP.md`.
