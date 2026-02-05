# GPT-2 From Scratch for Mechanistic Interpretability

A research-oriented, educational codebase for building a **decoder-only GPT-2 style model from first principles** and analyzing it with mechanistic interpretability tools.

This repository is designed for learners and researchers who want to:

- Implement transformer language models without hidden abstractions.
- Understand each component mechanistically (not just functionally).
- Run causal and attribution-based interpretability experiments.
- Progress from fundamentals to independent circuit-level research.

---

## Project Vision

This project follows a strict build-and-understand workflow:

1. **Implement** each transformer component from scratch in PyTorch.
2. **Instrument** the model to expose internal activations and pathways.
3. **Intervene** (patch/ablate/trace) to establish causal claims.
4. **Experiment** with known circuits (induction, copying, suppression, etc.).

The full curriculum and long-form roadmap live in [`COURSE_ROADMAP.md`](COURSE_ROADMAP.md).

---

## Current Repository Status

### ✅ Completed so far

- Added structured configuration system in `config.py` using dataclasses for:
  - data settings
  - model hyperparameters
  - training hyperparameters
  - logging/checkpoint paths
- Added deterministic character-level tokenizer in `tokenizer.py` with:
  - `fit`, `encode`, `decode`
  - `state_dict` / `load_state_dict` for checkpoint compatibility
- Added Tiny Shakespeare data pipeline in `data.py` with:
  - automatic dataset download (if missing)
  - UTF-8 loading
  - tokenization + train/validation split
  - contiguous random batch sampling for causal language modeling
- Added base directory placeholders for future work:
  - `experiments/`
  - `tests/`

### ⏳ Pending (next implementation stages)

The following planned modules are not implemented yet:

#### Core model components
- `embeddings.py`
- `attention.py`
- `mlp.py`
- `transformer_block.py`
- `model.py`

#### Training and inference
- `train.py`
- `generate.py`
- `losses.py`
- `optim.py`

#### Mechanistic interpretability tools
- `hooks.py`
- `activation_cache.py`
- `logit_lens.py`
- `patching.py`
- `attribution.py`
- `head_ablation.py`
- `probing.py`
- `residual_stream_analysis.py`

#### Experiment scripts
- `experiments/attention_visualization.py`
- `experiments/induction_head_detection.py`
- `experiments/head_importance.py`
- `experiments/feature_superposition_toy_model.py`
- `experiments/grokking_experiment.py`

#### Utilities
- `utils.py`
- `plotting.py`
- `metrics.py`

#### Tests
- minimal unit/runtime tests for:
  - tokenizer correctness
  - attention mask correctness
  - model forward pass
  - hook system functionality

---

## Repository Layout (in-progress)

```text
.
├── COURSE_ROADMAP.md
├── README.md
├── config.py
├── data.py
├── tokenizer.py
├── experiments/
└── tests/
```

As implementation proceeds, this will expand into model, training, interpretability, and visualization modules.

---

## Design Principles

- **PyTorch-first** (no prebuilt transformer model libraries).
- **Mechanistic clarity over abstraction depth**.
- **Explicit tensor operations and readable code paths**.
- **Hooks and activation access built in from the start**.
- **Reproducibility and controlled experiments**.

---

## Immediate Next Milestone

Build the first runnable GPT stack components:

1. Token + positional embeddings
2. Causal multi-head self-attention
3. MLP block
4. Transformer block assembly
5. End-to-end forward pass in `model.py`

Then attach training and interpretability infrastructure.

---

## Contribution / Collaboration Notes

This repository is being developed iteratively. If you want to contribute, open an issue with one of:

- implementation bug reports
- interpretability experiment ideas
- reproducibility improvements
- pedagogical clarity suggestions

---

## License

No license file is currently present in this repository.
If you want, we can add a standard open-source license (MIT/Apache-2.0) in the next update.
