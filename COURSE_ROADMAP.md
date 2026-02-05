# Mechanistic Interpretability + GPT-2 From Scratch — Full Research Course Roadmap

This roadmap is designed as a structured apprenticeship from first principles to independent mechanistic interpretability research. It emphasizes **implementation clarity**, **causal/mechanistic reasoning**, and **experiment-first understanding**.

---

## Course Philosophy

- Build every core GPT-2 component from scratch in PyTorch before relying on high-level libraries.
- Treat each architectural block as a hypothesis about computation and information routing.
- Constantly connect:
  1. **Math** (formal objective and decomposition)
  2. **Code** (explicit tensors and operations)
  3. **Mechanism** (what circuit is implementing what behavior)
  4. **Experiment** (causal interventions and diagnostics)
- Develop researcher habits: ablations, controls, reproducibility, and critical interpretation.

---

## Phase 0 — Mathematical & Conceptual Foundations

### Goal
Establish the minimum mathematical and conceptual language needed for deep mechanistic reasoning.

### Modules

1. **Vectors, matrices, tensors, and basis changes**
   - Learning objectives:
     - Understand linear maps and coordinate systems.
     - Interpret features as directions/subspaces, not just dimensions.
     - Build intuition for projection, decomposition, and orthogonality.

2. **Optimization and gradient flow**
   - Learning objectives:
     - Derive gradient descent and backprop from chain rule.
     - Understand loss landscapes, curvature, and optimization path dependence.

3. **Probability for language modeling**
   - Learning objectives:
     - Understand maximum likelihood and cross-entropy.
     - Connect logits → softmax → token probabilities.

4. **Information theory essentials**
   - Learning objectives:
     - Entropy, KL divergence, surprisal.
     - Interpret model confidence and calibration behavior.

5. **Mechanistic interpretability framing**
   - Learning objectives:
     - Distinguish behavioral vs mechanistic interpretability.
     - Define circuits, features, pathways, and causal interventions.

### Deliverables
- Notebook: linear algebra visualizations (projections, subspaces, superposition toy examples).
- Mini exercises with hand-derived gradients.

---

## Phase 1 — Neural Network From Scratch

### Goal
Build core NN machinery manually and understand representations before transformers.

### Modules

1. **Single-layer networks and nonlinearities**
   - Objectives:
     - Implement affine maps + activation functions from raw ops.
     - Analyze expressivity of piecewise linear networks.

2. **Backprop by hand + PyTorch autograd comparison**
   - Objectives:
     - Manually compute gradients for toy MLP.
     - Verify against autograd numerically.

3. **MLP internals and feature geometry**
   - Objectives:
     - Understand neurons as feature detectors.
     - Explore polysemantic neurons in toy settings.

4. **Regularization and optimization dynamics**
   - Objectives:
     - Examine overfitting, weight decay, dropout, and early stopping.
     - Track training dynamics with detailed logging.

### Deliverables
- From-scratch MLP library (`Linear`, `ReLU/GELU`, `Sequential`, `Trainer`).
- Unit tests for forward shapes, gradients, and loss decrease.
- Toy feature interpretability lab (2D synthetic datasets).

---

## Phase 2 — Attention Mechanisms Deep Dive

### Goal
Master attention mathematically and mechanistically before full transformer assembly.

### Modules

1. **Scaled dot-product attention from first principles**
   - Objectives:
     - Derive attention as content-based routing.
     - Explain why scaling by sqrt(d_k) is needed.

2. **Causal masking and autoregressive constraints**
   - Objectives:
     - Implement triangular masks and verify no future leakage.

3. **Single-head attention implementation**
   - Objectives:
     - Build Q/K/V projections with explicit tensor algebra.
     - Inspect attention score matrices and output composition.

4. **Multi-head attention as subspace factorization**
   - Objectives:
     - Understand head specialization via learned subspaces.
     - Analyze head redundancy and complementarity.

5. **Positional information pathways**
   - Objectives:
     - Compare learned absolute positions, sinusoidal positions, and rotary embeddings.

### Deliverables
- Clean attention module from raw matmul.
- Visualizer for attention scores and entropy per head.
- Experiments showing effects of mask and positional choices.

---

## Phase 3 — Transformer Architecture Fundamentals

### Goal
Assemble a pedagogical transformer block and reason about residual stream communication.

### Modules

1. **Residual stream as communication channel**
   - Objectives:
     - Treat each sublayer as read/write operation on shared state.
     - Understand additive composition and interference.

2. **LayerNorm mechanics (pre-norm vs post-norm)**
   - Objectives:
     - Implement both variants.
     - Compare stability and gradient behavior.

3. **MLP block internals and gating behavior**
   - Objectives:
     - Implement 2-layer feedforward + GELU.
     - Study feature extraction and recombination roles.

4. **End-to-end transformer block**
   - Objectives:
     - Compose Attention + MLP + residual pathways.
     - Validate causal generation and tensor invariants.

### Deliverables
- `TransformerBlock` from scratch.
- Diagnostic hooks for residual stream snapshots.

---

## Phase 4 — GPT-2 Full Implementation From Scratch

### Goal
Implement a GPT-2 style language model end-to-end with transparent components.

### Modules

1. **Tokenization pipeline**
   - Objectives:
     - Understand BPE principles and vocabulary construction.
     - Implement minimal tokenizer pipeline (or train/use small BPE).

2. **Embedding + unembedding mechanics**
   - Objectives:
     - Implement token embedding matrix and output projection.
     - Explore weight tying and its implications.

3. **Positional encoding comparison lab**
   - Objectives:
     - Implement absolute learned embeddings and RoPE.
     - Evaluate extrapolation/generalization effects.

4. **Model assembly and generation loop**
   - Objectives:
     - Build full decoder-only transformer stack.
     - Implement greedy, temperature, top-k, top-p sampling.

5. **Training script and checkpoints**
   - Objectives:
     - Implement robust training loop, logging, evaluation, checkpointing.

### Deliverables
- Full GPT-2 mini implementation (configurable depth/width).
- Reproducible training script on a small corpus.
- Unit tests for masking correctness, shape checks, and determinism.

---

## Phase 5 — Training Dynamics & Scaling Behavior

### Goal
Develop intuition for how circuits emerge during training and how to measure that.

### Modules

1. **Loss curves, token-level diagnostics, and failure modes**
   - Objectives:
     - Track per-token loss distributions.
     - Diagnose underfitting vs memorization.

2. **Emergence of attention patterns over time**
   - Objectives:
     - Snapshot checkpoints and inspect head evolution.

3. **Grokking and phase transitions in toy tasks**
   - Objectives:
     - Reproduce grokking-like behavior.
     - Relate optimization dynamics to circuit formation.

4. **Scaling knobs and compute tradeoffs**
   - Objectives:
     - Explore depth/width/context/batch effects.
     - Build principled experimentation habits.

### Deliverables
- Training dynamics dashboard notebook.
- Checkpoint comparison utilities.

---

## Phase 6 — Mechanistic Interpretability Core Methods

### Goal
Build and use foundational mech-interp tools akin to early TransformerLens workflows.

### Modules

1. **Activation caching and hook system**
   - Objectives:
     - Implement forward hooks for residual, attention, MLP activations.
     - Build reusable cache object.

2. **Logit lens and tuned lens style analyses**
   - Objectives:
     - Project intermediate residual states to vocabulary logits.
     - Interpret evolving token predictions by layer.

3. **Direct logit attribution (DLA)**
   - Objectives:
     - Decompose final logits into component contributions.

4. **Activation patching (causal tracing)**
   - Objectives:
     - Implement clean/corrupt run framework.
     - Patch specific activations and measure restoration effects.

5. **Path patching and edge-level causal analysis**
   - Objectives:
     - Move from node-level to path-level causality.
     - Identify minimal important computational pathways.

6. **Head and neuron ablations**
   - Objectives:
     - Zero/mean-replace/freeze activations and quantify behavior changes.

### Deliverables
- Core interpretability toolkit (`hooks`, `cache`, `patching`, `attribution`).
- Reproducible notebooks for each method.

---

## Phase 7 — Transformer Circuits Research Canon

### Goal
Understand and reproduce canonical circuit discoveries.

### Modules

1. **Induction heads**
   - Objectives:
     - Derive algorithmic role in in-context sequence continuation.
     - Detect induction pattern metrics empirically.

2. **Name mover and copy-related heads**
   - Objectives:
     - Identify heads that route entity information to output positions.

3. **Copy suppression / anti-copy behavior**
   - Objectives:
     - Analyze heads reducing naive copying and improving calibration.

4. **Composition of heads and MLP mediation**
   - Objectives:
     - Examine multi-step circuits across layers.

### Deliverables
- Circuit case studies with intervention evidence.
- Head taxonomy report for trained model.

---

## Phase 8 — Representation Theory, Superposition, and Features

### Goal
Develop a modern representation-centric understanding of learned features.

### Modules

1. **Superposition theory in toy models**
   - Objectives:
     - Reproduce sparse feature packing phenomena.
     - Understand interference and basis mismatch.

2. **Polysemantic neurons and sparse autoencoders (SAEs)**
   - Objectives:
     - Train simple SAEs on activations.
     - Recover interpretable feature dictionaries.

3. **Linear representation hypothesis and probes**
   - Objectives:
     - Design linear probes for syntactic/semantic properties.
     - Critically assess probe interpretability limitations.

4. **Weight-space decomposition**
   - Objectives:
     - Factorize attention/MLP weights to infer feature circuits.

### Deliverables
- Toy superposition experiments.
- SAE exploratory notebooks.
- Probe evaluation suite.

---

## Phase 9 — Advanced Topics & Open Problems

### Goal
Reach frontier-level awareness of unsolved questions and competing paradigms.

### Modules

1. **Mechanistic faithfulness and causal validity**
   - Objectives:
     - Distinguish explanatory stories from causal evidence.

2. **Scaling mech-interp to larger models**
   - Objectives:
     - Address tool scalability, memory bottlenecks, and hypothesis search.

3. **Automated circuit discovery and attribution methods**
   - Objectives:
     - Survey algorithmic discovery techniques and limitations.

4. **Cross-model transfer of features/circuits**
   - Objectives:
     - Explore whether circuits align across checkpoints/architectures.

5. **Safety relevance and alignment interfaces**
   - Objectives:
     - Connect mechanistic insights to robustness, deception, and control.

### Deliverables
- Research memo mapping open problems, hypotheses, and experiment plans.

---

## Phase 10 — Reproducing Papers + Independent Research Project

### Goal
Transition from guided learner to independent researcher.

### Modules

1. **Paper reproduction sprint sequence**
   - Objectives:
     - Reproduce 2–4 classic results with rigorous controls.

2. **Research question generation workshop**
   - Objectives:
     - Convert observations into falsifiable hypotheses.

3. **Experiment design and evaluation standards**
   - Objectives:
     - Predefine metrics, baselines, ablations, and uncertainty checks.

4. **Write-up and communication**
   - Objectives:
     - Produce publication-style report and replication package.

### Deliverables
- Capstone project proposal.
- End-to-end experimental report.
- Interp tool extension (mini-TransformerLens-like module).

---

## Cross-Phase Infrastructure (Built Progressively)

Across phases, we will incrementally build a single coherent codebase:

- `model/`
  - `tokenizer.py`
  - `embeddings.py`
  - `attention.py`
  - `mlp.py`
  - `layernorm.py`
  - `block.py`
  - `gpt2.py`
- `train/`
  - `dataset.py`
  - `trainer.py`
  - `eval.py`
- `interp/`
  - `hooks.py`
  - `cache.py`
  - `logit_lens.py`
  - `attribution.py`
  - `patching.py`
  - `circuits.py`
  - `probes.py`
- `viz/`
  - attention heatmaps, residual contribution plots, trajectory plots
- `tests/`
  - shape tests, masking tests, gradient tests, causal sanity tests
- `notebooks/`
  - per-phase exploratory labs

---

## Core Experiment Track (Threaded Through Entire Course)

These recurring experiments will be repeated with increasing sophistication:

1. **Attention pattern visualization** (what gets attended and when)
2. **Head ablation sweeps** (which heads matter for which tasks)
3. **Activation patching** (which components causally restore behavior)
4. **Logit attribution decomposition** (who writes to target logits)
5. **Information flow tracing** (path-level influence)
6. **Feature dictionary extraction** (superposition/polysemantic analysis)
7. **Training-time circuit emergence** (when mechanisms form)

---

## Historical & Conceptual Integration Plan

Interleaved short seminars will cover:

- Historical shift: saliency/feature visualization → circuit-level causal analysis.
- Anthropic circuits perspective vs broader representation-engineering approaches.
- TransformerLens-style practical workflows and their assumptions.
- Common misconceptions:
  - “Ablation importance equals mechanism.”
  - “Attention weights alone explain behavior.”
  - “Probe accuracy implies encoded causal feature.”

---

## Milestones and Mastery Checks

1. **Milestone A (after Phase 3):** Build and validate a single transformer block from scratch.
2. **Milestone B (after Phase 4):** Train and sample from a working mini GPT-2.
3. **Milestone C (after Phase 6):** Run causal tracing and logit attribution on custom prompts.
4. **Milestone D (after Phase 8):** Reproduce superposition/polysemantic toy findings.
5. **Milestone E (after Phase 10):** Complete an independent mech-interp project with write-up.

---

## Suggested Initial Reading Sequence (Used Throughout)

- Transformer original paper (for architecture baseline).
- GPT-2 paper (for autoregressive scaling and setup).
- Early interpretability/circuits materials.
- Induction head and in-context learning analyses.
- Superposition and sparse feature work.
- Recent causal/path patching and automated discovery papers.

(Exact paper list and reading annotations will be attached in the corresponding phase lessons.)

---

## What Happens Next

When you say **“Start Lesson 1”**, we begin **Phase 0, Module 1** with the required structure:

- Intuition
- Mathematical Formulation
- Implementation Walkthrough
- Mechanistic Interpretation Perspective
- Experiments
- Exercises
- Research Notes
- Summary
- Next Lesson Preview

And we will stop after that lesson for your responses before continuing.
