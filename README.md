# Cognitive Tree Search (CTS)

**KV-Cache-Free Per-Node Active O(1) Transitions for System 2 Inference via Deep Equilibrium Models**

*Anonymous — NeurIPS 2026 Submission*

<p align="center">
  <img src="assets/cts_architecture.png" alt="CTS Architecture" width="720">
</p>

---

## Overview

CTS replaces explicit autoregressive KV-cache sequences with **implicit fixed-point transitions** via Deep Equilibrium Models (DEQ), enabling deep Monte Carlo Tree Search on a **single 24 GB GPU** with a **constant &le; 16.7 GB active VRAM footprint** — where Vanilla MCTS triggers OOM at depth 15.

**Core contributions:**
1. **KV-Cache-Free DEQ Transitions** &mdash; Each MCTS node transition solves a fixed-point equation `z* = f(z*)` using L-Broyden, requiring only O(1) active memory per node (&sect;4.2).
2. **Adaptive Control Operators** &mdash; A learned meta-policy &pi;<sub>&phi;</sub> outputs &nu; = [&nu;<sub>expl</sub>, &nu;<sub>tol</sub>, &nu;<sub>temp</sub>, &nu;<sub>act</sub>] &in; &Ropf;<sup>4</sup> to modulate exploration, solver precision, routing temperature, and halting (&sect;4.1).
3. **Latent Context Window** &mdash; FAISS-IVF-PQ retrieves ancestral fixed-point vectors as soft-prompt prefixes, recovering 95% of full KV-cache context quality (&sect;4.3).
4. **Hybrid KV-Assisted Mode** &mdash; Shallow nodes (D &le; 5) optionally cache KV-states in spare VRAM headroom for a 21% wall-clock speedup with no accuracy change (&sect;7.7).

---

## Key Results

### Table 2 &mdash; Budget-Capped Performance (&le; 10<sup>14</sup> MACs, Gemma 4 E4B, 5 seeds, 95% bootstrap CI)

| Method | MATH-500 | GSM8K | AIME 2024 | ARC-AGI-Text | HumanEval | MACs (&times;10<sup>14</sup>) |
|--------|:--------:|:-----:|:---------:|:------------:|:---------:|:---:|
| Greedy | 45.2 | 76.5 | 28.3 | 36.1 | 56.4 | 0.05 |
| SC@14 | 59.3&pm;0.7 | 84.2&pm;0.5 | 34.8&pm;0.9 | 52.4&pm;0.8 | 65.2&pm;0.6 | 1.0 |
| Native Think | 57.0&pm;0.6 | 82.4&pm;0.4 | 42.5&pm;0.9 | 50.1&pm;0.7 | 63.3&pm;0.5 | &asymp;0.80 |
| MCTS (Early Stop) | 56.5&pm;0.9 | 81.2&pm;0.7 | 38.4&pm;0.8 | 48.1&pm;1.0 | 62.5&pm;0.7 | 0.8 |
| **CTS-4&nu; (Ours)** | **63.8&pm;0.8** | **88.4&pm;0.5** | **50.2&pm;1.1** | **57.8&pm;0.9** | **69.6&pm;0.7** | **0.65** |

### Table 1 &mdash; Active VRAM During Search Phase (W = 3)

| Method | Depth 1 | Depth 15 | Depth 35 | Depth 100+ |
|--------|:-------:|:--------:|:--------:|:----------:|
| MCTS (Vanilla) | 16.5 GB | OOM | &mdash; | &mdash; |
| MCTS (Prefix Cache) | 16.5 GB | 18.2 GB | OOM | &mdash; |
| **CTS (Ours)** | **16.5 GB** | **16.6 GB** | **16.6 GB** | **16.7 GB** |

---

## Reproducibility

This repository implements every component described in the paper. The mapping between paper sections and source files is documented below to facilitate reviewer verification.

### Paper &harr; Code Mapping

| Paper Section | Algorithm / Equation | Source File | Key Function / Class |
|:---|:---|:---|:---|
| Algorithm 1 | CTS Full Episode Loop | [`cts/mcts/cts_episode.py`](cts/mcts/cts_episode.py) | `cts_full_episode()` |
| &sect;4.1 | Meta-Policy &pi;<sub>&phi;</sub> (&nu; &in; &Ropf;<sup>4</sup>) | [`cts/policy/meta_policy.py`](cts/policy/meta_policy.py) | `MetaPolicy` |
| &sect;4.1 | Neuro-Critic V<sub>&psi;</sub> | [`cts/critic/neuro_critic.py`](cts/critic/neuro_critic.py) | `NeuroCritic` |
| &sect;4.2 Eq.&thinsp;2 | PUCT Selection | [`cts/mcts/puct.py`](cts/mcts/puct.py) | `puct_score()` |
| &sect;4.2 | DEQ Transition (KV-free) | [`cts/deq/transition.py`](cts/deq/transition.py) | `transition()`, `transition_batch()` |
| &sect;4.3 | W<sub>proj</sub> Soft-Prompt Decoding | [`cts/backbone/gemma_adapter.py`](cts/backbone/gemma_adapter.py) | `decode_from_z_star()` |
| &sect;4.3 | FAISS-IVF-PQ Latent Context | [`cts/latent/faiss_context.py`](cts/latent/faiss_context.py) | `LatentContextWindow` |
| &sect;5.2 | L-Broyden FP32 Solver (rank 16) | [`cts/deq/broyden_forward.py`](cts/deq/broyden_forward.py) | `broyden_fixed_point()` |
| &sect;5.2 | Jacobian Inheritance (Remark 2) | [`cts/deq/broyden_forward.py`](cts/deq/broyden_forward.py) | `BroydenInfo.jacobian_state` |
| &sect;5.3 Eq.&thinsp;3 | Sparse Top-k Routing | [`cts/routing/sparse_moe_ref.py`](cts/routing/sparse_moe_ref.py) | `routing_weights()` |
| &sect;5.3 | Triton Fused Kernel | [`cts/routing/sparse_moe_triton.py`](cts/routing/sparse_moe_triton.py) | `routing_weights_triton()` |
| &sect;6 | Stage 1: DEQ Warm-up (IFT + 0.1&middot;L<sub>CE</sub>) | [`cts/train/stage1_warmup.py`](cts/train/stage1_warmup.py) | `fixed_point_surrogate_loss()` |
| &sect;6 | Stage 2: PPO + GAE | [`cts/train/stage2_ppo_train.py`](cts/train/stage2_ppo_train.py) | `PPOTrainer` |
| &sect;7.7 | Hybrid KV-Assisted Mode | [`cts/mcts/hybrid_kv.py`](cts/mcts/hybrid_kv.py) | `HybridKVManager` |
| Table 5 | CTS-2&nu;/4&nu; Pareto Configs | [`cts/types.py`](cts/types.py) | `NuVector.apply_config()` |
| &sect;7.1 | Statistical Protocol | [`cts/eval/statistics.py`](cts/eval/statistics.py) | `bootstrap_ci()`, `wilcoxon_signed_rank()` |
| Table 7 | Default Hyperparameters | [`configs/default.yaml`](configs/default.yaml) | &mdash; |

---

## Installation

### Requirements

| Component | Specification |
|:---|:---|
| Python | &ge; 3.10 |
| GPU | Single NVIDIA GPU with &ge; 24 GB VRAM (tested: RTX 4090, A100) |
| VRAM Usage | ~16.0 GB model + ~0.7 GB CTS overhead |
| Disk | ~20 GB (model weights + datasets) |

### Setup

```bash
# Clone and install
git clone https://github.com/heosanghun/Cognitive-Tree-Search.git
cd Cognitive-Tree-Search
pip install -e ".[dev,data,train,faiss]"

# (Optional) Triton kernels for fused sparse routing
pip install triton

# Download datasets
python scripts/download_experiment_data.py          # MATH-500, OpenMath-2
python scripts/download_all_benchmarks.py           # GSM8K, AIME, ARC-AGI-Text, HumanEval
```

---

## Training

### Stage 1: DEQ Warm-up (IFT Residual + Language Model Preservation)

Paper &sect;6: `L = ||f(z*) - z*||^2 + 0.1 * L_CE` over 5,000 steps on OpenMath-2.

```bash
export HF_TOKEN="hf_..."
python scripts/run_stage1_openmath.py \
    --lora \
    --device cuda:0 \
    --config configs/default.yaml
```

### Stage 2: PPO with GAE

Paper &sect;6: 800 PPO episodes with MATH-500 environment reward.

```bash
python scripts/run_stage2_math_ppo.py \
    --stage1-ckpt artifacts/stage1_last.pt \
    --device cuda:0
```

---

## Evaluation

### Full Table 2 Reproduction (5 seeds, Wilcoxon + Bonferroni)

```bash
python scripts/run_cts_eval_full.py --table2 --seeds 5 --device cuda:0
```

### Individual Benchmarks

```bash
python scripts/run_math500.py --device cuda:0
python scripts/run_gsm8k.py --device cuda:0
python scripts/run_humaneval.py --device cuda:0
python scripts/run_arc_agi_text.py --device cuda:0
```

### Statistical Protocol

All reported results follow the paper's protocol (&sect;7.1):
- **5 seeds** (3 full re-trainings + 2 inference-only)
- **95% CI** via bootstrap (1,000 resamples)
- **Wilcoxon signed-rank** with Bonferroni correction (&alpha; = 0.05/12)

---

## Repository Structure

```
cts/                        Core framework (70 Python modules)
  backbone/                   Gemma 4 E4B adapter, Wproj, LoRA
  deq/                        L-Broyden solver, DEQ transition, Jacobian inheritance
  mcts/                       PUCT, Algorithm 1 episode loop, hybrid KV
  policy/                     Meta-policy (nu in R^4), Neuro-Critic
  latent/                     FAISS-IVF-PQ context window, bottleneck
  routing/                    Sparse top-k MoE, Triton fused kernel
  train/                      Stage 1 warm-up, Stage 2 PPO + GAE
  eval/                       Benchmarks, VRAM profiler, Iso-FLOP, statistics
  types.py                    Core datatypes, nu-config Pareto modes
configs/                    Hyperparameters aligned with paper Table 7
scripts/                    Training, evaluation, and profiling entry points
tests/                      38 unit tests (pytest tests/ -q)
doc/                        Paper PDF, third-party notices
```

---

## Configuration

All hyperparameters in [`configs/default.yaml`](configs/default.yaml) are aligned with paper Table 7:

| Parameter | Value | Paper Reference |
|:---|:---|:---|
| `K` (latent tokens) | 64 | &sect;4.2 |
| `W` (branching factor) | 3 | &sect;4.1 |
| `top_k` (sparse routing) | 3 of 19 modules | Eq.&thinsp;3 |
| `broyden_memory_limit` | 16 (rank) | Table 1: ~0.12 GB FP32 |
| `broyden_max_iter` | 30 | &sect;5.2 |
| `stage1_lambda_lm` | 0.1 | &sect;6 |
| `stage1_max_steps` | 5,000 | &sect;6.1 |
| `ppo_episodes` | 800 | &sect;6.2 |
| `tau` (MAC budget) | 10<sup>14</sup> | &sect;7.1 |

---

## Testing

```bash
# Run all tests
pytest tests/ -q

# Run specific test suites
pytest tests/test_broyden_convergence.py -v     # L-Broyden solver + FP32 buffer
pytest tests/test_transition_smoke.py -v        # DEQ transition convergence
pytest tests/test_faiss_context.py -v           # FAISS-IVF-PQ context window
pytest tests/test_meta_policy_logits_nu.py -v   # Meta-policy nu output
pytest tests/test_batch_transition.py -v        # Parallel batch DEQ
```

---

## Citation

```bibtex
@inproceedings{cts2026,
  title     = {Cognitive Tree Search: {KV}-Cache-Free Per-Node Active {O}(1)
               Transitions for System 2 Inference via Deep Equilibrium Models},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Under double-blind review}
}
```

## License

[Apache License 2.0](LICENSE). Third-party notices: [`doc/THIRD_PARTY_NOTICES.md`](doc/THIRD_PARTY_NOTICES.md).
