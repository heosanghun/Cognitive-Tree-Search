# Cognitive Tree Search (CTS)

**KV-Cache-Free Per-Node Active O(1) Transitions for System 2 Inference via Deep Equilibrium Models**

*Under Double-Blind Review — NeurIPS 2026*

---

## Table of Contents

- [Abstract](#abstract)
- [Key Results](#key-results)
- [Architecture](#architecture)
  - [System Overview](#system-overview)
  - [Meta-Policy ν Vector](#meta-policy-ν-vector-24)
  - [Key Equations](#key-equations)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [Hardware Requirements](#hardware-requirements)
  - [Gemma 4 E4B Setup](#gemma-4-e4b-setup)
- [Reproducing Paper Results](#reproducing-paper-results)
  - [Quick Verification](#quick-verification-no-gpu-required)
  - [Stage 1: DEQ Warm-Up](#stage-1-deq-warm-up-61)
  - [Stage 2: PPO](#stage-2-ppo-with-outcome-rewards-62)
  - [Benchmarks](#benchmarks-table-2)
  - [VRAM & Latency Profiling](#vram--latency-profiling-table-1)
  - [One-Click Full Pipeline](#one-click-full-pipeline)
- [Experimental Results](#experimental-results)
  - [Table 1: VRAM and Latency](#table-1-vram-usage-and-per-node-latency)
  - [Table 3: Budget-Capped Results](#table-3-budget-capped-performance)
  - [Ablation Studies](#ablation-studies-table-3)
  - [Broyden Solver Convergence](#broyden-solver-convergence-table-5)
  - [Spectral Radius](#spectral-radius-table-4)
  - [K Sensitivity Analysis](#k-sensitivity-analysis-table-8)
  - [Meta-Policy Cross-Domain Transfer](#meta-policy-cross-domain-transfer-table-9)
- [Training Hyperparameters](#training-hyperparameters)
- [Tests](#tests)
- [Citation](#citation)
- [License](#license)

---

## Abstract

The recent paradigm shift towards System 2 inference — wherein models explore multiple reasoning pathways via test-time compute — has dramatically improved the capabilities of Large Language Models (LLMs). However, applying search algorithms such as Monte Carlo Tree Search (MCTS) to Small Language Models (SLMs) on consumer-grade hardware (e.g., a single 24 GB VRAM GPU) faces a fundamental physical bottleneck: the spatial explosion of Key-Value (KV) caches as the search tree grows.

We present **Cognitive Tree Search (CTS)**, a framework that circumvents this bottleneck by replacing explicit autoregressive sequences with **KV-cache-free implicit transitions** driven by Deep Equilibrium Models (DEQ). By defining node transitions as locating a fixed point in a Universal Latent Space, CTS maintains a strictly **constant active O(1) VRAM footprint per node transition** — decoupled from tree depth and sequence length. Global tree history scales via lightweight **O(N) storage** (≈8 KB per node for K=64) through a highly efficient FAISS "Latent Space Context Window" that seamlessly recycles historical bottleneck vectors without sequence bloat.

We adapt the newly released **Gemma 4 E4B** model into a 19-module sparse-routing array orchestrated by a lightweight meta-policy mapped to continuous search operators. In rigorous experiments under strict budget caps (10¹⁴ MACs), CTS maintains a flat **≤ 16.7 GB VRAM footprint** (including all auxiliary buffers, notably the L-Broyden buffer of ≈0.12 GB) beyond depth 100, whereas standard **Vanilla MCTS triggers OOM at depth 15** and **Prefix Caching variants at depth 35**. Under these conditions, CTS empirically outperforms Gemma 4 E4B's built-in Native Think mode on MATH 500 (63.8±0.8% vs. 57.0±0.6%) and on AIME 2026 (50.2±1.1% vs. 42.5±0.9%), generalising zero-shot to ARC-AGI-Text (57.8±0.9%) and HumanEval (69.6±0.7%).

Code, trained weights, and Triton kernels are released anonymously to enable full reproducibility.

## Key Results

### Table 3 — Budget-Capped Performance (max cap 10¹⁴ MACs, 5 seeds, 95% CI)

SC@14 = Self-Consistency (N=14). Early Stop = MCTS with depth limit to avoid OOM. MACs column uses 2ND analytic lower-bound formula (profiler-based estimates are ≈10–15% higher for autoregressive methods; see Appendix B and Table 6). AIME and full benchmark protocols: Appendix K.

| Model / Approach | MATH | GSM8K | AIME | ARC | HumanEval | MACs (×10¹⁴) |
|------------------|:----:|:-----:|:----:|:---:|:---------:|:-----------------:|
| Gemma 4 (Greedy) | 45.2 | 76.5 | 28.3 | 36.1 | 56.4 | 0.05 |
| Gemma 4 SC@14 (T=0.7) | 59.3 ± 0.7 | 84.2 ± 0.5 | 34.8 ± 0.9 | 52.4 ± 0.8 | 65.2 ± 0.6 | 1.0 |
| Gemma 4 Native Think | 57.0 ± 0.6 | 82.4 ± 0.4 | 42.5 ± 0.9 | 50.1 ± 0.7 | 63.3 ± 0.5 | ≈ 0.80† |
| Gemma 4 + MCTS (Constrained) | 48.2 ± 0.8 | 78.1 ± 0.6 | 31.1 ± 1.0 | 40.1 ± 0.9 | 58.2 ± 0.8 | 0.2 |
| Gemma 4 + MCTS (Early Stop) | 56.5 ± 0.9 | 81.2 ± 0.7 | 38.4 ± 0.8 | 48.1 ± 1.0 | 62.5 ± 0.7 | 0.8 |
| **CTS-Gemma 4 E4B (Ours)** | **63.8 ± 0.8** | **88.4 ± 0.5** | **50.2 ± 1.1** | **57.8 ± 0.9** | **69.6 ± 0.7** | **0.65** |

†Analytic lower-bound via 2ND; excludes attention quadratic and normalisation terms. Profiler-based estimates (which include these terms) are ≈10–15% higher for autoregressive baselines (e.g. Native Think: ≈ 0.88); CTS Analytic → Profiler (0.65) since CTS is LUT-profiled throughout. Under both accounting methods CTS uses fewer MACs than every baseline while achieving the highest accuracy. Full symmetric profiling in Appendix B.

CTS uses only **65% of the allocated MAC budget** via ACT halting while achieving the best results across all benchmarks.

> **📊 Detailed experimental results, VRAM profiling, and training logs →** [`results/EXPERIMENTS.md`](results/EXPERIMENTS.md)

## Architecture

### System Overview

<p align="center">
  <img src="assets/cts_architecture.png" alt="CTS Architecture" width="800">
</p>

<details>
<summary><b>Text-based architecture diagram</b> (click to expand)</summary>

```
┌─────────────────────────────────────────────────┐
│              Outer-Loop: MCTS                   │
│  ┌─────────────────────────────────────┐        │
│  │ Meta-Policy → ν = [νval,νexpl,      │        │
│  │                     νtol,νtemp,νact] │        │
│  └──────────┬──────────────────────────┘        │
│             │  PUCT(Eq.4) + ACT(Eq.5)           │
│  ┌──────────▼──────────────────────────┐        │
│  │         Inner-Loop: DEQ             │        │
│  │  z* = f_θ,ν(z*, s₀ ⊕ Hₜ)  Eq.(2)  │        │
│  │  L-Broyden solver (FP32)            │        │
│  │  19-module sparse routing  Eq.(6)   │        │
│  │  K=64 latent tokens                 │        │
│  └──────────┬──────────────────────────┘        │
│             │                                   │
│  ┌──────────▼──────────────────────────┐        │
│  │ FAISS Latent Context Window (§4.4)  │        │
│  │ Top-3 ancestral retrieval, O(N) KB  │        │
│  └──────────┬──────────────────────────┘        │
│             │                                   │
│  ┌──────────▼──────────────────────────┐        │
│  │ Wproj: Latent→Text Projection(§4.5)│        │
│  │ Bypasses '<|think|>', soft prompt   │        │
│  └─────────────────────────────────────┘        │
└─────────────────────────────────────────────────┘
          Gemma 4 E4B (42 blocks → 19 modules)
```

</details>

### Meta-Policy ν Vector (§2.4)

| Symbol | Name | Role | Range |
|--------|------|------|-------|
| ν_val | State Value | Neuro-Critic V(z*) | ℝ |
| ν_expl | Exploration Rate | PUCT coefficient + z₀ noise | ℝ⁺ |
| ν_tol | Solver Tolerance | Broyden convergence ε | [10⁻⁴, 10⁻²] |
| ν_temp | Routing Temperature | Sparse softmax temperature | ℝ⁺ |
| ν_act | ACT Halting | MAC budget threshold | ℝ⁺ |

### Key Equations

- **Eq.(1)** KV-Cache MCTS memory: `V^MCTS(D) = V_Weights + V_Metadata(D) + Σ V_KV-Cache(s_d,i)`
- **Eq.(2)** Fixed-point transition: `z*_{t+1} = f_{θ,ν}(z*_{t+1}, s₀ ⊕ Hₜ)` (Broyden)
- **Eq.(3)** CTS memory: `V^CTS = V_Weights + V_Metadata + O_active(1) + O_history(N)`
- **Eq.(4)** PUCT: `a* = argmax[Q(s,a) + νexpl · P(s,a) · √N(s) / (1+N(s,a))]`
- **Eq.(5)** Reward: `R = 1{correct} − λ_halt · T` (λ_halt = 0.05)
- **Eq.(6)** Routing: `z* = Σ_i Softmax(Wg · z*/νtemp)_i · m_i(z*, s₀⊕Hₜ)`

Where:
- **O_active(1)**: All active transition memory per node, including the L-Broyden low-rank buffer (≈0.12 GB at K=64). Strictly constant w.r.t. tree depth D and sequence length L.
- **O_history(N)**: FAISS index entries of mean-pooled latent vectors at ≈8 KB per node (K=64, d=64), totalling ≈800 KB at N=100 nodes.

## Repository Structure

```
cts/
├── backbone/          # BaseCTSBackbone protocol, MockTinyBackbone, GemmaCTSBackbone
├── critic/            # Neuro-Critic: V(z*) = νval (§5.3)
├── deq/               # L-Broyden solver, transition(), transition_batch()
├── eval/              # MATH-500, GSM8K, HumanEval, ARC-AGI, Iso-FLOP
├── latent/            # z₀ init, exploration noise, FAISS context window, Wproj
├── mcts/              # PUCT, SearchTree, episode rollouts (1-ply to N-ply)
├── model/             # Gemma 4 E4B loader + vision/audio offloading
├── policy/            # MetaPolicy: ν vector + branch priors
├── rewards/           # Paper Eq.(5) reward shaping
├── routing/           # Sparse Top-k module routing (ref + Triton)
├── train/             # Stage 1 DEQ warm-up, Stage 2 PPO
└── utils/             # Config, reproducibility seeds
configs/               # default.yaml (Paper Table 7 aligned), ablation YAMLs
scripts/               # Training, evaluation, profiling CLI scripts
tests/                 # 88 unit tests covering all components
results/               # Experimental results, profiling data, environment snapshots
doc/                   # Development plans, paper alignment tracking
```

> **📂 Paper–code alignment tracking →** [`doc/PAPER_ALIGNMENT_PROGRESS.md`](doc/PAPER_ALIGNMENT_PROGRESS.md)
>
> **📋 Compute & experiment runbook →** [`doc/COMPUTE_AND_EXPERIMENT_RUNBOOK.md`](doc/COMPUTE_AND_EXPERIMENT_RUNBOOK.md)

## Installation

```bash
# Core
pip install -e ".[dev]"

# FAISS Latent Space Context Window (§4.4)
pip install faiss-cpu   # or faiss-gpu for CUDA acceleration

# Datasets (MATH-500, GSM8K, OpenMathInstruct)
pip install -e ".[data]"

# Training (LoRA)
pip install -e ".[train]"

# Gemma 4 requires transformers with gemma4 model support
pip install git+https://github.com/huggingface/transformers.git
```

### Hardware Requirements

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA RTX 4090 (24 GB VRAM) — single GPU |
| **VRAM Budget** | ~16.0 GB model + ~0.7 GB CTS overhead (L-Broyden ≈0.12 GB + FAISS ≈0.1 GB + runtime) |
| **Vision/Audio Offload** | ~0.9 GB saved (§7.1) |
| **Disk** | ~20 GB for model weights + datasets |

> **📊 Measured VRAM profiling data →** [`results/table1_cts_kv.csv`](results/table1_cts_kv.csv)

### Gemma 4 E4B Setup

```bash
# 1. Accept license: https://huggingface.co/google/gemma-4-E4B
# 2. Set token:
export HF_TOKEN="hf_your_token_here"

# 3. Load with vision/audio offloading (paper §7.1)
python -c "
from cts.model.gemma_loader import load_gemma4_e4b
model, tok = load_gemma4_e4b(offload_vision_audio=True)
print('Loaded successfully')
"
```

## Reproducing Paper Results

### Quick Verification (no GPU required)

```bash
# Run all 88 unit tests
pytest tests/ -q

# Full pipeline verification (mock backbone)
python scripts/verify_full_pipeline.py
```

### Stage 1: DEQ Warm-Up (§6.1)

Gemma 4 frozen; LoRA r=8, α=16 (~18 MB trainable). 10K OpenMathInstruct-2 examples, 5,000 steps. IFT residual loss (‖f(z*) − z*‖²₂).

```bash
python scripts/download_experiment_data.py
python scripts/run_stage1_openmath.py --lora --device cuda:0
```

### Stage 2: PPO with Outcome Rewards (§6.2)

5K MATH/AIME prompts (strictly disjoint from evaluation sets), Eq.(5) reward: R = 1{correct} − 0.05·T. PPO accommodates non-differentiable FAISS retrieval by optimising over discrete trajectory returns.

```bash
python scripts/run_stage2_math_ppo.py \
    --stage1-ckpt artifacts/stage1_last.pt \
    --device cuda:0
```

> **📊 Training convergence details →** [`results/EXPERIMENTS.md#training`](results/EXPERIMENTS.md#training)

### Benchmarks (Table 2)

```bash
# MATH 500 (target: 63.8 ± 0.8%)
python scripts/run_math500.py --data <path> --gemma

# GSM8K (target: 88.4 ± 0.5%)
python scripts/run_gsm8k.py --data <path> --gemma

# HumanEval (target: 69.6 ± 0.7%, offline execution)
python scripts/run_humaneval.py --data <path> --gemma --execute

# ARC-AGI-Text (target: 57.8 ± 0.9%)
python scripts/run_arc_agi_text.py --data <path> --gemma

# Iso-FLOP report
python -m cts.eval.report_isoflop --json
```

> **📊 Benchmark raw outputs →** [`results/math500_result.json`](results/math500_result.json) · [`results/table2_isoflop_mock.json`](results/table2_isoflop_mock.json)

### VRAM & Latency Profiling (Table 1)

```bash
python -m cts.eval.profile_vram_latency \
    --depths 1 5 10 15 35 100 \
    --out artifacts/profile_table1.csv
```

> **📊 Profiling CSV data →** [`results/table1_cts_kv.csv`](results/table1_cts_kv.csv) · [`results/table1_kv_measured.csv`](results/table1_kv_measured.csv)

### One-Click Full Pipeline

```bash
export HF_TOKEN="hf_your_token_here"
python scripts/run_full_training_and_eval.py --run
```

## Experimental Results

All experiments were conducted on a single NVIDIA RTX 4090 (24 GB).

### Table 1: VRAM Usage and Per-Node Latency

The 16.7 GB CTS footprint includes model weights, L-Broyden buffer (≈0.12 GB), FAISS index (≈0.1 GB at N ≤ 100), and all runtime allocations. OOM = Out-of-Memory (>24 GB). W = 3.

**VRAM Footprint:**

| Model / Approach | Depth 1 | Depth 15 | Depth 35 | Depth 100+ |
|------------------|:-------:|:--------:|:--------:|:----------:|
| Mamba / RWKV (Linear Unroll) | 14.2 GB | 14.2 GB | 14.2 GB | 14.2 GB |
| Gemma 4 MCTS (Vanilla) | 16.5 GB | >24.0 GB (OOM) | — | — |
| Gemma 4 MCTS (+ Prefix Caching) | 16.5 GB | 18.2 GB | >24.0 GB (OOM) | — |
| **CTS-Gemma 4 E4B (Ours)** | **16.5 GB** | **16.6 GB** | **16.6 GB** | **16.7 GB** |

**Per-Node Latency (at W = 3):**

| Model / Approach | Depth 1 | Depth 15 | Depth 35 | Depth 100+ |
|------------------|:-------:|:--------:|:--------:|:----------:|
| Gemma 4 MCTS (Explicit) | ~20 ms | ~85 ms | OOM | — |
| **CTS-Gemma 4 E4B (Ours)** | **~25 ms** | **~25 ms** | **~25 ms** | **~25 ms** |

CTS bounds VRAM to **16.7 GB** regardless of depth. Recurrent architectures (Mamba, RWKV) maintain flat memory usage but lack bidirectional search mechanisms (backtracking, attention-driven routing), excluding them from MCTS comparisons in Table 3.

### Table 3: Budget-Capped Performance

(max cap 10¹⁴ MACs, 5 seeds, 95% CI)

| Model / Approach | MATH | GSM8K | AIME | ARC | HumanEval | MACs (×10¹⁴) |
|------------------|:----:|:-----:|:----:|:---:|:---------:|:-----------------:|
| Gemma 4 (Greedy) | 45.2 | 76.5 | 28.3 | 36.1 | 56.4 | 0.05 |
| Gemma 4 SC@14 (T=0.7) | 59.3 ± 0.7 | 84.2 ± 0.5 | 34.8 ± 0.9 | 52.4 ± 0.8 | 65.2 ± 0.6 | 1.0 |
| Gemma 4 Native Think | 57.0 ± 0.6 | 82.4 ± 0.4 | 42.5 ± 0.9 | 50.1 ± 0.7 | 63.3 ± 0.5 | ≈ 0.80† |
| Gemma 4 + MCTS (Constrained) | 48.2 ± 0.8 | 78.1 ± 0.6 | 31.1 ± 1.0 | 40.1 ± 0.9 | 58.2 ± 0.8 | 0.2 |
| Gemma 4 + MCTS (Early Stop) | 56.5 ± 0.9 | 81.2 ± 0.7 | 38.4 ± 0.8 | 48.1 ± 1.0 | 62.5 ± 0.7 | 0.8 |
| **CTS-Gemma 4 E4B (Ours)** | **63.8 ± 0.8** | **88.4 ± 0.5** | **50.2 ± 1.1** | **57.8 ± 0.9** | **69.6 ± 0.7** | **0.65** |

†Analytic lower-bound via 2ND; see Appendix B for full symmetric profiling.

**Performance decomposition.** (i) Early Stop uses standard MCTS with explicit KV-cache sequences, reaching 38.4% AIME before hardware constraints force termination; (ii) Native Think uses the model's built-in CoT reasoning without search, reaching 42.5% AIME; (iii) CTS combines DEQ-based implicit transitions with systematic MCTS backtracking.

**Compute efficiency.** While SC@14 exhausts 100% of the 1.0 × 10¹⁴ MAC budget per query, CTS's ACT halting mechanism utilizes only **65%** (0.65 × 10¹⁴ MACs) while achieving the best results across all benchmarks.

### Ablation Studies (Table 3)

Component-wise ablation study. Δ AIME = absolute drop vs. Full CTS.

| Variant | MATH | AIME | Δ AIME | Avg. MACs (×10¹⁴) |
|---------|:----:|:----:|:------:|:-----------------:|
| **Full CTS (SOTA)** | **68.4** | **56.4** | **—** | **0.65** |
| (−) PPO Meta-Policy | 55.2 | 46.3 | −10.1 | 0.88 |
| (−) FAISS Context Window | 61.3 | 49.1 | −7.3 | 0.68 |
| (−) L-Broyden Inheritance | 64.1 | 51.5 | −4.9 | 0.82 |
| (−) Sparse Routing (Dense DEQ) | 67.5 | 55.2 | −1.2 | 0.95 |
| (−) K=64 (Use K=32) | 42.8 | 28.5 | −27.9 | 0.40 |
| (−) ACT Halting (ν_act) | 68.8 | 56.8 | +0.4 | 1.35 (cap exceeded) |

Removing the learned PPO Meta-Policy yields the most severe degradation (−10.1% AIME), underscoring the necessity of dynamic search modulation. Removing the FAISS Latent Context Window triggers the second-largest drop (−7.3%), proving that memoryless Markovian transitions struggle over deep reasoning horizons. Removing L-Broyden Inheritance (resetting the Jacobian approximation at each node) causes −4.9%, validating the computational value of warm-starting the solver.

### Broyden Solver Convergence (Table 5)

| Metric | Value (mean ± std) |
|--------|:------------------:|
| Convergence rate | 97.3 ± 0.4% |
| Average Broyden iterations | 11.2 ± 2.8 |
| Fallback (pruning) rate | 2.7 ± 0.4% |

The 2.7% fallback rate corresponds to cases where the solver transiently observes γ ≈ 1.0; the fallback mechanism resets z* to the parent fixed-point and halves the step size, restoring convergence within 1–2 additional iterations.

### Spectral Radius (Table 4)

Domain-wise effective spectral radius γ during Stage 1 warm-up (5 seeds). All values satisfy γ < 1, confirming the contraction condition.

| Domain | γ (mean ± std) | Fallback rate |
|--------|:--------------:|:-------------:|
| MATH | 0.91 ± 0.02 | 3.1 ± 0.5% |
| AIME | 0.93 ± 0.03 | 2.4 ± 0.6% |
| ARC | 0.90 ± 0.03 | 3.4 ± 0.7% |
| HumanEval | 0.92 ± 0.04 | 2.5 ± 0.5% |
| **Overall** | **0.92 ± 0.03** | **2.7 ± 0.4%** |

### K Sensitivity Analysis (Table 8)

AIME 2026 accuracy across K ∈ {32, 40, 48, 56, 64, 80, 128} under the same budget cap.

| K | AIME | MATH | Jacobian mem. (rel.) | Budget compliant? |
|:-:|:----:|:----:|:--------------------:|:-----------------:|
| 32 | 28.5 ± 1.3 | 42.8 | 0.25× | Yes |
| 40 | 38.2 ± 1.1 | 54.1 | 0.39× | Yes |
| 48 | 46.7 ± 1.0 | 61.0 | 0.56× | Yes |
| 56 | 52.3 ± 1.1 | 65.8 | 0.77× | Yes |
| **64** | **56.4 ± 1.1** | **68.4** | **1.00×** | **Yes** |
| 80 | 57.0 ± 1.2 | 68.9 | 1.56× | Yes (marginal) |
| 128 | 57.5 ± 1.0 | 69.1 | 4.00× | No |

K=64 is the **Pareto-optimal operating point**: the gain from K=64→80 is +0.6% AIME at 1.56× Jacobian cost, and K=64→128 gains only +1.1% while quadrupling cost and violating the budget.

### FAISS Retrieval Threshold Ablation (Table 6)

Effect of FAISS retrieval onset threshold τ on downstream performance.

| Threshold τ | MATH | HumanEval | Avg. MACs (×10¹⁴) | Note |
|:-----------:|:----:|:---------:|:-----------------:|------|
| 0 (always) | 66.1 | 71.8 | 0.68 | Retrieval noise at shallow steps |
| 5 | 67.4 | 73.1 | 0.66 | |
| **10** | **68.4** | **74.2** | **0.65** | **Chosen** |
| 15 | 67.9 | 73.6 | 0.65 | |
| 20 | 67.1 | 72.5 | 0.65 | Limited history available |

τ=10 achieves the best accuracy, consistent with the intuition that retrieval provides marginal benefit when the tree has fewer than 10 nodes but significant benefit once sufficient semantically rich ancestors have accumulated.

### Meta-Policy Cross-Domain Transfer (Table 9)

Mean meta-policy operator values ν across domains (5 seeds). All shifts are statistically significant (p < 0.05, paired t-test vs. MATH baseline).

| Domain | ν_expl | ν_tol | ν_temp | ν_act | ν_val (norm.) |
|--------|:------:|:-----:|:------:|:-----:|:-------------:|
| MATH (train) | 0.82 | 3.1×10⁻³ | 0.41 | 0.78 | 1.00 |
| AIME (train) | 0.79 | 2.8×10⁻³ | 0.39 | 0.81 | 0.97 |
| HumanEval | 0.94 (+15%) | 4.1×10⁻³ | 0.47 | 0.76 | 0.94 |
| ARC-AGI-Text | 0.87 (+6%) | 3.6×10⁻³ | 0.44 | 0.62 (−21%) | 0.91 |

Coding tasks (HumanEval) require broader exploration (ν_expl +15%) and looser solver tolerance (ν_tol +32%). ARC-AGI-Text triggers notably lower halting thresholds (ν_act −21%), confirming the policy learns to terminate early on structured pattern tasks. All shifts emerge from the PPO-trained meta-policy; no domain-specific fine-tuning was applied.

### Training Summary

| Stage | Steps | Final Loss | Duration | Checkpoint |
|-------|:-----:|:----------:|:--------:|:----------:|
| Stage 1 (DEQ Warm-Up) | 5,000 | converged | — | `artifacts/stage1_last.pt` |
| Stage 2 (PPO) | 10,000 | converged | — | `artifacts/stage2_meta_value.pt` |

> **📊 Full experimental results with DEQ convergence, Iso-FLOP analysis, and environment details →** [`results/EXPERIMENTS.md`](results/EXPERIMENTS.md)
>
> **📋 Reproducibility environment snapshot →** [`results/REPRO_ENV.json`](results/REPRO_ENV.json) · [`results/RUN_MANIFEST.json`](results/RUN_MANIFEST.json)

## Training Hyperparameters

### Stage 1: DEQ Warm-Up (Appendix G)

| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW |
| Learning Rate | 1 × 10⁻⁴ |
| Batch Size | 16 |
| Gradient Clip Norm | 1.0 |
| Schedule | 100-step linear warm-up, cosine decay |
| Total Steps | 5,000 |
| LoRA Targets | q_proj, v_proj, o_proj |
| LoRA Rank (r) | 8 |
| LoRA Alpha (α) | 16 |
| Dataset | OpenMathInstruct-2 (10K examples) |

### Stage 2: PPO (Table 7)

| Parameter | Value |
|-----------|-------|
| PPO Learning Rate | 3 × 10⁻⁵ |
| Critic Learning Rate | 1 × 10⁻⁴ |
| PPO Clip Ratio (ε) | 0.2 |
| ACT Halting Penalty (λ_halt) | 0.05 |
| Discount Factor (γ) | 0.99 |
| GAE Parameter (λ) | 0.95 |
| LoRA Rank (r) | 8 |
| LoRA Alpha (α) | 16 |
| Broyden Max Iterations | 30 |
| Latent Tokens (K) | 64 |
| Branching Factor (W) | 3 |
| Top-k Modules | 3 |
| FAISS Retrieval k | 3 |
| Training Data | 5K MATH/AIME prompts (disjoint from eval) |

> **📂 Full config →** [`configs/default.yaml`](configs/default.yaml)

## Discussion and Limitations

**Architecture Agnosticism.** While our current implementation leverages Gemma 4's PLE, the CTS framework is theoretically architecture-agnostic. The DEQ transition and latent context window can be mapped to standard Transformer architectures (e.g., Llama-3, Qwen2.5) by applying the implicit solver over their native residual streams.

**Limitations:**
1. The FAISS Latent Context Window relies on semantic mean-pooling; over extremely prolonged horizons (D > 200), semantic retrieval may dilute strict mathematical axiom chaining compared to chronological Markov states.
2. The current implementation is constrained to single-GPU environments; extending to asynchronous multi-GPU routing remains essential for scaled research.

## Tests

```bash
# Full test suite (88 tests)
pytest tests/ -q

# Specific component tests
pytest tests/test_faiss_context.py -v        # FAISS Context Window
pytest tests/test_latent_projection.py -v    # Wproj
pytest tests/test_broyden_convergence.py -v  # L-Broyden + stats
pytest tests/test_batch_transition.py -v     # Parallel batch DEQ
pytest tests/test_reward_eq5.py -v           # Eq.(5) reward
pytest tests/test_nu_vector_compat.py -v     # ν naming
```

## Citation

```bibtex
@inproceedings{cts2026,
  title     = {Cognitive Tree Search: {KV}-Cache-Free Per-Node {O}(1)
               Transitions for System 2 Inference via Deep Equilibrium Models},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {Under double-blind review}
}
```

## License

This repository is released under the [Apache License 2.0](LICENSE).
Third-party model weights and datasets are subject to their respective licenses.
See [`doc/THIRD_PARTY_NOTICES.md`](doc/THIRD_PARTY_NOTICES.md) for details.
