# Cognitive Tree Search (CTS)

**KV-Cache-Free Per-Node O(1) Transitions for System 2 Inference via Deep Equilibrium Models**

*Anonymous — NeurIPS 2026 (under review)*

<p align="center">
  <img src="assets/cts_architecture.png" alt="CTS Architecture" width="720">
</p>

CTS replaces explicit autoregressive KV-cache sequences with **implicit fixed-point transitions** via Deep Equilibrium Models, enabling MCTS on a single 24 GB GPU with a **constant ≤ 16.7 GB VRAM footprint** beyond depth 100 — where Vanilla MCTS triggers OOM at depth 15.

## Key Results

**Table 3 — Budget-Capped Performance** (≤ 10¹⁴ MACs, Gemma 4 E4B, 5 seeds, 95 % CI)

| Method | MATH | GSM8K | AIME | ARC | HumanEval | MACs (×10¹⁴) |
|--------|:----:|:-----:|:----:|:---:|:---------:|:---:|
| Greedy | 45.2 | 76.5 | 28.3 | 36.1 | 56.4 | 0.05 |
| SC@14 | 59.3±0.7 | 84.2±0.5 | 34.8±0.9 | 52.4±0.8 | 65.2±0.6 | 1.0 |
| Native Think | 57.0±0.6 | 82.4±0.4 | 42.5±0.9 | 50.1±0.7 | 63.3±0.5 | ≈0.80 |
| MCTS (Early Stop) | 56.5±0.9 | 81.2±0.7 | 38.4±0.8 | 48.1±1.0 | 62.5±0.7 | 0.8 |
| **CTS (Ours)** | **63.8±0.8** | **88.4±0.5** | **50.2±1.1** | **57.8±0.9** | **69.6±0.7** | **0.65** |

CTS uses only **65 %** of the MAC budget via ACT halting while achieving the best accuracy on all five benchmarks.

**Table 1 — VRAM Footprint** (W = 3)

| Method | Depth 1 | Depth 15 | Depth 35 | Depth 100+ |
|--------|:-------:|:--------:|:--------:|:----------:|
| MCTS (Vanilla) | 16.5 GB | OOM | — | — |
| MCTS (Prefix Cache) | 16.5 GB | 18.2 GB | OOM | — |
| **CTS (Ours)** | **16.5 GB** | **16.6 GB** | **16.6 GB** | **16.7 GB** |

## Quick Start

```bash
pip install -e ".[dev]"
pip install faiss-cpu
python scripts/download_experiment_data.py          # downloads MATH-500, OpenMath
python scripts/download_all_benchmarks.py           # downloads GSM8K, AIME, ARC, HumanEval

export HF_TOKEN="hf_..."
python scripts/run_stage1_openmath.py --lora --device cuda:0
python scripts/run_stage2_math_ppo.py --stage1-ckpt artifacts/stage1_last.pt --device cuda:0
python scripts/run_paper_reproduction.py --phase eval
```

## Repository Structure

```
cts/           Core framework (DEQ transition, Broyden solver, MCTS, routing, training)
configs/       Experiment configs (default.yaml aligned with Paper Table 7)
scripts/       Training, evaluation, and profiling scripts
data/          Benchmark datasets (auto-downloaded, .gitignored)
artifacts/     Checkpoints and experiment outputs
tests/         88 unit tests — pytest tests/ -q
```

## Hardware

| Requirement | Specification |
|-------------|---------------|
| GPU | Single NVIDIA RTX 4090 (24 GB) |
| VRAM | ~16.0 GB model + ~0.7 GB CTS overhead |
| Disk | ~20 GB (model weights + datasets) |

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

[Apache License 2.0](LICENSE). Third-party notices: [`doc/THIRD_PARTY_NOTICES.md`](doc/THIRD_PARTY_NOTICES.md).
