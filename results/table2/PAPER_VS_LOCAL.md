# Paper vs Local &mdash; Table 2 cross-reference index

> **Status banner**: this document is a cross-reference index linking
> paper Table 2's reported headline numbers to their reviewer-facing
> reproduction evidence inside this codebase. The authoritative
> headline accuracy figures themselves live in the paper PDF
> (Table 2); the per-component code coverage map lives in
> [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) &sect;5; reviewer-facing
> FAQ entries live in [`REVIEWER_FAQ.md`](../../REVIEWER_FAQ.md)
> (Q1-Q18); open methodological caveats are consolidated in
> [`LIMITATIONS.md`](../../LIMITATIONS.md).

## Cross-reference index

| Paper element | Reviewer-facing entry inside this repository |
|:---|:---|
| §3-§7 method ↔ code anchors (Table 2 row computation) | [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md) §5 (25 rows) |
| Algorithm 1 — CTS Full Episode Loop | [`cts/mcts/cts_episode.py`](../../cts/mcts/cts_episode.py) `cts_full_episode()` |
| §4.1 — Meta-Policy πφ + Neuro-Critic Vψ (4-d ν vector, Eq. 1) | [`cts/policy/meta_policy.py`](../../cts/policy/meta_policy.py), [`cts/critic/neuro_critic.py`](../../cts/critic/neuro_critic.py) |
| §4.2 — KV-Cache-Free DEQ Transition | [`cts/deq/transition.py`](../../cts/deq/transition.py), [`cts/deq/broyden_forward.py`](../../cts/deq/broyden_forward.py) |
| §4.3 — Wproj soft-prompt decoding + FAISS-IVF-PQ Latent Context | [`cts/backbone/gemma_adapter.py`](../../cts/backbone/gemma_adapter.py) `decode_from_z_star()`, [`cts/latent/faiss_context.py`](../../cts/latent/faiss_context.py) |
| §5.3 Eq. 3 — Sparse Top-k Routing (CPU reference + Triton fused kernel) | [`cts/routing/sparse_moe_ref.py`](../../cts/routing/sparse_moe_ref.py), [`cts/routing/sparse_moe_triton.py`](../../cts/routing/sparse_moe_triton.py) |
| §6 — Stage 1 (DEQ warm-up, IFT + 0.1·LCE) + Stage 2 (PPO + GAE) | [`cts/train/stage1_warmup.py`](../../cts/train/stage1_warmup.py), [`cts/train/stage2_ppo_train.py`](../../cts/train/stage2_ppo_train.py), [`cts/train/ppo_core.py`](../../cts/train/ppo_core.py) |
| §7.1 — Statistical protocol (bootstrap CI + Wilcoxon + Bonferroni) | [`cts/eval/statistics.py`](../../cts/eval/statistics.py) |
| §7.7 — Hybrid KV-Assisted Mode (decision-plumbed) | [`cts/mcts/hybrid_kv.py`](../../cts/mcts/hybrid_kv.py) |
| Reviewer Quick Start audit (38/38 PASS) | [`scripts/_reviewer_local_audit.py`](../../scripts/_reviewer_local_audit.py) |
| Identity audit (PII = 0/0/0) | [`scripts/_audit_anon_zip.py`](../../scripts/_audit_anon_zip.py) |
| Anonymous submission ZIP | [`anonymous_submission_neurips2026.zip`](../../anonymous_submission_neurips2026.zip) |

## Why the Gap?

The headline absolute Table 2 numbers depend on the paper's full
compute envelope (8&times;A100, &tau;<sub>budget</sub>=10<sup>14</sup>,
multi-host PPO budget) which a single-host single-GPU reproduction
window cannot match without proportional compute scaling. Within that
constraint, the *relative ordering* of methods, the per-Node O(1) VRAM
signature (paper Table 1, structurally guaranteed by the KV-cache-free
DEQ transition), and the &nu;-control adaptive-operator mechanism
(paper Table 19) all reproduce on the local hardware class.
The full paper-headline accuracy on the paper backbone (Gemma 4 E4B)
is reserved for the camera-ready window. See
[`REVIEWER_FAQ.md`](../../REVIEWER_FAQ.md) Q4 for the per-row scaling
factor analysis, Q13 for the per-cell methodological context, and
Q15 for the single-host environment caveat.
