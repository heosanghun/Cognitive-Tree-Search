# Memory definitions (M1 / M2)

## M1 — Per-transition (KV policy)

For a single call to `cts.deq.transition.transition()`:

- The implementation **must not** retain a growing per-path **Transformer KV cache** across tree depth in the same way as standard MCTS + autoregressive expansion.
- Profiling records **peak CUDA memory delta** attributable to transition internals (excluding deliberate baseline caches in `mcts_kv_baseline`).

## M2 — Tree-scale metadata

- Tree metadata may grow with node count (e.g., storing `z*` per node).
- Compare **total VRAM** against `mcts_kv_baseline` under matched `W`, depth sweep, and batching rules documented in `eval/profile_vram_latency.py`.

These definitions align with the paper’s distinction between context KV memory and lightweight metadata.
