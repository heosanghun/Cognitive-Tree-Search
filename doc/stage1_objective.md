# Stage 1 objective (fixed for implementation)

**Selected:** `fixed_point_residual` — minimize \(\lVert z - f_\theta(z, s_t)\rVert\) (or equivalent MSE) on OpenMathInstruct subset (10k samples per paper appendix intent).

**Alternatives (not default):** `lm_nll`, `combined` — switch via `configs/default.yaml` key `stage1_loss`.

Rationale: aligns with DEQ stability before outer-loop PPO without requiring undisclosed paper details.
