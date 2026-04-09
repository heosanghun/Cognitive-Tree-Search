# Layer ↔ module mapping (paper Table 3, 0-based code)

Paper uses **1-based** layer indices. Code uses **0-based** `layer_idx` where `layer_idx = paper_layer - 1`.

| Module | Paper layers (inclusive) | Code `layer_idx` range |
|--------|--------------------------|-------------------------|
| m1–m4 | 1–8 | 0–7 |
| m5–m8 | 9–16 | 8–15 |
| m9–m14 | 17–28 | 16–27 |
| m15–m19 | 29–42 | 28–41 |

Total paper layers: **42** → code indices **0..41**.

See `cts.model.module_partition.LAYER_TO_MODULE`.
