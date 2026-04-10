"""FAISS Latent Space Context Window (paper §4.4).

Historical fixed-points z* are mean-pooled and indexed in FAISS.
At each step t > faiss_min_steps, Top-k ancestral vectors are retrieved
to recycle context without linear sequence bloat.
"""

from __future__ import annotations

from typing import List, Optional

import torch

try:
    import faiss  # type: ignore[import-untyped]

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


def faiss_available() -> bool:
    return _FAISS_AVAILABLE


class LatentContextWindow:
    """Paper §4.4: FAISS-backed Latent Space Context Window.

    - Stores mean-pooled 1D representations of z* fixed points
    - Retrieves Top-k semantically relevant ancestral vectors
    - Injected as prepended global soft-prefix (p-RoPE isolated)
    """

    def __init__(self, dim: int, *, retrieval_k: int = 3, min_steps: int = 10) -> None:
        self.dim = dim
        self.retrieval_k = retrieval_k
        self.min_steps = min_steps
        self._vectors: List[torch.Tensor] = []
        self._step_count = 0

        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._index = None

    @property
    def size(self) -> int:
        return len(self._vectors)

    @property
    def step_count(self) -> int:
        return self._step_count

    def add(self, z_star: torch.Tensor) -> None:
        """Add a fixed-point to the context window.

        z_star: [K, d] latent tokens → mean-pooled to [d].
        """
        pooled = z_star.detach().float().mean(dim=0)
        if pooled.dim() > 1:
            pooled = pooled.reshape(-1)
        assert pooled.shape[0] == self.dim, (
            f"Expected dim={self.dim}, got {pooled.shape[0]}"
        )
        self._vectors.append(pooled.cpu())
        self._step_count += 1

        if self._index is not None:
            import numpy as np

            vec_np = pooled.cpu().numpy().reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(vec_np)
            self._index.add(vec_np)

    def retrieve(
        self, z_star: torch.Tensor, k: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Retrieve Top-k ancestral vectors (paper: t > min_steps).

        Returns: [k, d] tensor of retrieved vectors, or None if insufficient history.
        """
        k = k or self.retrieval_k
        if self._step_count <= self.min_steps:
            return None
        if len(self._vectors) < k:
            return None

        pooled = z_star.detach().float().mean(dim=0).reshape(-1)

        if self._index is not None:
            import numpy as np

            query = pooled.cpu().numpy().reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query)
            _, indices = self._index.search(query, min(k, self._index.ntotal))
            retrieved = [self._vectors[int(idx)] for idx in indices[0] if idx >= 0]
        else:
            # Fallback: cosine similarity without FAISS
            query_norm = pooled / (pooled.norm() + 1e-8)
            sims = []
            for i, v in enumerate(self._vectors):
                v_norm = v / (v.norm() + 1e-8)
                sims.append((float(torch.dot(query_norm, v_norm)), i))
            sims.sort(reverse=True)
            retrieved = [self._vectors[idx] for _, idx in sims[:k]]

        if not retrieved:
            return None
        return torch.stack(retrieved)

    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        per_vector = self.dim * 4  # float32
        return len(self._vectors) * per_vector

    def memory_kb_per_node(self) -> float:
        if not self._vectors:
            return 0.0
        return self.memory_bytes() / len(self._vectors) / 1024.0

    def reset(self) -> None:
        self._vectors.clear()
        self._step_count = 0
        if self._index is not None:
            self._index = faiss.IndexFlatIP(self.dim)


def prepend_soft_prefix(
    context: torch.Tensor, retrieved: torch.Tensor
) -> torch.Tensor:
    """Inject retrieved ancestral vectors as prepended global soft-prefix.

    Paper §4.4: mathematically isolates relative position indices from
    the active transition step (p-RoPE separation).

    context: [seq_len, d]
    retrieved: [k, d]
    Returns: [k + seq_len, d]
    """
    if retrieved.device != context.device:
        retrieved = retrieved.to(context.device)
    if retrieved.dtype != context.dtype:
        retrieved = retrieved.to(context.dtype)
    return torch.cat([retrieved, context], dim=0)
