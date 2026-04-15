"""Hybrid KV-Assisted Acceleration (paper §7.7, Conclusion).

Paper §7.7: "CTS uses <=16.7 GB of the 24 GB available, leaving ~7 GB unused
during tree search. An optional Hybrid KV-Assisted mode opportunistically
re-allocates this headroom: KV-states are selectively cached for shallow
nodes (D <= 5), where reuse is highest; deeper nodes use the full
KV-cache-free DEQ transitions. No retraining is required."

Result: wall-clock 27.3s -> 21.5s (-21%), accuracy unchanged (p=0.89),
VRAM <= 24 GB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class KVCacheEntry:
    """Cached KV-states for a shallow MCTS node."""
    node_id: int
    depth: int
    past_key_values: Any
    vram_bytes: int = 0


class HybridKVManager:
    """Manages selective KV-cache allocation for shallow nodes.

    Policy:
      - D <= shallow_depth_limit (default 5): cache KV-states
      - D > shallow_depth_limit: pure DEQ (KV-cache-free)
      - Total KV VRAM budget: max_kv_vram_bytes (default ~7 GB)
      - LRU eviction when budget exceeded
    """

    def __init__(
        self,
        *,
        shallow_depth_limit: int = 5,
        max_kv_vram_gb: float = 7.0,
    ) -> None:
        self.shallow_depth_limit = shallow_depth_limit
        self.max_kv_vram_bytes = int(max_kv_vram_gb * 1024**3)
        self._cache: Dict[int, KVCacheEntry] = {}
        self._access_order: List[int] = []
        self._total_vram: int = 0

    def should_cache_kv(self, depth: int) -> bool:
        """Paper §7.7: KV-states cached for D <= 5."""
        return depth <= self.shallow_depth_limit

    def get_cached_kv(self, node_id: int) -> Optional[Any]:
        """Retrieve cached KV-states for a node, if available."""
        entry = self._cache.get(node_id)
        if entry is not None:
            if node_id in self._access_order:
                self._access_order.remove(node_id)
            self._access_order.append(node_id)
            return entry.past_key_values
        return None

    def store_kv(
        self,
        node_id: int,
        depth: int,
        past_key_values: Any,
        vram_bytes: int = 0,
    ) -> None:
        """Cache KV-states for a shallow node with LRU eviction."""
        if not self.should_cache_kv(depth):
            return

        if vram_bytes == 0:
            vram_bytes = self._estimate_kv_size(past_key_values)

        while self._total_vram + vram_bytes > self.max_kv_vram_bytes and self._access_order:
            evict_id = self._access_order.pop(0)
            if evict_id in self._cache:
                self._total_vram -= self._cache[evict_id].vram_bytes
                del self._cache[evict_id]

        self._cache[node_id] = KVCacheEntry(
            node_id=node_id,
            depth=depth,
            past_key_values=past_key_values,
            vram_bytes=vram_bytes,
        )
        self._access_order.append(node_id)
        self._total_vram += vram_bytes

    def _estimate_kv_size(self, past_key_values: Any) -> int:
        """Estimate VRAM usage of KV-cache in bytes."""
        if past_key_values is None:
            return 0
        total = 0
        try:
            for layer_kv in past_key_values:
                if isinstance(layer_kv, (tuple, list)):
                    for t in layer_kv:
                        if isinstance(t, torch.Tensor):
                            total += t.nelement() * t.element_size()
        except (TypeError, AttributeError):
            total = 50 * 1024 * 1024
        return total

    @property
    def cached_nodes(self) -> int:
        return len(self._cache)

    @property
    def total_vram_mb(self) -> float:
        return self._total_vram / (1024 * 1024)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()
        self._total_vram = 0

    def report(self) -> Dict[str, Any]:
        return {
            "cached_nodes": self.cached_nodes,
            "total_vram_mb": round(self.total_vram_mb, 1),
            "max_vram_gb": self.max_kv_vram_bytes / (1024**3),
            "shallow_limit": self.shallow_depth_limit,
            "cache_hit_ids": list(self._cache.keys()),
        }


def hybrid_transition_decision(
    depth: int,
    node_id: int,
    kv_manager: Optional[HybridKVManager],
    backbone: nn.Module,
    parent_text: str,
) -> Tuple[bool, Optional[Any]]:
    """Decide whether to use cached KV or pure DEQ for a transition.

    Returns:
        (use_kv_cache, cached_past_key_values)
        - (True, past_kv): use cached KV for fast AR-style transition
        - (False, None): use pure DEQ transition (KV-cache-free)
    """
    if kv_manager is None:
        return False, None

    if not kv_manager.should_cache_kv(depth):
        return False, None

    cached = kv_manager.get_cached_kv(node_id)
    if cached is not None:
        return True, cached

    return False, None
