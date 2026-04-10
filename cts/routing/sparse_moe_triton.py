"""Triton fused sparse routing (paper Appendix A.2) — fallback to PyTorch ref."""

from __future__ import annotations

import torch

from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights


def routing_weights_triton(
    z: torch.Tensor, w_g: torch.Tensor, nu_temp: float, top_k: int
) -> torch.Tensor:
    # Placeholder: call reference until kernel lands
    alpha = routing_weights(z, w_g, nu_temp)
    return sparse_module_weights(alpha, top_k)
