"""
Stage 1: DEQ warm-up toward fixed-point residual (see `doc/stage1_objective.md`).

Full OpenMathInstruct pipeline is not wired here; this module provides:
- `fixed_point_surrogate_loss`: one-step ||f(z) - z||^2 style loss on a backbone step
- `run_stage1_demo_step`: single optimizer step on `MockTinyBackbone` (CPU-friendly)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cts.backbone.mock_tiny import MockTinyBackbone
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights
from cts.types import NuVector


def fixed_point_surrogate_loss(
    backbone: nn.Module,
    parent_text: str,
    z: torch.Tensor,
    nu: NuVector,
    *,
    w_g: torch.Tensor,
    top_k: int = 3,
    extra: Dict[str, Any] | None = None,
) -> torch.Tensor:
    """
    Surrogate: minimize ||Phi(z) - z||^2 where Phi is one `deq_step` (DEQ literature uses
    implicit differentiation at z*; warm-up often uses unrolled / one-step consistency).
    """
    device = z.device
    context = backbone.encode_context(parent_text)
    if context.dim() == 1:
        context = context.unsqueeze(0)
    context = context.to(device=device, dtype=torch.float32)
    zf = z.float()
    alpha = routing_weights(zf, w_g.to(zf.device), nu.nu_ach)
    mw = sparse_module_weights(alpha, top_k)
    ex = dict(extra or {})
    phi_z = backbone.deq_step(z, context, mw, ex)
    return F.mse_loss(phi_z.float(), z.float())


def run_stage1_demo_step(
    *,
    lr: float = 1e-2,
    device: torch.device | None = None,
) -> Tuple[float, MockTinyBackbone]:
    """One Adam step on `MockTinyBackbone` mix/projection layers; returns (loss_value, backbone)."""
    dev = device or torch.device("cpu")
    bb = MockTinyBackbone(hidden=64, num_layers=42).to(dev)
    opt = torch.optim.Adam(bb.parameters(), lr=lr)
    nu = NuVector(nu_ne=0.5, nu_ach=1.0, nu_5ht=1.0)
    w_g = torch.randn(19, 64, device=dev) * 0.02
    z = torch.randn(8, 64, device=dev)
    opt.zero_grad()
    loss = fixed_point_surrogate_loss(bb, "demo prompt", z, nu, w_g=w_g)
    loss.backward()
    opt.step()
    return float(loss.detach().cpu().item()), bb


def run_stage1_stub() -> None:
    raise NotImplementedError(
        "Use run_stage1_demo_step() for smoke; full run: python scripts/run_stage1_openmath.py (optional --lora)"
    )
