"""Single CTS transition API: DEQ fixed point + optional ACT + mock decode."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cts.backbone.protocol import BaseCTSBackbone
from cts.deq.broyden_forward import broyden_fixed_point, map_nu_ne_to_tol
from cts.latent.bottleneck import add_serotonin_noise, init_z0
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights
from cts.types import NuVector, RuntimeBudgetState, TransitionResult


def _load_mac_lut() -> list:
    p = Path(__file__).resolve().parent.parent / "routing" / "lut_mac.json"
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["mac_per_module"])


def transition(
    parent_text: str,
    branch_index: int,
    nu: NuVector,
    budget: RuntimeBudgetState,
    backbone: BaseCTSBackbone,
    *,
    K: int = 8,
    d: Optional[int] = None,
    top_k: int = 3,
    broyden_max_iter: int = 30,
    broyden_tol_min: float = 1e-4,
    broyden_tol_max: float = 1e-2,
    tau_flops_budget: float = 1e15,
    generator: Optional[torch.Generator] = None,
    routing_mode: str = "sparse",
    max_decode_tokens: int = 1,
) -> TransitionResult:
    """
    One KV-cache-free style transition using DEQ on mock/real backbone.
    `routing_mode`: "sparse" (paper) or "dense" (ablation).
    `max_decode_tokens`: passed to `decode_from_z_star(..., max_new_tokens=...)` when supported.
    """
    if isinstance(backbone, nn.Module):
        device = next(backbone.parameters()).device
    else:
        device = torch.device("cpu")
    d = d or backbone.hidden_size
    gen = generator or torch.Generator(device=device)
    gen.manual_seed(2026 + branch_index * 31 + (len(parent_text) % 997))

    z0 = init_z0(K, d, device, gen)
    z0 = add_serotonin_noise(z0, nu.nu_5ht, gen)

    context = backbone.encode_context(parent_text)
    if context.dim() == 1:
        context = context.unsqueeze(0)
    context = context.to(device=device, dtype=torch.float32)

    if hasattr(backbone, "routing_matrix"):
        w_g = backbone.routing_matrix().to(device=device, dtype=torch.float32)
    else:
        w_g = torch.randn(19, d, device=device, dtype=torch.float32) * 0.02
    macs = _load_mac_lut()

    def phi(zz: torch.Tensor) -> torch.Tensor:
        alpha = routing_weights(zz, w_g, nu.nu_ach)
        if routing_mode == "dense":
            mw = alpha
        else:
            mw = sparse_module_weights(alpha, top_k)
        extra: Dict[str, Any] = {"top_k": top_k}
        return backbone.deq_step(zz, context, mw, extra)

    tol = map_nu_ne_to_tol(nu.nu_ne, broyden_tol_min, broyden_tol_max)
    z_star, info = broyden_fixed_point(phi, z0, tol=tol, max_iter=broyden_max_iter)

    budget = budget.clone()
    flops = 0.0
    with torch.no_grad():
        alpha = routing_weights(z_star, w_g, nu.nu_ach)
        mw = alpha if routing_mode == "dense" else sparse_module_weights(alpha, top_k)
        for i in range(19):
            flops += float(mw[i].item()) * macs[i] * nu.nu_ado_scale
    budget.flops_spent_step = flops
    budget.ado_accumulated += flops

    # Broyden calls F(z)=z-phi(z) twice per outer iter (see `broyden_forward`): ~2 phi evals / iter
    phi_evals_per_broyden_iter = 2
    flops_broyden_estimate = flops * float(info.iterations) * float(phi_evals_per_broyden_iter)

    # FLOP-related key names: see `cts.eval.flops_contract.SOLVER_STATS_KEYS_TRANSITION`
    solver_stats: Dict[str, Any] = {
        "iterations": info.iterations,
        "residual_norm": info.residual_norm,
        "converged": info.converged,
        "flops_used": flops,
        "flops_inner_once": flops,
        "flops_broyden_estimate": flops_broyden_estimate,
        "phi_evals_per_broyden_iter": phi_evals_per_broyden_iter,
    }

    if not info.converged:
        return TransitionResult(
            child_text=None,
            z_star_child=z_star,
            solver_stats=solver_stats,
            prune=True,
            budget=budget,
        )

    if budget.ado_accumulated > tau_flops_budget:
        solver_stats["act_halt"] = True

    if hasattr(backbone, "decode_from_z_star"):
        try:
            dec = getattr(backbone, "decode_from_z_star")
            sig = inspect.signature(dec)
            if "max_new_tokens" in sig.parameters:
                child_text = dec(z_star, max_new_tokens=max_decode_tokens)
            else:
                child_text = dec(z_star)
        except Exception:
            flat = z_star.detach().cpu().reshape(-1)[:16]
            h = int(torch.sum(flat * 1e4).item()) & 0xFFFFFFFF
            child_text = f"<step branch={branch_index} h={h}>"
    else:
        flat = z_star.detach().cpu().reshape(-1)[:16]
        h = int(torch.sum(flat * 1e4).item()) & 0xFFFFFFFF
        child_text = f"<step branch={branch_index} h={h}>"

    return TransitionResult(
        child_text=child_text,
        z_star_child=z_star,
        solver_stats=solver_stats,
        prune=False,
        budget=budget,
    )
