"""CTS transition API: DEQ fixed point + FAISS context + batch + ACT (paper §4)."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from cts.backbone.protocol import BaseCTSBackbone
from cts.deq.broyden_forward import (
    broyden_fixed_point,
    broyden_fixed_point_batch,
    map_nu_tol_to_tol,
)
from cts.latent.bottleneck import add_exploration_noise, init_z0
from cts.latent.faiss_context import LatentContextWindow, prepend_soft_prefix
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
    K: int = 64,
    d: Optional[int] = None,
    top_k: int = 3,
    broyden_max_iter: int = 30,
    broyden_tol_min: float = 1e-4,
    broyden_tol_max: float = 1e-2,
    tau_flops_budget: float = 1e14,
    generator: Optional[torch.Generator] = None,
    routing_mode: str = "sparse",
    max_decode_tokens: int = 1,
    faiss_context: Optional[LatentContextWindow] = None,
    fp32_buffer: bool = True,
    parent_inv_jacobian: Optional[torch.Tensor] = None,
) -> TransitionResult:
    """
    One KV-cache-free transition using DEQ (paper §4.2).
    Supports FAISS Latent Context Window (§4.4) and L-Broyden FP32 (§5.2).
    Parent inverse Jacobian inheritance accelerates convergence (§5.2).
    """
    if isinstance(backbone, nn.Module):
        device = next(backbone.parameters()).device
    else:
        device = torch.device("cpu")
    d = d or backbone.hidden_size
    gen = generator or torch.Generator(device=device)
    gen.manual_seed(2026 + branch_index * 31 + (len(parent_text) % 997))

    z0 = init_z0(K, d, device, gen)
    z0 = add_exploration_noise(z0, nu.nu_expl, gen)

    context = backbone.encode_context(parent_text)
    if context.dim() == 1:
        context = context.unsqueeze(0)
    context = context.to(device=device, dtype=torch.float32)

    # FAISS Latent Space Context Window (paper §4.4)
    faiss_retrieved = None
    if faiss_context is not None:
        faiss_retrieved_raw = faiss_context.retrieve(z0, k=3)
        if faiss_retrieved_raw is not None:
            context = prepend_soft_prefix(context, faiss_retrieved_raw)
            faiss_retrieved = faiss_retrieved_raw

    if hasattr(backbone, "routing_matrix"):
        w_g = backbone.routing_matrix().to(device=device, dtype=torch.float32)
    else:
        w_g = torch.randn(19, d, device=device, dtype=torch.float32) * 0.02
    macs = _load_mac_lut()

    def phi(zz: torch.Tensor) -> torch.Tensor:
        alpha = routing_weights(zz, w_g, nu.nu_temp)
        if routing_mode == "dense":
            mw = alpha
        else:
            mw = sparse_module_weights(alpha, top_k)
        extra: Dict[str, Any] = {"top_k": top_k}
        return backbone.deq_step(zz, context, mw, extra)

    tol = map_nu_tol_to_tol(nu.nu_tol, broyden_tol_min, broyden_tol_max)
    z_star, info = broyden_fixed_point(
        phi, z0, tol=tol, max_iter=broyden_max_iter, fp32_buffer=fp32_buffer,
        parent_inv_jacobian=parent_inv_jacobian,
    )

    budget = budget.clone()
    budget.terminal_depth += 1
    flops = 0.0
    with torch.no_grad():
        alpha = routing_weights(z_star, w_g, nu.nu_temp)
        mw = alpha if routing_mode == "dense" else sparse_module_weights(alpha, top_k)
        for i in range(19):
            flops += float(mw[i].item()) * macs[i] * nu.nu_act
    budget.flops_spent_step = flops
    budget.mac_accumulated += flops

    phi_evals_per_broyden_iter = 2
    flops_broyden_estimate = (
        flops * float(info.iterations) * float(phi_evals_per_broyden_iter)
    )

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
            faiss_retrieved=faiss_retrieved,
        )

    if budget.mac_accumulated > tau_flops_budget * nu.nu_act:
        solver_stats["act_halt"] = True

    # Add z* to FAISS context window
    if faiss_context is not None:
        faiss_context.add(z_star)

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
        faiss_retrieved=faiss_retrieved,
    )


def transition_batch(
    parent_text: str,
    nu: NuVector,
    budget: RuntimeBudgetState,
    backbone: BaseCTSBackbone,
    *,
    W: int = 3,
    K: int = 64,
    d: Optional[int] = None,
    top_k: int = 3,
    broyden_max_iter: int = 30,
    broyden_tol_min: float = 1e-4,
    broyden_tol_max: float = 1e-2,
    tau_flops_budget: float = 1e14,
    routing_mode: str = "sparse",
    max_decode_tokens: int = 1,
    faiss_context: Optional[LatentContextWindow] = None,
    fp32_buffer: bool = True,
    parent_inv_jacobian: Optional[torch.Tensor] = None,
) -> List[TransitionResult]:
    """Paper §4.1: parallel batch DEQ for W sibling branches.

    W branches are evaluated as a single parallel batch during the DEQ solve,
    maintaining ~25ms latency irrespective of branching width (paper §4.1).
    """
    if isinstance(backbone, nn.Module):
        device = next(backbone.parameters()).device
    else:
        device = torch.device("cpu")
    d_actual = d or backbone.hidden_size

    context = backbone.encode_context(parent_text)
    if context.dim() == 1:
        context = context.unsqueeze(0)
    context = context.to(device=device, dtype=torch.float32)

    if faiss_context is not None:
        faiss_retrieved_raw = faiss_context.retrieve(
            init_z0(K, d_actual, device, torch.Generator(device=device)), k=3
        )
        if faiss_retrieved_raw is not None:
            context = prepend_soft_prefix(context, faiss_retrieved_raw)

    if hasattr(backbone, "routing_matrix"):
        w_g = backbone.routing_matrix().to(device=device, dtype=torch.float32)
    else:
        w_g = torch.randn(19, d_actual, device=device, dtype=torch.float32) * 0.02
    macs = _load_mac_lut()
    tol = map_nu_tol_to_tol(nu.nu_tol, broyden_tol_min, broyden_tol_max)

    z0_list = []
    for bi in range(W):
        gen = torch.Generator(device=device)
        gen.manual_seed(2026 + bi * 31 + (len(parent_text) % 997))
        z0 = init_z0(K, d_actual, device, gen)
        z0 = add_exploration_noise(z0, nu.nu_expl, gen)
        z0_list.append(z0)

    def phi(zz: torch.Tensor) -> torch.Tensor:
        alpha = routing_weights(zz, w_g, nu.nu_temp)
        mw = alpha if routing_mode == "dense" else sparse_module_weights(alpha, top_k)
        return backbone.deq_step(zz, context, mw, {"top_k": top_k})

    batch_results = broyden_fixed_point_batch(
        phi, z0_list, tol=tol, max_iter=broyden_max_iter,
        fp32_buffer=fp32_buffer, parent_inv_jacobian=parent_inv_jacobian,
    )

    results: List[TransitionResult] = []
    for bi, (z_star, info) in enumerate(batch_results):
        b = budget.clone()
        b.terminal_depth += 1
        flops = 0.0
        with torch.no_grad():
            alpha = routing_weights(z_star, w_g, nu.nu_temp)
            mw = alpha if routing_mode == "dense" else sparse_module_weights(alpha, top_k)
            for i in range(19):
                flops += float(mw[i].item()) * macs[i] * nu.nu_act
        b.flops_spent_step = flops
        b.mac_accumulated += flops

        solver_stats = {
            "iterations": info.iterations,
            "residual_norm": info.residual_norm,
            "converged": info.converged,
            "flops_used": flops,
        }

        if not info.converged:
            results.append(TransitionResult(
                child_text=None, z_star_child=z_star, solver_stats=solver_stats,
                prune=True, budget=b,
            ))
            continue

        if b.mac_accumulated > tau_flops_budget * nu.nu_act:
            solver_stats["act_halt"] = True

        if faiss_context is not None:
            faiss_context.add(z_star)

        if hasattr(backbone, "decode_from_z_star"):
            try:
                import inspect as _insp
                sig = _insp.signature(backbone.decode_from_z_star)
                if "max_new_tokens" in sig.parameters:
                    child_text = backbone.decode_from_z_star(z_star, max_new_tokens=max_decode_tokens)
                else:
                    child_text = backbone.decode_from_z_star(z_star)
            except Exception:
                child_text = f"<step branch={bi}>"
        else:
            child_text = f"<step branch={bi}>"

        results.append(TransitionResult(
            child_text=child_text, z_star_child=z_star, solver_stats=solver_stats,
            prune=False, budget=b,
        ))

    return results
