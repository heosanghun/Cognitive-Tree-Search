"""Broyden-style solve for z* = phi(z*) (paper: FP32 internal loop)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch


@dataclass
class BroydenInfo:
    iterations: int
    residual_norm: float
    converged: bool


def map_nu_ne_to_tol(nu_ne: float, tol_min: float, tol_max: float) -> float:
    """Map nu_ne in [0,1] to tolerance (monotone)."""
    n = max(0.0, min(1.0, float(nu_ne)))
    return tol_min + (tol_max - tol_min) * n


def broyden_fixed_point(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    tol: float,
    max_iter: int = 30,
) -> Tuple[torch.Tensor, BroydenInfo]:
    """
    Find z such that F(z)=z-phi(z)=0 using dense Broyden updates on Jacobian approx.
    Runs in float32 for numerical stability (paper appendix).
    """
    orig_shape = z0.shape
    device = z0.device
    z = z0.detach().reshape(-1).to(device=device, dtype=torch.float32).clone()
    n = z.numel()

    def F(vec: torch.Tensor) -> torch.Tensor:
        zz = vec.view(orig_shape).to(z0.dtype)
        p = phi(zz).reshape(-1).to(torch.float32)
        return vec - p

    Fv = F(z)
    B = torch.eye(n, device=device, dtype=torch.float32)

    for it in range(max_iter):
        res = float(Fv.norm().item())
        if res < tol:
            z_out = z.view(orig_shape).to(z0.dtype)
            return z_out, BroydenInfo(it + 1, res, True)
        try:
            step = torch.linalg.solve(B, -Fv)
        except RuntimeError:
            step = -Fv
        z_new = z + step
        F_new = F(z_new)
        s = z_new - z
        y = F_new - Fv
        denom = torch.dot(s, s).clamp_min(1e-12)
        B = B + torch.outer(y - B @ s, s) / denom
        z, Fv = z_new, F_new

    z_out = z.view(orig_shape).to(z0.dtype)
    return z_out, BroydenInfo(max_iter, float(Fv.norm().item()), False)
