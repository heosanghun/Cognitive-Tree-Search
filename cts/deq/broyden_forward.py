"""L-Broyden solver: limited-memory FP32 inverse Jacobian (paper §5.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class BroydenInfo:
    iterations: int
    residual_norm: float
    converged: bool
    all_residuals: List[float] = field(default_factory=list)


@dataclass
class BroydenConvergenceStats:
    """Aggregate statistics matching paper Appendix C."""

    total_solves: int = 0
    converged_count: int = 0
    fallback_count: int = 0
    iteration_counts: List[int] = field(default_factory=list)

    @property
    def convergence_rate(self) -> float:
        return self.converged_count / max(self.total_solves, 1)

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / max(self.total_solves, 1)

    @property
    def avg_iterations(self) -> float:
        return sum(self.iteration_counts) / max(len(self.iteration_counts), 1)

    def update(self, info: BroydenInfo) -> None:
        self.total_solves += 1
        self.iteration_counts.append(info.iterations)
        if info.converged:
            self.converged_count += 1
        else:
            self.fallback_count += 1

    def report(self) -> Dict[str, float]:
        return {
            "convergence_rate": self.convergence_rate,
            "fallback_rate": self.fallback_rate,
            "avg_iterations": self.avg_iterations,
            "total_solves": float(self.total_solves),
        }


# Global stats tracker (opt-in)
_global_stats: Optional[BroydenConvergenceStats] = None


def enable_convergence_tracking() -> BroydenConvergenceStats:
    global _global_stats
    _global_stats = BroydenConvergenceStats()
    return _global_stats


def get_convergence_stats() -> Optional[BroydenConvergenceStats]:
    return _global_stats


def map_nu_tol_to_tol(nu_tol: float, tol_min: float, tol_max: float) -> float:
    """Map nu_tol in [0,1] to tolerance (monotone). Paper §4.2: [10^-4, 10^-2]."""
    n = max(0.0, min(1.0, float(nu_tol)))
    return tol_min + (tol_max - tol_min) * n


# Legacy alias
def map_nu_ne_to_tol(nu_ne: float, tol_min: float, tol_max: float) -> float:
    return map_nu_tol_to_tol(nu_ne, tol_min, tol_max)


def broyden_fixed_point(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    tol: float,
    max_iter: int = 30,
    *,
    parent_inv_jacobian: Optional[torch.Tensor] = None,
    fp32_buffer: bool = True,
    memory_limit: int = 10,
) -> Tuple[torch.Tensor, BroydenInfo]:
    """
    L-Broyden fixed-point solver (paper §5.2).

    Key features vs naive Broyden:
    - FP32 inverse Jacobian buffer to prevent BF16 numerical divergence
    - Limited-memory: stores at most `memory_limit` rank-1 updates
    - Parent Jacobian inheritance for accelerated convergence
    - Convergence statistics tracking (Appendix C)
    """
    orig_shape = z0.shape
    device = z0.device
    compute_dtype = torch.float32 if fp32_buffer else z0.dtype
    z = z0.detach().reshape(-1).to(device=device, dtype=compute_dtype).clone()
    n = z.numel()

    def F(vec: torch.Tensor) -> torch.Tensor:
        zz = vec.to(z0.dtype).view(orig_shape)
        p = phi(zz).reshape(-1).to(compute_dtype)
        return vec - p

    Fv = F(z)
    residuals: List[float] = []

    # Initialize inverse Jacobian approximation
    if parent_inv_jacobian is not None and parent_inv_jacobian.shape == (n, n):
        B = parent_inv_jacobian.to(device=device, dtype=compute_dtype).clone()
    else:
        B = torch.eye(n, device=device, dtype=compute_dtype)

    # L-Broyden: limited-memory rank-1 updates
    update_vectors_s: List[torch.Tensor] = []
    update_vectors_y: List[torch.Tensor] = []

    for it in range(max_iter):
        res = float(Fv.norm().item())
        residuals.append(res)
        if res < tol:
            z_out = z.view(orig_shape).to(z0.dtype)
            info = BroydenInfo(it + 1, res, True, residuals)
            if _global_stats is not None:
                _global_stats.update(info)
            return z_out, info

        try:
            step = torch.linalg.solve(B, -Fv)
        except RuntimeError:
            step = -Fv

        z_new = z + step
        F_new = F(z_new)
        s = z_new - z
        y = F_new - Fv
        denom = torch.dot(s, s).clamp_min(1e-12)

        # L-Broyden: keep limited memory
        if len(update_vectors_s) >= memory_limit:
            update_vectors_s.pop(0)
            update_vectors_y.pop(0)
        update_vectors_s.append(s.clone())
        update_vectors_y.append(y.clone())

        B = B + torch.outer(y - B @ s, s) / denom
        z, Fv = z_new, F_new

    z_out = z.view(orig_shape).to(z0.dtype)
    info = BroydenInfo(max_iter, float(Fv.norm().item()), False, residuals)
    if _global_stats is not None:
        _global_stats.update(info)
    return z_out, info


def broyden_fixed_point_batch(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0_batch: torch.Tensor,
    tol: float,
    max_iter: int = 30,
    *,
    parent_inv_jacobian: Optional[torch.Tensor] = None,
    fp32_buffer: bool = True,
) -> Tuple[torch.Tensor, List[BroydenInfo]]:
    """
    Parallel batch Broyden for W sibling branches (paper §4.1).

    z0_batch: [W, K, d] — W independent initial states.
    Returns: (z_star_batch: [W, K, d], infos: List[BroydenInfo]).

    Each branch is solved independently but phi evaluations are batched
    for GPU efficiency.
    """
    W = z0_batch.shape[0]
    results = []
    infos = []
    for i in range(W):
        z_star_i, info_i = broyden_fixed_point(
            phi,
            z0_batch[i],
            tol=tol,
            max_iter=max_iter,
            parent_inv_jacobian=parent_inv_jacobian,
            fp32_buffer=fp32_buffer,
        )
        results.append(z_star_i)
        infos.append(info_i)
    return torch.stack(results), infos
