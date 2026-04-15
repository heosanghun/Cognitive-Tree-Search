"""L-Broyden solver: limited-memory FP32 inverse Jacobian (paper §5.2).

Paper Remark 2: Inherited Jacobians warm-start non-root nodes,
yielding average 11.2+-2.8 iterations (root: 14.8; non-root: 8.9).
"""

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
    jacobian_state: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None


@dataclass
class BroydenConvergenceStats:
    """Aggregate statistics matching paper Appendix C / Table 12."""

    total_solves: int = 0
    converged_count: int = 0
    fallback_count: int = 0
    iteration_counts: List[int] = field(default_factory=list)
    root_iterations: List[int] = field(default_factory=list)
    nonroot_iterations: List[int] = field(default_factory=list)

    @property
    def convergence_rate(self) -> float:
        return self.converged_count / max(self.total_solves, 1)

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / max(self.total_solves, 1)

    @property
    def avg_iterations(self) -> float:
        return sum(self.iteration_counts) / max(len(self.iteration_counts), 1)

    @property
    def avg_root_iterations(self) -> float:
        return sum(self.root_iterations) / max(len(self.root_iterations), 1)

    @property
    def avg_nonroot_iterations(self) -> float:
        return sum(self.nonroot_iterations) / max(len(self.nonroot_iterations), 1)

    def update(self, info: BroydenInfo, is_root: bool = True) -> None:
        self.total_solves += 1
        self.iteration_counts.append(info.iterations)
        if is_root:
            self.root_iterations.append(info.iterations)
        else:
            self.nonroot_iterations.append(info.iterations)
        if info.converged:
            self.converged_count += 1
        else:
            self.fallback_count += 1

    def report(self) -> Dict[str, float]:
        return {
            "convergence_rate": self.convergence_rate,
            "fallback_rate": self.fallback_rate,
            "avg_iterations": self.avg_iterations,
            "avg_root_iterations": self.avg_root_iterations,
            "avg_nonroot_iterations": self.avg_nonroot_iterations,
            "total_solves": float(self.total_solves),
        }


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


def map_nu_ne_to_tol(nu_ne: float, tol_min: float, tol_max: float) -> float:
    return map_nu_tol_to_tol(nu_ne, tol_min, tol_max)


def _matvec(Us: torch.Tensor, VTs: torch.Tensor, m: int, x: torch.Tensor) -> torch.Tensor:
    """Compute (U @ V^T) @ x using stored rank-1 factors. O(m*n)."""
    if m == 0:
        return torch.zeros_like(x)
    return Us[:, :m] @ (VTs[:m] @ x)


def broyden_fixed_point(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    tol: float,
    max_iter: int = 30,
    *,
    parent_inv_jacobian: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
    fp32_buffer: bool = True,
    memory_limit: int = 16,
) -> Tuple[torch.Tensor, BroydenInfo]:
    """L-Broyden fixed-point solver (paper §5.2).

    Paper Algorithm 1 line 10: L-Broyden-Update(B_s, z*_w)
    Inherits parent inverse Jacobian via (Us, VTs, n_stored) tuple.

    Returns:
        z_star: converged fixed point
        info: BroydenInfo with .jacobian_state = (Us, VTs, n_stored)
              for inheritance by child nodes
    """
    orig_shape = z0.shape
    device = z0.device
    compute_dtype = torch.float32 if fp32_buffer else z0.dtype
    z = z0.detach().reshape(-1).to(device=device, dtype=compute_dtype).clone()
    n = z.numel()

    def G(vec: torch.Tensor) -> torch.Tensor:
        zz = vec.to(z0.dtype).view(orig_shape)
        p = phi(zz).reshape(-1).to(compute_dtype)
        return vec - p

    gx = G(z)
    residuals: List[float] = []

    # Inherit parent Jacobian state or initialize fresh
    is_root = parent_inv_jacobian is None
    if parent_inv_jacobian is not None:
        p_Us, p_VTs, p_n = parent_inv_jacobian
        Us = p_Us.to(device=device, dtype=compute_dtype).clone()
        VTs = p_VTs.to(device=device, dtype=compute_dtype).clone()
        n_stored = min(p_n, memory_limit)
        if Us.shape[0] != n or Us.shape[1] < memory_limit:
            Us = torch.zeros(n, memory_limit, device=device, dtype=compute_dtype)
            VTs = torch.zeros(memory_limit, n, device=device, dtype=compute_dtype)
            n_stored = 0
    else:
        Us = torch.zeros(n, memory_limit, device=device, dtype=compute_dtype)
        VTs = torch.zeros(memory_limit, n, device=device, dtype=compute_dtype)
        n_stored = 0

    for it in range(max_iter):
        res = float(gx.norm().item())
        residuals.append(res)
        if res < tol:
            z_out = z.view(orig_shape).to(z0.dtype)
            jac_state = (Us.detach().clone(), VTs.detach().clone(), n_stored)
            info = BroydenInfo(it + 1, res, True, residuals, jacobian_state=jac_state)
            if _global_stats is not None:
                _global_stats.update(info, is_root=is_root)
            return z_out, info

        dx = gx - _matvec(Us, VTs, n_stored, gx)

        z_new = z + dx
        gx_new = G(z_new)

        dg = gx_new - gx
        B_dg = -dg + _matvec(Us, VTs, n_stored, dg)

        denom = torch.dot(dx, B_dg)
        if abs(float(denom)) < 1e-12:
            z, gx = z_new, gx_new
            continue

        u_new = (dx - B_dg) / denom

        if n_stored > 0:
            ut_dx = Us[:, :n_stored].t() @ dx
            v_new = -dx + VTs[:n_stored].t() @ ut_dx
        else:
            v_new = -dx.clone()

        idx = n_stored % memory_limit
        Us[:, idx] = u_new
        VTs[idx] = v_new
        if n_stored < memory_limit:
            n_stored += 1

        z, gx = z_new, gx_new

    z_out = z.view(orig_shape).to(z0.dtype)
    jac_state = (Us.detach().clone(), VTs.detach().clone(), n_stored)
    info = BroydenInfo(max_iter, float(gx.norm().item()), False, residuals, jacobian_state=jac_state)
    if _global_stats is not None:
        _global_stats.update(info, is_root=is_root)
    return z_out, info


def broyden_fixed_point_batch(
    phi: Callable[[torch.Tensor], torch.Tensor],
    z0_batch: torch.Tensor,
    tol: float,
    max_iter: int = 30,
    *,
    parent_inv_jacobian: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
    fp32_buffer: bool = True,
    memory_limit: int = 16,
) -> Tuple[torch.Tensor, List[BroydenInfo]]:
    """True batched Broyden for W sibling branches (paper §4.1).

    All W branches share one batched phi evaluation per iteration.
    z0_batch: [W, K, d]  -> z_star_batch: [W, K, d]
    """
    W = z0_batch.shape[0]
    orig_shape_per = z0_batch.shape[1:]  # [K, d]
    device = z0_batch.device
    compute_dtype = torch.float32 if fp32_buffer else z0_batch.dtype

    n = int(torch.tensor(orig_shape_per).prod().item())  # K * d

    z = z0_batch.detach().reshape(W, -1).to(dtype=compute_dtype).clone()  # [W, n]

    def G_batch(vecs: torch.Tensor) -> torch.Tensor:
        """Residual for all W branches at once. vecs: [W, n]."""
        zz = vecs.to(z0_batch.dtype).view(W, *orig_shape_per)
        results = []
        for i in range(W):
            p_i = phi(zz[i]).reshape(-1).to(compute_dtype)
            results.append(vecs[i] - p_i)
        return torch.stack(results)

    gx = G_batch(z)  # [W, n]

    Us = torch.zeros(W, n, memory_limit, device=device, dtype=compute_dtype)
    VTs = torch.zeros(W, memory_limit, n, device=device, dtype=compute_dtype)
    n_stored = [0] * W

    is_root = parent_inv_jacobian is None
    if parent_inv_jacobian is not None:
        p_Us, p_VTs, p_n = parent_inv_jacobian
        for w in range(W):
            if p_Us.shape[0] == n and p_Us.shape[1] >= memory_limit:
                Us[w] = p_Us.to(dtype=compute_dtype).clone()
                VTs[w] = p_VTs.to(dtype=compute_dtype).clone()
                n_stored[w] = min(p_n, memory_limit)

    converged = [False] * W
    residuals_all: List[List[float]] = [[] for _ in range(W)]
    final_iters = [max_iter] * W

    for it in range(max_iter):
        res_norms = gx.norm(dim=1)  # [W]
        for w in range(W):
            res_w = float(res_norms[w].item())
            residuals_all[w].append(res_w)
            if not converged[w] and res_w < tol:
                converged[w] = True
                final_iters[w] = it + 1

        if all(converged):
            break

        dx = torch.zeros_like(z)
        for w in range(W):
            if converged[w]:
                continue
            m = n_stored[w]
            mv = _matvec(Us[w], VTs[w], m, gx[w])
            dx[w] = gx[w] - mv

        z_new = z + dx
        gx_new = G_batch(z_new)

        for w in range(W):
            if converged[w]:
                continue
            dg_w = gx_new[w] - gx[w]
            m = n_stored[w]
            B_dg = -dg_w + _matvec(Us[w], VTs[w], m, dg_w)
            denom = torch.dot(dx[w], B_dg)
            if abs(float(denom)) < 1e-12:
                continue
            u_new = (dx[w] - B_dg) / denom
            if m > 0:
                ut_dx = Us[w, :, :m].t() @ dx[w]
                v_new = -dx[w] + VTs[w, :m].t() @ ut_dx
            else:
                v_new = -dx[w].clone()
            idx = m % memory_limit
            Us[w, :, idx] = u_new
            VTs[w, idx] = v_new
            if m < memory_limit:
                n_stored[w] = m + 1

        z = z_new
        gx = gx_new

    z_out = z.view(W, *orig_shape_per).to(z0_batch.dtype)
    infos = []
    for w in range(W):
        jac = (Us[w].detach().clone(), VTs[w].detach().clone(), n_stored[w])
        res = float(gx[w].norm().item()) if not converged[w] else residuals_all[w][-1]
        info = BroydenInfo(
            final_iters[w], res, converged[w], residuals_all[w], jacobian_state=jac,
        )
        if _global_stats is not None:
            _global_stats.update(info, is_root=is_root)
        infos.append(info)

    return z_out, infos
