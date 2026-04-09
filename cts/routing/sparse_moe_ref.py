"""Sparse module routing (paper Eq. 5) — reference PyTorch."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def routing_weights(
    z: torch.Tensor,
    w_g: torch.Tensor,
    nu_ach: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """alpha = softmax( (W_g @ pool(z)) / nu_ach ), z: [K, d], w_g: [n_mod, d]"""
    pooled = z.mean(dim=0)
    logits = (w_g @ pooled) / max(float(nu_ach), eps)
    return F.softmax(logits, dim=-1)


def top_k_mask(alpha: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, alpha.numel())
    _, idx = torch.topk(alpha, k)
    m = torch.zeros_like(alpha)
    m[idx] = 1.0
    return m


def sparse_module_weights(alpha: torch.Tensor, k: int) -> torch.Tensor:
    """Renormalize over top-k for weighted sum (reference path)."""
    m = top_k_mask(alpha, k)
    w = alpha * m
    return w / (w.sum().clamp_min(1e-8))
