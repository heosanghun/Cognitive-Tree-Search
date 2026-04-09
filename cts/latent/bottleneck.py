"""Soft thought bottleneck z0 init + exploration noise (paper Sec 4.1)."""

from __future__ import annotations

import torch


def init_z0(K: int, d: int, device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
    g = generator or torch.Generator(device=device)
    z = torch.randn(K, d, generator=g, dtype=torch.float32, device=device) * 0.02
    return z


def add_serotonin_noise(z0: torch.Tensor, nu_5ht: float, generator: torch.Generator) -> torch.Tensor:
    sigma = 0.05 * float(nu_5ht)
    noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device, generator=generator)
    return z0 + sigma * noise
