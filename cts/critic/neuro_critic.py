"""Value head on z* (paper: anchored to universal latent)."""

from __future__ import annotations

import torch
import torch.nn as nn


class NeuroCritic(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, z_dim), nn.Tanh(), nn.Linear(z_dim, 1))

    def forward(self, z_star_flat: torch.Tensor) -> torch.Tensor:
        return self.net(z_star_flat)
