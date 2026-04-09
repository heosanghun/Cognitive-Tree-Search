"""Lightweight meta-policy: ν vector + branch priors (paper Sec 4.1)."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from cts.types import NuVector


class MetaPolicy(nn.Module):
    """Placeholder MLP; swap for LoRA-on-Gemma in full training."""

    def __init__(self, text_dim: int = 64, hidden: int = 32, W: int = 3) -> None:
        super().__init__()
        self.W = W
        self.enc = nn.Linear(text_dim, hidden)
        self.head_nu = nn.Linear(hidden, 5)
        self.head_prior = nn.Linear(hidden, W)
        self.act = nn.ReLU()

    def logits_and_nu(self, text_features: torch.Tensor) -> Tuple[NuVector, torch.Tensor]:
        """Same as `forward` but returns branch logits tensor (for PPO log-prob)."""
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        h = self.act(self.enc(text_features))
        raw = self.head_nu(h).squeeze(0)
        nu = NuVector(
            nu_da=float(torch.nn.functional.softplus(raw[0]).item()),
            nu_5ht=float(torch.nn.functional.softplus(raw[1]).item()) + 0.5,
            nu_ne=float(torch.sigmoid(raw[2]).item()),
            nu_ach=float(torch.nn.functional.softplus(raw[3]).item()) + 0.5,
            nu_ado_scale=float(torch.nn.functional.softplus(raw[4]).item()) + 0.5,
        )
        logits = self.head_prior(h).squeeze(0)
        return nu, logits

    def forward(self, text_features: torch.Tensor) -> Tuple[NuVector, List[float]]:
        nu, logits = self.logits_and_nu(text_features)
        p = torch.softmax(logits, dim=-1).tolist()
        return nu, p
