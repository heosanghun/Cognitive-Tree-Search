"""GAE and PPO clipped surrogate (Stage 2)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float = 0.95,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    Generalized Advantage Estimation. `values` length should match `rewards`;
    bootstrap `values[t+1]` at terminal when `dones[t]`.
    """
    T = len(rewards)
    if len(values) != T or len(dones) != T:
        raise ValueError("rewards, values, dones must have same length")
    adv = [0.0] * T
    last_gae = 0.0
    for t in range(T - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < T else 0.0
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    returns = [adv[i] + values[i] for i in range(T)]
    return adv, returns


def ppo_clipped_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip: float = 0.2,
) -> torch.Tensor:
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    return -torch.mean(torch.min(unclipped, clipped))


def value_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)
