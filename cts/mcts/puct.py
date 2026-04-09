"""PUCT selection (paper Eq. 4 vs AlphaZero-style)."""

from __future__ import annotations

import math
from typing import List, Literal

PUCTVariant = Literal["paper", "alphazero"]


def puct_score(
    variant: PUCTVariant,
    nu_5ht: float,
    prior: float,
    n_parent: int,
    n_sa: int,
    q_sa: float,
    c_puct: float = 1.0,
) -> float:
    """
    Paper (CTS): U(s,a) = nu_5ht * P(s,a) * sqrt(N(s)) / (1 + N(s,a))  (exploration term only;
    add Q in tree policy when selecting argmax Q+U).
    AlphaZero: Q + c_puct * P * sqrt(N_parent) / (1 + N_sa)
    """
    if n_parent < 0 or n_sa < 0:
        raise ValueError("visit counts must be non-negative")
    if variant == "paper":
        exploration = nu_5ht * prior * math.sqrt(n_parent) / (1.0 + n_sa)
        return q_sa + exploration
    # alphazero-style combined
    exploration = c_puct * prior * math.sqrt(n_parent) / (1.0 + n_sa)
    return q_sa + exploration


def select_action(
    variant: PUCTVariant,
    nu_5ht: float,
    priors: List[float],
    ns: List[int],
    qs: List[float],
    n_parent: int,
) -> int:
    scores = [
        puct_score(variant, nu_5ht, priors[a], n_parent, ns[a], float(qs[a]))
        for a in range(len(priors))
    ]
    return int(max(range(len(scores)), key=lambda i: scores[i]))
