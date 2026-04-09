"""Total reward: sum nu_DA + terminal - lambda * nu_Ado (paper Eq. 6)."""

from __future__ import annotations


def total_reward_stub(process_term: float, correct: bool, ado_accumulated: float, lam: float) -> float:
    term = 1.0 if correct else 0.0
    return process_term + term - lam * ado_accumulated
