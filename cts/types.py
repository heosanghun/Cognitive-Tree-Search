"""Core CTS datatypes: neuromodulator policy output vs runtime budget state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class NuVector:
    """Policy output per step (paper ν). Not the accumulated Adenosine budget."""

    nu_da: float = 1.0
    nu_5ht: float = 1.0
    nu_ne: float = 0.5  # maps to Broyden tol in [tol_min, tol_max]
    nu_ach: float = 1.0
    nu_ado_scale: float = 1.0


@dataclass
class RuntimeBudgetState:
    """Environment-held compute accumulation (Adenosine / FLOPs)."""

    ado_accumulated: float = 0.0
    flops_spent_step: float = 0.0
    wall_clock_ms_step: float = 0.0

    def clone(self) -> "RuntimeBudgetState":
        return RuntimeBudgetState(
            ado_accumulated=self.ado_accumulated,
            flops_spent_step=self.flops_spent_step,
            wall_clock_ms_step=self.wall_clock_ms_step,
        )


@dataclass
class TransitionResult:
    child_text: Optional[str]
    z_star_child: torch.Tensor
    solver_stats: Dict[str, Any]
    prune: bool
    budget: RuntimeBudgetState


@dataclass
class MCTSStats:
    visit_count: int = 0
    q_value: float = 0.0
    prior: float = 0.0


@dataclass
class TreeNode:
    """Discrete search node with hard-anchored text state s_t (paper)."""

    text_state: str
    z_star: Optional[torch.Tensor]
    depth: int
    parent_id: Optional[int]
    node_id: int
    children_ids: List[int] = field(default_factory=list)
    mcts_N: int = 0
    mcts_W: int = 3
    mcts_Q: List[float] = field(default_factory=list)
    mcts_prior: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.mcts_Q:
            self.mcts_Q = [0.0] * self.mcts_W
        if not self.mcts_prior:
            self.mcts_prior = [1.0 / self.mcts_W] * self.mcts_W
