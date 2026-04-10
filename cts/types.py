"""Core CTS datatypes aligned with paper §2.3: ν = [νval, νexpl, νtol, νtemp, νact]."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class NuVector:
    """Meta-policy output per step (paper §2.3).

    Paper naming: νval (state value), νexpl (exploration rate),
    νtol (solver tolerance), νtemp (routing temperature), νact (ACT halting).
    """

    nu_val: float = 1.0
    nu_expl: float = 1.0
    nu_tol: float = 0.5  # maps to Broyden tol in [tol_min, tol_max]
    nu_temp: float = 1.0
    nu_act: float = 1.0

    # Backward-compatible aliases (legacy neurotransmitter names)
    @property
    def nu_da(self) -> float:
        return self.nu_val

    @property
    def nu_5ht(self) -> float:
        return self.nu_expl

    @property
    def nu_ne(self) -> float:
        return self.nu_tol

    @property
    def nu_ach(self) -> float:
        return self.nu_temp

    @property
    def nu_ado_scale(self) -> float:
        return self.nu_act


@dataclass
class RuntimeBudgetState:
    """Environment-held compute accumulation (paper §4.3 ACT)."""

    mac_accumulated: float = 0.0
    terminal_depth: int = 0
    flops_spent_step: float = 0.0
    wall_clock_ms_step: float = 0.0

    # Backward-compatible alias
    @property
    def ado_accumulated(self) -> float:
        return self.mac_accumulated

    def clone(self) -> "RuntimeBudgetState":
        return RuntimeBudgetState(
            mac_accumulated=self.mac_accumulated,
            terminal_depth=self.terminal_depth,
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
    faiss_retrieved: Optional[torch.Tensor] = None


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
