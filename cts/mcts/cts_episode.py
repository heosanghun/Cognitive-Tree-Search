"""Algorithm 1: Cognitive Tree Search (CTS) — Single Episode.

Paper-aligned full MCTS episode loop with:
  - PUCT selection across the full tree (Eq. 2)
  - MetaPolicy nu sampling (§4.1)
  - W parallel DEQ transitions with parent z* noise (line 6)
  - Neuro-Critic Q evaluation (line 12)
  - FAISS registration (line 12)
  - Tree backpropagation (line 14)
  - ACT halting (line 15)
  - Best-trajectory decoding via Wproj (line 18)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from cts.backbone.protocol import BaseCTSBackbone
from cts.critic.neuro_critic import NeuroCritic
from cts.deq.transition import transition
from cts.latent.bottleneck import init_z0
from cts.latent.faiss_context import LatentContextWindow
from cts.mcts.puct import PUCTVariant, select_action
from cts.mcts.tree import SearchTree
from cts.policy.meta_policy import MetaPolicy
from cts.types import NuVector, RuntimeBudgetState, TransitionResult, TreeNode


@dataclass
class CtsEpisodeResult:
    """Output of a single CTS episode."""
    answer: str
    best_z_star: Optional[torch.Tensor]
    tree: SearchTree
    total_mac: float
    total_iterations: int
    stats: Dict[str, Any] = field(default_factory=dict)


def _pool_z_star(z: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Mean-pool z* [K, d] -> [d] for MetaPolicy / Critic input."""
    if z is None:
        return None
    return z.detach().float().mean(dim=0)


def _backpropagate(tree: SearchTree, node_id: int, q_values: List[float]) -> None:
    """Backpropagate mean Q from children up to root (paper line 14)."""
    node = tree.nodes[node_id]
    if not q_values:
        return
    mean_q = sum(q_values) / len(q_values)

    cur_id: Optional[int] = node_id
    while cur_id is not None:
        cur = tree.nodes[cur_id]
        old_n = cur.mcts_N
        cur.mcts_N = old_n + 1
        if cur.parent_id is not None:
            parent = tree.nodes[cur.parent_id]
            child_idx = parent.children_ids.index(cur_id) if cur_id in parent.children_ids else 0
            if child_idx < len(parent.mcts_Q):
                old_q = parent.mcts_Q[child_idx]
                old_visits = max(1, old_n)
                parent.mcts_Q[child_idx] = (old_q * old_visits + mean_q) / (old_visits + 1)
        cur_id = cur.parent_id


def _select_leaf(tree: SearchTree, nu_expl: float, variant: PUCTVariant = "paper") -> int:
    """PUCT tree traversal: select a leaf node for expansion (paper line 3)."""
    cur = 0
    while tree.nodes[cur].children_ids:
        node = tree.nodes[cur]
        children = node.children_ids
        n_parent = max(1, node.mcts_N)

        best_score = float("-inf")
        best_child = children[0]
        for idx, cid in enumerate(children):
            child = tree.nodes[cid]
            prior = node.mcts_prior[idx] if idx < len(node.mcts_prior) else 1.0 / len(children)
            q = node.mcts_Q[idx] if idx < len(node.mcts_Q) else 0.0
            n_sa = child.mcts_N

            from cts.mcts.puct import puct_score
            score = puct_score(variant, nu_expl, prior, n_parent, n_sa, q)
            if score > best_score:
                best_score = score
                best_child = cid
        cur = best_child
    return cur


def cts_full_episode(
    prompt: str,
    *,
    backbone: BaseCTSBackbone,
    meta_policy: MetaPolicy,
    critic: NeuroCritic,
    W: int = 3,
    K: int = 64,
    tau_budget: float = 1e14,
    broyden_max_iter: int = 30,
    broyden_tol_min: float = 1e-4,
    broyden_tol_max: float = 1e-2,
    top_k: int = 3,
    puct_variant: PUCTVariant = "paper",
    faiss_context: Optional[LatentContextWindow] = None,
    max_decode_tokens: int = 64,
    routing_mode: str = "sparse",
    noise_sigma: float = 0.02,
    device: Optional[torch.device] = None,
) -> CtsEpisodeResult:
    """Algorithm 1: Cognitive Tree Search — Single Episode.

    Require: Prompt s0, budget tau, W, f_theta, pi_phi, V_psi, FAISS F
    Ensure: Decoded answer y_hat
    """
    if device is None:
        if hasattr(backbone, "parameters"):
            device = next(backbone.parameters()).device
        else:
            device = torch.device("cpu")

    d = backbone.hidden_size

    # Line 1: z*_0 <- FwdPass(s0); B0 <- 0.1*I; init T; MAC <- 0
    with torch.no_grad():
        context_0 = backbone.encode_context(prompt)
    if context_0.dim() == 1:
        context_0 = context_0.unsqueeze(0)

    z0_root = init_z0(K, d, device, torch.Generator(device=device).manual_seed(2026))

    tree = SearchTree()
    root_id = tree.new_node(prompt, z0_root, depth=0, parent_id=None, W=W)

    mac_accumulated = 0.0
    total_iterations = 0

    if faiss_context is not None:
        faiss_context.reset()

    # Line 2: while MAC < tau do
    while mac_accumulated < tau_budget:
        # Line 3: s <- PUCT(T, V_psi, nu_expl)
        leaf_id = _select_leaf(tree, nu_expl=1.0, variant=puct_variant)
        leaf = tree.nodes[leaf_id]

        # Line 4: nu_A <- pi_phi(z*_s)
        z_star_s = leaf.z_star
        z_pooled = _pool_z_star(z_star_s)
        if z_pooled is None:
            z_pooled = torch.zeros(d, device=device, dtype=torch.float32)

        with torch.no_grad():
            nu, priors = meta_policy(z_pooled.to(device))

        # Line 5: MAC += LUT[pi_phi] + LUT[V_psi]
        meta_mac = 0.002e14
        mac_accumulated += meta_mac

        # Line 3 (refined): use nu_expl from policy for PUCT in subsequent iterations
        if leaf.depth > 0:
            leaf_id = _select_leaf(tree, nu_expl=nu.nu_expl, variant=puct_variant)
            leaf = tree.nodes[leaf_id]
            z_star_s = leaf.z_star
            z_pooled = _pool_z_star(z_star_s)
            if z_pooled is None:
                z_pooled = torch.zeros(d, device=device, dtype=torch.float32)
            with torch.no_grad():
                nu, priors = meta_policy(z_pooled.to(device))

        # Line 6: t <- depth(s); {z_tilde_w} <- z*_s + epsilon_w
        t = leaf.depth

        # Line 7-11: for w = 1,...,W in parallel
        child_q_values: List[float] = []
        for w in range(W):
            budget_w = RuntimeBudgetState(mac_accumulated=mac_accumulated)
            r = transition(
                leaf.text_state,
                w,
                nu,
                budget_w,
                backbone,
                K=K,
                d=d,
                top_k=top_k,
                broyden_max_iter=broyden_max_iter,
                broyden_tol_min=broyden_tol_min,
                broyden_tol_max=broyden_tol_max,
                tau_flops_budget=tau_budget,
                routing_mode=routing_mode,
                max_decode_tokens=1,
                faiss_context=faiss_context if t >= 10 else None,
                parent_z_star=z_star_s,
                noise_sigma=noise_sigma,
            )

            iters = r.solver_stats.get("iterations", 0)
            total_iterations += iters
            step_mac = r.solver_stats.get("flops_broyden_estimate", r.solver_stats.get("flops_used", 0.0))
            mac_accumulated += step_mac

            # Line 12: Q_w <- V_psi(z*_w); F.add(z*_w); AddChild(T, z*_w, B_w)
            z_child = r.z_star_child
            z_child_pooled = _pool_z_star(z_child)
            if z_child_pooled is None:
                z_child_pooled = torch.zeros(d, device=device)

            with torch.no_grad():
                q_w = float(critic(z_child_pooled.unsqueeze(0).to(device)).item())

            if r.prune:
                q_w = 0.0

            child_q_values.append(q_w)

            child_text = r.child_text or f"<d={t+1} w={w}>"
            child_id = tree.new_node(
                child_text, z_child, depth=t + 1, parent_id=leaf_id, W=W,
            )

        # Update priors on the expanded node
        tree.nodes[leaf_id].mcts_prior = list(priors) if len(priors) == W else [1.0 / W] * W
        tree.nodes[leaf_id].mcts_Q = child_q_values[:W]

        # Line 14: BackProp(T, {Q_w})
        _backpropagate(tree, leaf_id, child_q_values)

        # Line 15-16: if MAC >= tau * nu_act then break
        if mac_accumulated >= tau_budget * nu.nu_act:
            break

    # Line 18: y_hat <- Decode(W_proj @ z*_best)
    best_id = 0
    best_q = float("-inf")
    for node in tree.nodes:
        if node.z_star is not None and node.depth > 0:
            z_p = _pool_z_star(node.z_star)
            if z_p is not None:
                with torch.no_grad():
                    v = float(critic(z_p.unsqueeze(0).to(device)).item())
                if v > best_q:
                    best_q = v
                    best_id = node.node_id

    best_z = tree.nodes[best_id].z_star

    answer = ""
    if best_z is not None and hasattr(backbone, "decode_from_z_star"):
        try:
            answer = backbone.decode_from_z_star(best_z, max_new_tokens=max_decode_tokens)
        except Exception:
            answer = tree.nodes[best_id].text_state

    return CtsEpisodeResult(
        answer=answer,
        best_z_star=best_z,
        tree=tree,
        total_mac=mac_accumulated,
        total_iterations=total_iterations,
        stats={
            "tree_size": len(tree.nodes),
            "max_depth": max(n.depth for n in tree.nodes),
            "best_node_id": best_id,
            "best_q": best_q,
        },
    )
