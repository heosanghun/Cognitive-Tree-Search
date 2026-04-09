"""paper_parity.yaml merges over default with expected overrides."""

from pathlib import Path

from cts.utils.config import load_config


def test_paper_parity_merges_stage_steps_and_parallel():
    cfg = load_config("paper_parity")
    assert cfg["stage1_max_steps"] == 10000
    assert cfg["stage2_total_ppo_steps"] == 10000
    assert cfg["cts_deq_map_mode"] == "parallel"
    assert cfg["stage2_parallel_map"] is True
    assert cfg["broyden_max_iter"] == 30


def test_paper_parity_keeps_default_mcts_w():
    cfg = load_config("paper_parity")
    assert cfg["mcts_branching_W"] == 3
