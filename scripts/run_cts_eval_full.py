#!/usr/bin/env python3
"""CTS Full Evaluation Pipeline — Table 2 Reproduction (paper §7).

Paper §7.1: "5 seeds (3 full re-trainings + 2 inference-only);
95% CI via bootstrap (1000 resamples);
Wilcoxon signed-rank; Bonferroni-corrected for 12 primary comparisons."

Usage:
    python scripts/run_cts_eval_full.py --benchmarks math500 aime gsm8k
    python scripts/run_cts_eval_full.py --mode 4nu --seeds 5
    python scripts/run_cts_eval_full.py --table2  # full Table 2 reproduction
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cts.eval.statistics import (
    StatisticalResult,
    bonferroni_correct,
    bootstrap_ci,
    format_result,
    wilcoxon_signed_rank,
)
from cts.types import NuConfigMode
from cts.utils.config import load_config


BENCHMARKS = ["math500", "gsm8k", "aime", "arc_agi_text", "humaneval"]

TABLE2_METHODS = [
    "greedy",
    "think_off_greedy",
    "native_think",
    "ft_nt",
    "bon_13",
    "deq_only",
    "bandit_ucb1",
    "cts_2nu",
    "cts_4nu",
]


def run_single_evaluation(
    method: str,
    benchmark: str,
    seed: int,
    *,
    config_name: str = "default",
    device: str = "cuda:0",
    model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single evaluation and return scores."""
    cfg = load_config(config_name)
    from cts.utils.seed import set_seed
    set_seed(seed)

    result: Dict[str, Any] = {
        "method": method,
        "benchmark": benchmark,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        if benchmark == "math500":
            from cts.eval.math500 import load_math500, evaluate_math500
            problems = load_math500()
            scores = _run_cts_on_problems(method, problems, cfg, device, model_dir)
            result["accuracy"] = sum(scores) / max(len(scores), 1)
            result["scores"] = scores

        elif benchmark == "gsm8k":
            from cts.eval.gsm8k import load_gsm8k, evaluate_gsm8k
            problems = load_gsm8k()
            scores = _run_cts_on_problems(method, problems, cfg, device, model_dir)
            result["accuracy"] = sum(scores) / max(len(scores), 1)
            result["scores"] = scores

        elif benchmark == "aime":
            scores = _run_cts_on_problems(method, [], cfg, device, model_dir)
            result["accuracy"] = sum(scores) / max(len(scores), 1) if scores else 0.0
            result["scores"] = scores

        elif benchmark == "arc_agi_text":
            from cts.eval.arc_agi_text import load_arc_tasks
            tasks = load_arc_tasks()
            scores = _run_cts_on_problems(method, tasks, cfg, device, model_dir)
            result["accuracy"] = sum(scores) / max(len(scores), 1)
            result["scores"] = scores

        elif benchmark == "humaneval":
            from cts.eval.humaneval import load_humaneval
            problems = load_humaneval()
            scores = _run_cts_on_problems(method, problems, cfg, device, model_dir)
            result["accuracy"] = sum(scores) / max(len(scores), 1)
            result["scores"] = scores

    except Exception as e:
        result["error"] = str(e)
        result["accuracy"] = 0.0
        result["scores"] = []

    return result


def _run_cts_on_problems(
    method: str,
    problems: list,
    cfg: dict,
    device: str,
    model_dir: Optional[str],
) -> List[float]:
    """Placeholder for actual CTS/baseline evaluation."""
    return []


def run_table2_reproduction(
    *,
    seeds: List[int],
    benchmarks: List[str],
    config_name: str = "default",
    device: str = "cuda:0",
    output_dir: str = "results/table2",
    model_dir: Optional[str] = None,
) -> Dict[str, Dict[str, StatisticalResult]]:
    """Reproduce Table 2 with full statistical protocol."""
    os.makedirs(output_dir, exist_ok=True)
    all_results: Dict[str, Dict[str, List[float]]] = {}

    for method in TABLE2_METHODS:
        all_results[method] = {b: [] for b in benchmarks}

        for seed in seeds:
            for bench in benchmarks:
                print(f"  [{method}] {bench} seed={seed}")
                result = run_single_evaluation(
                    method, bench, seed,
                    config_name=config_name,
                    device=device,
                    model_dir=model_dir,
                )
                acc = result.get("accuracy", 0.0)
                all_results[method][bench].append(acc)

    table2: Dict[str, Dict[str, StatisticalResult]] = {}
    for method in TABLE2_METHODS:
        table2[method] = {}
        for bench in benchmarks:
            scores = all_results[method][bench]
            table2[method][bench] = bootstrap_ci(scores, ci_level=0.95)

    _print_table2(table2, benchmarks)
    _run_wilcoxon_tests(all_results, benchmarks, output_dir)
    _save_results(all_results, table2, output_dir)

    return table2


def _print_table2(
    table2: Dict[str, Dict[str, StatisticalResult]],
    benchmarks: List[str],
) -> None:
    """Print Table 2 in paper format."""
    print("\n" + "=" * 80)
    print("Table 2: Budget-capped performance (10^14 MACs max)")
    print("=" * 80)

    header = f"{'Method':<25}" + "".join(f"{b:>12}" for b in benchmarks)
    print(header)
    print("-" * len(header))

    for method, bench_stats in table2.items():
        row = f"{method:<25}"
        for bench in benchmarks:
            s = bench_stats.get(bench)
            if s and s.n_samples > 0:
                ci_half = (s.ci_upper - s.ci_lower) / 2.0 * 100
                row += f"{s.mean * 100:>8.1f}+-{ci_half:.1f}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    print("=" * 80)


def _run_wilcoxon_tests(
    all_results: Dict[str, Dict[str, List[float]]],
    benchmarks: List[str],
    output_dir: str,
) -> None:
    """Wilcoxon signed-rank + Bonferroni (paper: alpha=0.05/12)."""
    print("\nWilcoxon signed-rank tests (Bonferroni-corrected, alpha=0.05/12):")
    print("-" * 60)

    comparisons = []
    cts_4nu = all_results.get("cts_4nu", {})

    for baseline in TABLE2_METHODS:
        if baseline == "cts_4nu":
            continue
        baseline_data = all_results.get(baseline, {})
        for bench in benchmarks:
            x = cts_4nu.get(bench, [])
            y = baseline_data.get(bench, [])
            if x and y and len(x) == len(y):
                w, p = wilcoxon_signed_rank(x, y)
                comparisons.append((f"CTS-4nu vs {baseline} ({bench})", p))

    if comparisons:
        p_values = [p for _, p in comparisons]
        corrected = bonferroni_correct(p_values, n_comparisons=len(comparisons))
        for (name, raw_p), corr_p in zip(comparisons, corrected):
            sig = "***" if corr_p < 0.05 / 12 else "n.s."
            print(f"  {name}: p_raw={raw_p:.4f}, p_corr={corr_p:.4f} {sig}")


def _save_results(
    all_results: Dict[str, Dict[str, List[float]]],
    table2: Dict[str, Dict[str, StatisticalResult]],
    output_dir: str,
) -> None:
    """Save results to JSON."""
    out_path = Path(output_dir) / "table2_results.json"
    serializable = {}
    for method, bench_data in table2.items():
        serializable[method] = {}
        for bench, stat in bench_data.items():
            serializable[method][bench] = {
                "mean": stat.mean,
                "std": stat.std,
                "ci_lower": stat.ci_lower,
                "ci_upper": stat.ci_upper,
                "n_samples": stat.n_samples,
            }

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CTS Full Evaluation Pipeline")
    parser.add_argument("--table2", action="store_true", help="Full Table 2 reproduction")
    parser.add_argument("--benchmarks", nargs="+", default=["math500", "aime"],
                        choices=BENCHMARKS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--config", default="default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/table2")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--mode", default="4nu", choices=["4nu", "2nu_fast", "1nu"])
    args = parser.parse_args()

    seed_list = list(range(args.seeds))

    if args.table2:
        benchmarks = BENCHMARKS
    else:
        benchmarks = args.benchmarks

    print(f"CTS Evaluation Pipeline")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Seeds: {seed_list}")
    print(f"  Mode: {args.mode}")
    print(f"  Config: {args.config}")
    print()

    run_table2_reproduction(
        seeds=seed_list,
        benchmarks=benchmarks,
        config_name=args.config,
        device=args.device,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
