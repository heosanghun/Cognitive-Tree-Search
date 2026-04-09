#!/usr/bin/env python3
"""
CTS single transition on local Gemma 4 weights (E4B-it recommended).

단계별 실행 (권장):
  **1단계 — blend** (가벼움, 파이프라인·Broyden 수렴 확인)
      python scripts/run_cts_local_gemma.py
      # 또는 명시: --map blend

  **2단계 — parallel** (논문 Eq.(5)에 가까운 희소 모듈 맵, Broyden 반복당 GPU 부담 큼)
      python scripts/run_cts_local_gemma.py --parallel
      # 또는: --map parallel

환경 변수:
  CTS_GEMMA_MODEL_DIR   기본: D:\\AI\\cts\\gemma-4-E4B-it
  CTS_DEQ_MAP_MODE      스크립트 인자가 없을 때만 사용 (--map / --parallel 이 우선)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import transition
from cts.model.gemma_loader import load_gemma4_e4b
from cts.types import NuVector, RuntimeBudgetState


def main() -> None:
    parser = argparse.ArgumentParser(description="CTS local Gemma — staged blend → parallel")
    parser.add_argument(
        "--map",
        choices=("blend", "full", "parallel"),
        default=None,
        help="Inner DEQ map (default: blend = Stage 1)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Shortcut for Stage 2: same as --map parallel",
    )
    parser.add_argument(
        "--max-decode",
        type=int,
        default=16,
        help="AR decode tokens from z* after DEQ (Gemma only; 1=greedy single token)",
    )
    args = parser.parse_args()

    if args.parallel:
        mode = "parallel"
    elif args.map is not None:
        mode = args.map
    else:
        mode = os.environ.get("CTS_DEQ_MAP_MODE", "blend")

    os.environ["CTS_DEQ_MAP_MODE"] = mode

    local = os.environ.get("CTS_GEMMA_MODEL_DIR", r"D:\AI\cts\gemma-4-E4B-it")
    os.environ.setdefault("CTS_GEMMA_MODEL_DIR", local)
    dm = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("local_model=", local)
    print("device=", dm)
    print("CTS_DEQ_MAP_MODE=", mode, "(1단계=blend, 2단계=parallel)")
    model, tok = load_gemma4_e4b(model_id=local, device_map=dm, torch_dtype=torch.bfloat16)
    bb = GemmaCTSBackbone(model, tok)
    nu = NuVector(nu_ne=0.5, nu_ach=1.0, nu_5ht=1.0)
    r = transition(
        "2+2=?",
        0,
        nu,
        RuntimeBudgetState(),
        bb,
        K=4,
        d=bb.hidden_size,
        broyden_max_iter=15,
        broyden_tol_min=1e-1,
        broyden_tol_max=2e-1,
        tau_flops_budget=1e22,
        max_decode_tokens=args.max_decode,
    )
    print("prune", r.prune, "iters", r.solver_stats.get("iterations"), "child", r.child_text)


if __name__ == "__main__":
    main()
