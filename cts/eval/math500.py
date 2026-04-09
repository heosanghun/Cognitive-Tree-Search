"""
MATH-500 style evaluation hook.

Wire a dataset JSONL (e.g. HuggingFace `HuggingFaceH4/MATH-500` export) with fields:
  `problem` or `question`, `answer` (final boxed or numeric), optional `id`.

Grader: for a minimal path, compare `normalize_answer(pred) == normalize_answer(gold)`.
Full sympy integration can be added later.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    # strip common LaTeX wrappers
    s = s.replace("\\boxed{", "").rstrip("}")
    return s


def load_math_samples(path: Path | str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_pass_at_1(
    samples: List[Dict[str, Any]],
    predict_fn: Callable[[str], str],
    *,
    gold_key: str = "answer",
    question_key: str = "problem",
    include_items: bool = False,
    pred_max_chars: int = 4096,
) -> Dict[str, Any]:
    """`predict_fn(question)` returns model string; compared to `gold_key` after normalize."""
    ok = 0
    n = 0
    items: List[Dict[str, Any]] = []
    for ex in samples:
        q = ex.get(question_key) or ex.get("question") or ""
        gold = ex.get(gold_key, "")
        if not q:
            continue
        pred = predict_fn(str(q))
        n += 1
        match = normalize_answer(str(pred)) == normalize_answer(str(gold))
        if match:
            ok += 1
        if include_items:
            rid = ex.get("unique_id", ex.get("id", ""))
            items.append(
                {
                    "id": str(rid) if rid is not None else "",
                    "match": match,
                    "gold": str(gold)[:512],
                    "pred": str(pred)[:pred_max_chars],
                }
            )
    out: Dict[str, Any] = {
        "pass_at_1": (ok / n) if n else 0.0,
        "n": n,
        "correct": ok,
    }
    if include_items:
        out["items"] = items
    return out


def evaluate_stub() -> dict:
    return {
        "pass_at_1": None,
        "note": "Set MATH500_JSONL and run evaluate_pass_at_1(load_math_samples(path), predict_fn)",
    }
