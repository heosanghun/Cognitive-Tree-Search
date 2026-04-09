"""Extract prompt text from OpenMathInstruct-1 JSONL rows (streaming JSON)."""

from __future__ import annotations

import json
from typing import Any, Dict


def prompt_text_from_openmath_row(row: Dict[str, Any]) -> str:
    """
    Prefer `question` (OpenMathInstruct-1 train rows), then common fallbacks.
    """
    if not isinstance(row, dict):
        return str(row)[:8192]
    q = row.get("question")
    if isinstance(q, str) and q.strip():
        return q.strip()
    for key in ("problem", "instruction", "prompt", "input", "query"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        parts = []
        for m in msgs:
            if isinstance(m, dict) and m.get("content"):
                parts.append(str(m["content"]))
        if parts:
            return "\n".join(parts)[:8192]
    try:
        return json.dumps(row, ensure_ascii=False)[:8192]
    except Exception:
        return str(row)[:8192]
