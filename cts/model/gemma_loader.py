"""Load Gemma 4 E4B from Hugging Face (text path via multimodal checkpoint)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

DEFAULT_GEMMA4_E4B_ID = "google/gemma-4-E4B"


def default_hub_cache_dir() -> Optional[Path]:
    """Prefer repo-local cache when HF_HUB_CACHE is unset (avoids full C: drive)."""
    repo_root = Path(__file__).resolve().parents[2]
    local = repo_root / ".hf_cache"
    return local if local.parent.exists() else None


def ensure_hub_cache_env() -> None:
    if os.environ.get("HF_HUB_CACHE"):
        return
    cand = default_hub_cache_dir()
    if cand is not None:
        cand.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(cand))


def load_gemma4_e4b(
    model_id: str = DEFAULT_GEMMA4_E4B_ID,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict[str, Any] | None = "auto",
    token: Optional[str] = None,
    hub_cache: Optional[str | Path] = None,
    low_cpu_mem_usage: bool = True,
) -> Tuple[Any, Any]:
    """
    Returns (Gemma4ForConditionalGeneration, AutoTokenizer).

    Requires transformers >= 5.5 (Gemma 4 support). Install dev build if needed:
    pip install git+https://github.com/huggingface/transformers.git

    Gated models: set HF_TOKEN or pass token=.

    If `model_id` is still the default Hub id, set env **CTS_GEMMA_MODEL_DIR** to a local
    folder that contains `config.json` and `model.safetensors` (e.g. snapshot from
    `huggingface_hub.snapshot_download`).
    """
    ensure_hub_cache_env()
    if hub_cache is not None:
        os.environ["HF_HUB_CACHE"] = str(hub_cache)

    token = token or os.environ.get("HF_TOKEN")

    if model_id == DEFAULT_GEMMA4_E4B_ID:
        override = os.environ.get("CTS_GEMMA_MODEL_DIR")
        if override:
            p = Path(override).expanduser().resolve()
            if p.is_dir() and (p / "config.json").is_file():
                model_id = str(p)

    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
        token=token,
        trust_remote_code=True,
    )
    return model, tokenizer
