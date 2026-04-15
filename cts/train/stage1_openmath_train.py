"""
Stage 1: OpenMathInstruct JSONL + GemmaCTSBackbone + IFT residual loss (paper §6.1).

Base Gemma weights are frozen; trains `routing_proj`, `_blend`, and optional LoRA adapters.
LoRA r=8, ~18 MB trainable parameters. 10,000 examples from OpenMathInstruct-2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.model.gemma_loader import load_gemma4_e4b
from cts.train.jsonl_iter import iter_jsonl
from cts.train.openmath_text import prompt_text_from_openmath_row
from cts.train.stage1_warmup import fixed_point_surrogate_loss
from cts.types import NuVector
from cts.utils.config import load_config
from cts.utils.repro_seed import apply_global_seed


def _set_trainable_params(bb: GemmaCTSBackbone, *, train_lora: bool) -> None:
    for n, p in bb.named_parameters():
        if "routing_proj" in n or "_blend" in n:
            p.requires_grad = True
        elif "lora_" in n and train_lora:
            p.requires_grad = True
        else:
            p.requires_grad = False


def _maybe_apply_lora(
    bb: GemmaCTSBackbone,
    *,
    rank: int,
    target_modules: list[str],
) -> GemmaCTSBackbone:
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
    except ImportError:
        return bb
    lm = bb.cg.model.language_model
    lcfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    wrapped = get_peft_model(lm, lcfg)
    bb.cg.model.language_model = wrapped
    return bb


def _save_checkpoint(
    bb: GemmaCTSBackbone,
    opt: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    config_name: str,
    openmath_jsonl: str,
    lora: bool,
    losses: list[float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone_state_dict": bb.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "step": step,
            "total_steps": total_steps,
            "config_name": config_name,
            "openmath_jsonl": openmath_jsonl,
            "lora": lora,
            "losses": losses[-200:],
        },
        path,
    )
    print(f"  [checkpoint] saved step {step} → {path}")


def run_stage1_openmath_training(
    *,
    openmath_jsonl: Path | str,
    config_name: str = "default",
    max_steps: Optional[int] = None,
    device: Optional[str] = None,
    lora: bool = False,
    lora_targets: Optional[list[str]] = None,
    log_every: int = 20,
    model_dir: Optional[str] = None,
    resume: bool = False,
    save_every: int = 500,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = load_config(config_name)
    apply_global_seed()
    deq_from_cfg = cfg.get("cts_deq_map_mode")
    if deq_from_cfg and not os.environ.get("CTS_DEQ_MAP_MODE"):
        os.environ["CTS_DEQ_MAP_MODE"] = str(deq_from_cfg)
    steps = int(
        max_steps if max_steps is not None else cfg.get("stage1_max_steps", 5000)
    )
    K = int(cfg.get("soft_thought_K", 8))
    nu = NuVector(nu_tol=0.5, nu_temp=1.0, nu_expl=1.0)

    dev_s = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_s)
    map_gpu = dev_s if dev_s.startswith("cuda") else None
    if map_gpu is None and dev.type == "cuda":
        map_gpu = str(dev)

    mid = model_dir or os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
    model, tok = load_gemma4_e4b(
        model_id=mid,
        device_map=map_gpu if map_gpu else "auto",
        torch_dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
    )
    bb = GemmaCTSBackbone(model, tok)
    bb.train()

    targets = lora_targets or list(cfg.get("lora_target", ["q_proj", "v_proj"]))
    if lora:
        bb = _maybe_apply_lora(
            bb, rank=int(cfg.get("lora_rank", 8)), target_modules=targets
        )
    _set_trainable_params(bb, train_lora=lora)

    params = [p for p in bb.parameters() if p.requires_grad]
    lr = float(cfg.get("lr", 3e-5))
    opt = torch.optim.AdamW(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None

    start_step = 0
    losses: list[float] = []
    ckpt_path = Path("artifacts") / "stage1_last.pt"

    if resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        bb.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                print("  [warn] optimizer state incompatible, resetting optimizer")
        start_step = ckpt.get("step", 0)
        losses = ckpt.get("losses", [])
        print(f"  Resumed at step {start_step}, continuing to {steps}")

    path = Path(openmath_jsonl)
    if not path.is_file():
        raise FileNotFoundError(f"OpenMath JSONL not found: {path}")

    row_iter = iter_jsonl(path)

    for step in range(start_step, steps):
        try:
            row = next(row_iter)
        except StopIteration:
            row_iter = iter_jsonl(path)
            row = next(row_iter)

        text = prompt_text_from_openmath_row(row)
        d = bb.hidden_size
        z = torch.randn(K, d, device=dev, dtype=torch.float32) * 0.02
        w_g = bb.routing_matrix().to(device=dev, dtype=torch.float32)

        opt.zero_grad(set_to_none=True)
        extra: Dict[str, Any] = {
            "top_k": int(cfg.get("top_k_modules", 3)),
            "deq_map_mode": bb.deq_map_mode,
        }

        lambda_lm = float(cfg.get("stage1_lambda_lm", 0.1))
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = fixed_point_surrogate_loss(
                    bb, text, z, nu, w_g=w_g, extra=extra,
                    lambda_lm=lambda_lm, tokenizer=tok,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                params, float(cfg.get("max_grad_norm", 1.0))
            )
            scaler.step(opt)
            scaler.update()
        else:
            loss = fixed_point_surrogate_loss(
                bb, text, z, nu, w_g=w_g, extra=extra,
                lambda_lm=lambda_lm, tokenizer=tok,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                params, float(cfg.get("max_grad_norm", 1.0))
            )
            opt.step()

        lv = float(loss.detach().cpu().item())
        losses.append(lv)
        if log_every and (step + 1) % log_every == 0:
            tail = sum(losses[-log_every:]) / min(len(losses), log_every)
            print(f"stage1 step={step + 1}/{steps} loss={lv:.6f} avg_last={tail:.6f}")

        if save_every and (step + 1) % save_every == 0:
            _save_checkpoint(
                bb, opt, step=step + 1, total_steps=steps,
                config_name=config_name, openmath_jsonl=str(path),
                lora=lora, losses=losses, path=ckpt_path,
            )

    _save_checkpoint(
        bb, opt, step=steps, total_steps=steps,
        config_name=config_name, openmath_jsonl=str(path),
        lora=lora, losses=losses, path=ckpt_path,
    )
    print(f"Stage 1 complete: {steps} steps, final loss={losses[-1]:.6f}")
    return {
        "checkpoint": str(ckpt_path),
        "final_loss": losses[-1] if losses else 0.0,
        "steps": steps,
    }
