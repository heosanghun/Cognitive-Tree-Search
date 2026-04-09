"""
Gemma 4 E4B backbone for CTS (paper-aligned paths).

- encode_context: Gemma4Model text-only forward, mean-pooled anchor (RoPE on s_t in HF).
  **RoPE / inner-z:** Single source of truth for the **API contract** is `cts.backbone.rope_contract`.
  HF 단일 `forward`만으로는 inner block에 대한 “커스텀 position_ids”가 필요 없는 경로로
  앵커는 `encode_context`, latent는 `deq_step`로 분리한다. Optional Phase-2 HF hook은
  `rope_contract.phase2_custom_forward_available()`.
- deq_step: configurable (`CTS_DEQ_MAP_MODE`) —
    * `blend` (**default**): 가벼운 어댑터만 사용 → Broyden·파이프라인 동작 확인용(권장 1단계).
    * `parallel`: Eq.(5) 스타일 희소 병렬 모듈(무거움, GPU 권장 2단계).
    * `full`: 42층 전체 순차 1패스(애블레이션).
- routing_proj: learnable W_g [19, H] for alpha = softmax(W_g @ pool(z) / nu_ACh).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
import torch.nn as nn

from cts.backbone.protocol import BaseCTSBackbone
from cts.deq.gemma_latent_forward import full_stack_forward, parallel_sparse_module_forward
from cts.model.module_partition import layers_for_module


class GemmaCTSBackbone(BaseCTSBackbone, nn.Module):
    def __init__(self, cg_model: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.cg = cg_model
        self.tokenizer = tokenizer
        cfg = cg_model.config.get_text_config()
        self._hidden = int(cfg.hidden_size)
        self._num_layers = int(cfg.num_hidden_layers)
        dev = next(cg_model.parameters()).device
        dt = next(cg_model.parameters()).dtype
        # Paper Eq.(5): W_g for routing logits (trainable in Stage 2).
        self.routing_proj = nn.Parameter(torch.randn(19, self._hidden, device=dev, dtype=dt) * 0.02)
        # Legacy fast path
        self._blend = nn.Linear(self._hidden, self._hidden, bias=True).to(device=dev)
        nn.init.normal_(self._blend.weight, std=0.02)
        nn.init.zeros_(self._blend.bias)
        # parallel | full | blend — 기본은 blend(동작 확인 후 parallel 권장)
        self.deq_map_mode = os.environ.get("CTS_DEQ_MAP_MODE", "blend")

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_size(self) -> int:
        return self._hidden

    def _device(self) -> torch.device:
        return next(self.cg.parameters()).device

    def routing_matrix(self) -> torch.Tensor:
        """W_g in paper (19 x H)."""
        return self.routing_proj

    def encode_context(self, parent_text: str) -> torch.Tensor:
        enc = self.tokenizer(
            parent_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        device = self._device()
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.set_grad_enabled(self.training):
            out = self.cg.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        h = out.last_hidden_state.float()
        ctx = h.mean(dim=1)
        return ctx.to(dtype=next(self.cg.parameters()).dtype)

    def _lm(self) -> nn.Module:
        return self.cg.model.language_model

    def deq_step(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        module_weights: torch.Tensor,
        extra: Dict[str, Any],
    ) -> torch.Tensor:
        mode = extra.get("deq_map_mode", self.deq_map_mode)
        top_k = int(extra.get("top_k", 3))
        ctx = context
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)

        if mode in ("parallel", "paper"):
            lm = self._lm()
            out = parallel_sparse_module_forward(
                lm,
                z,
                ctx,
                module_weights,
                layers_for_module,
                top_k=top_k,
            )
            return out.to(dtype=z.dtype)

        if mode == "full":
            out = full_stack_forward(self._lm(), z, ctx)
            return out.to(dtype=z.dtype)

        # blend fallback
        zf = z.float()
        ctxf = ctx.float().expand(zf.shape[0], -1)
        gate = float(module_weights.sum().item() / max(module_weights.numel(), 1))
        gate = max(0.25, min(1.5, gate))
        h = zf + ctxf
        delta = self._blend(h)
        mixed = 0.82 * zf + 0.18 * gate * torch.tanh(delta)
        return mixed.to(dtype=z.dtype)

    def decode_from_z_star(self, z_star: torch.Tensor, *, max_new_tokens: int = 1) -> str:
        """
        Decode from fixed-point latent z* [K, H].
        `max_new_tokens==1`: single greedy token via lm_head (cheap).
        `max_new_tokens>1`: causal AR from pooled z as first `inputs_embeds` step, then KV cache.
        """
        if max_new_tokens <= 0:
            return ""
        dt = next(self.cg.parameters()).dtype
        device = self._device()
        if max_new_tokens == 1:
            h = z_star.mean(dim=0).to(dtype=dt)
            logits = self.cg.lm_head(h.unsqueeze(0)).squeeze(0)
            tid = int(logits.argmax(dim=-1).item())
            return self.tokenizer.decode([tid], skip_special_tokens=True)

        z0 = z_star.mean(dim=0).to(dtype=dt).view(1, 1, -1)
        attn = torch.ones(1, 1, device=device, dtype=torch.long)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            try:
                eos = int(self.cg.config.get_text_config().eos_token_id)
            except Exception:
                eos = 1
        ids: List[int] = []
        try:
            with torch.no_grad():
                past = None
                for step in range(max_new_tokens):
                    if step == 0:
                        out = self.cg.model(
                            inputs_embeds=z0,
                            attention_mask=attn,
                            use_cache=True,
                            return_dict=True,
                        )
                    else:
                        tid_t = torch.tensor([[ids[-1]]], device=device, dtype=torch.long)
                        out = self.cg.model(
                            input_ids=tid_t,
                            past_key_values=past,
                            use_cache=True,
                            return_dict=True,
                        )
                    past = out.past_key_values
                    h = out.last_hidden_state[:, -1, :]
                    logits = self.cg.lm_head(h)
                    next_id = int(logits.argmax(dim=-1).item())
                    ids.append(next_id)
                    if next_id == eos:
                        break
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            # Fallback: single-token path if multimodal forward rejects inputs_embeds-only
            h = z_star.mean(dim=0).to(dtype=dt)
            logits = self.cg.lm_head(h.unsqueeze(0)).squeeze(0)
            tid = int(logits.argmax(dim=-1).item())
            return self.tokenizer.decode([tid], skip_special_tokens=True)
