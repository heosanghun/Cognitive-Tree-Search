"""
Gemma 4 text LM forward on continuous latent z (soft thought bottleneck).

Gemma4TextModel requires per-layer embeddings (PLE). Passing **zeros** as the
second argument to `project_per_layer_inputs` avoids discrete-token `get_per_layer_inputs`
while still producing valid [B,T,L,d_pl] tensors (see HF Gemma4TextModel).
"""

from __future__ import annotations

from typing import List

import torch


def project_ple_inputs(
    lm: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    *,
    zero_init: bool = True,
) -> torch.Tensor:
    """Returns combined per-layer inputs [B,T,num_layers,d_pl]."""
    cfg = lm.config
    b, t, _ = inputs_embeds.shape
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    nL = cfg.num_hidden_layers
    dpl = cfg.hidden_size_per_layer_input
    if zero_init:
        zpl = torch.zeros(b, t, nL, dpl, device=device, dtype=dtype)
    else:
        zpl = None
    return lm.project_per_layer_inputs(inputs_embeds, zpl)


def full_stack_forward(
    lm: torch.nn.Module,
    z: torch.Tensor,
    context_row: torch.Tensor,
) -> torch.Tensor:
    """
    One full sequential pass through all decoder layers (paper inner map without sparse ablation).

    z: [K, H], context_row: [1, H] — context is added to each latent slot (conditioning).
    Returns z_out [K, H].
    """
    device = z.device
    dtype = z.dtype
    k, h = z.shape
    ctx = context_row.to(device=device, dtype=dtype)
    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(0)
    h0 = z + ctx.expand(k, -1)
    inputs_embeds = h0.unsqueeze(0)
    attention_mask = torch.ones(1, k, device=device, dtype=torch.long)
    past_seen = 0
    position_ids = torch.arange(k, device=device, dtype=torch.long).unsqueeze(0) + past_seen

    per_layer_inputs = project_ple_inputs(lm, inputs_embeds, zero_init=True)

    # Match HF: build causal mask dict for sliding/full attention layers
    from transformers.masking_utils import create_masks_for_generate

    causal_mask_mapping = create_masks_for_generate(
        lm.config,
        inputs_embeds,
        attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )

    position_embeddings = {}
    for layer_type in lm.unique_layer_types:
        position_embeddings[layer_type] = lm.rotary_emb(inputs_embeds, position_ids, layer_type)

    hidden_states = inputs_embeds
    for i, decoder_layer in enumerate(lm.layers[: lm.config.num_hidden_layers]):
        pli = per_layer_inputs[:, :, i, :]
        hidden_states = decoder_layer(
            hidden_states,
            pli,
            position_embeddings=position_embeddings[lm.config.layer_types[i]],
            attention_mask=causal_mask_mapping[lm.config.layer_types[i]],
            position_ids=position_ids,
            past_key_values=None,
        )

    hidden_states = lm.norm(hidden_states)
    return hidden_states.squeeze(0)


def parallel_sparse_module_forward(
    lm: torch.nn.Module,
    z: torch.Tensor,
    context_row: torch.Tensor,
    module_alpha: torch.Tensor,
    layers_for_module,
    top_k: int = 3,
) -> torch.Tensor:
    """
    Paper Eq.(5)-style: sum_m alpha_m * Module_m(z), each module runs its layer subset from shared h0.

    module_alpha: [19] softmax weights (already sparse if desired).
    """
    device = z.device
    dtype = z.dtype
    k, h = z.shape
    ctx = context_row.to(device=device, dtype=dtype)
    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(0)
    h0 = (z + ctx.expand(k, -1)).unsqueeze(0)
    per_layer_inputs = project_ple_inputs(lm, h0, zero_init=True)

    attention_mask = torch.ones(1, k, device=device, dtype=torch.long)
    position_ids = torch.arange(k, device=device, dtype=torch.long).unsqueeze(0)
    from transformers.masking_utils import create_masks_for_generate

    causal_mask_mapping = create_masks_for_generate(
        lm.config,
        h0,
        attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    position_embeddings = {}
    for layer_type in lm.unique_layer_types:
        position_embeddings[layer_type] = lm.rotary_emb(h0, position_ids, layer_type)

    m = module_alpha.numel()
    k_top = min(top_k, m)
    sel_alpha, top_idx = torch.topk(module_alpha, k_top)
    sel_alpha = sel_alpha / (sel_alpha.sum().clamp_min(1e-8))

    acc = torch.zeros_like(h0)
    for j, mod in enumerate(top_idx.tolist()):
        a = float(sel_alpha[j].item())
        hidden = h0.clone()
        layer_ids: List[int] = layers_for_module(mod)
        for li in layer_ids:
            pli = per_layer_inputs[:, :, li, :]
            hidden = lm.layers[li](
                hidden,
                pli,
                position_embeddings=position_embeddings[lm.config.layer_types[li]],
                attention_mask=causal_mask_mapping[lm.config.layer_types[li]],
                position_ids=position_ids,
                past_key_values=None,
            )
        acc = acc + a * hidden

    acc = lm.norm(acc)
    return acc.squeeze(0)
