"""Row chunking exato para o transformer MiniMax H3.

Adaptado de ComfyUI-NynxzH3, Copyright (c) 2026 Nynxz, sob licenca MIT.
O aviso MIT completo do projeto de origem e preservado em THIRD_PARTY_NOTICES.md.

As projecoes QKV e o MLP sao operacoes independentes por linha. Dividi-las
pela dimensao da sequencia reduz o pico de ativacao e evita o limite de
indexacao int32 dos kernels CUDA, sem aproximar a funcao do modelo.
"""

from __future__ import annotations

import torch


INT32_ELEMENTS = 2**31
INT32_MARGIN = 0.85
TARGET_BYTES = 1 << 30


def _diffusion_model(model):
    inner = getattr(model, "model", None)
    return getattr(inner, "diffusion_model", None)


def _is_h3(model):
    dit = _diffusion_model(model)
    return dit is not None and type(dit).__name__ == "MiniMaxH3Model"


def _out_features(linear):
    value = getattr(linear, "out_features", None)
    return int(value) if value is not None else int(linear.weight.shape[0])


def _chunk_rows(seq_len, width, itemsize, override=0):
    if override > 0:
        return max(1, min(int(override), seq_len))
    by_int32 = int(INT32_ELEMENTS * INT32_MARGIN) // max(1, width)
    by_bytes = TARGET_BYTES // max(1, width * max(1, itemsize))
    return max(1, min(seq_len, by_int32, by_bytes))


def _linear_chunked(layer, x, chunk):
    rows = x.shape[0]
    if chunk >= rows:
        return layer(x)
    out = None
    for a in range(0, rows, chunk):
        b = min(a + chunk, rows)
        piece = layer(x[a:b])
        if out is None:
            out = torch.empty((rows, *piece.shape[1:]), dtype=piece.dtype, device=piece.device)
        out[a:b] = piece
    return out


def _mlp_forward(mlp, x, override=0):
    import comfy.ops

    rows = x.shape[0]
    chunk = _chunk_rows(rows, _out_features(mlp.fc1), x.element_size(), override)
    if chunk >= rows:
        return comfy.ops.linear_input_act(mlp.fc2, mlp.fc1(x), "swiglu")

    out = torch.empty_like(x)
    for a in range(0, rows, chunk):
        b = min(a + chunk, rows)
        out[a:b] = comfy.ops.linear_input_act(mlp.fc2, mlp.fc1(x[a:b]), "swiglu")
    return out


def _attention_forward(attn, x, rope_freqs=None, transformer_options={}, override=0):
    import comfy.model_management
    import comfy.quant_ops
    from comfy.ldm.modules.attention import optimized_attention

    seq = x.shape[0]
    heads, head_dim = attn.heads, attn.head_dim
    inner = heads * head_dim
    chunk = _chunk_rows(seq, _out_features(attn.qkv_proj), x.element_size(), override)

    if chunk >= seq:
        q, k, v = attn.qkv_proj(x).split(inner, dim=-1)
        q = q.view(1, seq, heads, head_dim)
        k = k.view(1, seq, heads, head_dim)
        v = v.view(seq, heads, head_dim)
    else:
        q = k = v = None
        for a in range(0, seq, chunk):
            b = min(a + chunk, seq)
            qc, kc, vc = attn.qkv_proj(x[a:b]).split(inner, dim=-1)
            if q is None:
                opts = {"dtype": qc.dtype, "device": qc.device}
                q = torch.empty(1, seq, heads, head_dim, **opts)
                k = torch.empty(1, seq, heads, head_dim, **opts)
                v = torch.empty(seq, heads, head_dim, **opts)
            q[0, a:b] = qc.view(-1, heads, head_dim)
            k[0, a:b] = kc.view(-1, heads, head_dim)
            v[a:b] = vc.view(-1, heads, head_dim)

    if rope_freqs is not None:
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
            )
        q, k = q[0], k[0]
    else:
        q = attn.q_norm(q[0])
        k = attn.k_norm(k[0])

    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    out = optimized_attention(
        q, k, v, heads, mask=None, skip_reshape=True,
        transformer_options=transformer_options,
    ).squeeze(0)
    out_chunk = _chunk_rows(seq, _out_features(attn.out_proj), out.element_size(), override)
    return _linear_chunked(attn.out_proj, out, out_chunk)


class BruxosH3RowChunk:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "chunk_rows": ("INT", {
                "default": 0, "min": 0, "max": 1000000, "step": 1024,
                "tooltip": "0 = automatico (~1 GiB por intermediario). Use manual apenas se ainda ocorrer CUDA illegal memory access.",
            }),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "relatorio")
    FUNCTION = "apply"
    CATEGORY = "Bruxos do VFX/MiniMax H3/Otimizacao"
    DESCRIPTION = "Divide QKV e MLP por linhas: menor pico de VRAM e evita o limite int32 em sequencias grandes, sem aproximacao."

    def apply(self, model, chunk_rows=0):
        if not _is_h3(model):
            wired = type(_diffusion_model(model)).__name__
            raise ValueError(
                "Bruxos H3 Row Chunk requer um MiniMax H3; o modelo conectado e "
                f"{wired!r}."
            )

        patched = model.clone()
        dit = _diffusion_model(patched)
        blocks = list(dit.blocks)
        for index, block in enumerate(blocks):
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.mlp.forward",
                lambda x, _m=block.mlp, _o=chunk_rows: _mlp_forward(_m, x, _o),
            )
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.attn.forward",
                lambda x, rope_freqs=None, transformer_options={}, _m=block.attn, _o=chunk_rows: (
                    _attention_forward(_m, x, rope_freqs, transformer_options, _o)
                ),
            )

        widest = max(_out_features(blocks[0].mlp.fc1), _out_features(blocks[0].attn.qkv_proj))
        ceiling = INT32_ELEMENTS // widest
        mode = f"manual: {chunk_rows:,} linhas" if chunk_rows else "automatico: alvo de ~1 GiB"
        report = (
            f"Bruxos H3 Row Chunk aplicado em {len(blocks)} blocos\n"
            f"modo: {mode}\n"
            f"limite int32 estimado sem chunk: {ceiling:,} linhas\n"
            "qualidade: operacao exata por linha; pode ficar conectado em clips curtos\n"
            "Turbo LoRA/SolAttn: preservados porque as camadas e optimized_attention continuam sendo chamadas"
        )
        return patched, report


NODE_CLASS_MAPPINGS = {"BruxosH3RowChunk": BruxosH3RowChunk}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosH3RowChunk": "H3 Row Chunk Exato (Bruxos)"}

