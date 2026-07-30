# -*- coding: utf-8 -*-
"""Bernini Tiled — experimento de KV cache temporal (NAO INSTALADO).

Este arquivo e uma base de integracao para o backend `CausalWanModel` incluido
no ComfyUI. Ele nao deve ser adicionado ao __init__.py ate que um sampler
causal dedicado seja conectado ao Bernini.

Por que nao reutilizar K/V entre tiles espaciais normais?
---------------------------------------------------------
Cada passo de denoise altera os tokens do tile. Mesmo na sobreposicao, K/V de
um tile independente nao representa os tokens do tile seguinte (posicao RoPE,
contexto e ruido diferem). Reutiliza-los produziria uma imagem incorreta.

Onde o cache e valido:
----------------------
`CausalWanModel.forward_block()` processa blocos TEMPORAIS em ordem e recebe
KV/cross-attention caches por camada. O cache pode ser mantido entre blocos
temporais do mesmo passo/stream, eliminando a recomputacao do passado causal.

Estado atual:
-------------
O Bernini Infinity usa o sampler normal e carrega `WanModel`/`WAN21`, nao o
`CausalWanModel`. Este modulo detecta isso e falha de forma explicita em vez
de simular um cache que nao seria matematicamente valido.
"""

from dataclasses import dataclass


def _unwrap_model(model, max_depth=5):
    """Encontra o modulo torch interno sem depender de um ModelPatcher fixo."""
    cur = model
    for _ in range(max_depth):
        if cur is None:
            return None
        if hasattr(cur, "forward_block") and hasattr(cur, "init_kv_caches"):
            return cur
        nxt = None
        for name in ("model", "diffusion_model", "model_sampling"):
            candidate = getattr(cur, name, None)
            if candidate is not None and candidate is not cur:
                nxt = candidate
                break
        if nxt is None:
            break
        cur = nxt
    return cur


@dataclass
class KVTemporalState:
    """Caches alocados para um unico stream temporal causal."""
    kv_caches: list
    crossattn_caches: list
    max_seq_len: int


class CausalWanKVBridge:
    """Ponte de baixo nivel para CausalWanModel.

    O chamador deve manter uma instancia por tile espacial e por passada
    (high/low). Nunca compartilhe este estado entre tiles nem entre timesteps.
    """

    def __init__(self, diffusion_model):
        self.model = _unwrap_model(diffusion_model)
        if self.model is None or not hasattr(self.model, "forward_block"):
            actual = type(_unwrap_model(diffusion_model)).__name__ if _unwrap_model(diffusion_model) else "desconhecido"
            raise RuntimeError(
                "KV temporal requer CausalWanModel; backend atual: " + actual + ". "
                "O WAN21/Bernini normal nao aceita cache causal."
            )
        self.state = None

    def begin(self, batch_size, max_seq_len, device, dtype):
        self.state = KVTemporalState(
            kv_caches=self.model.init_kv_caches(batch_size, max_seq_len, device, dtype),
            crossattn_caches=self.model.init_crossattn_caches(batch_size, device, dtype),
            max_seq_len=int(max_seq_len),
        )
        return self.state

    def reset_for_timestep(self):
        """Reseta cache antes de cada timestep de denoise.

        Blocos temporais de um mesmo timestep podem reutilizar o passado;
        entre timesteps o latent muda, logo o cache anterior e invalido.
        """
        if self.state is None:
            raise RuntimeError("begin() deve ser chamado antes de reset_for_timestep().")
        self.model.reset_kv_caches(self.state.kv_caches)
        self.model.reset_crossattn_caches(self.state.crossattn_caches)

    def forward_block(self, x, timestep, context, start_frame, clip_fea=None):
        if self.state is None:
            raise RuntimeError("begin() deve ser chamado antes de forward_block().")
        return self.model.forward_block(
            x=x,
            timestep=timestep,
            context=context,
            start_frame=int(start_frame),
            kv_caches=self.state.kv_caches,
            crossattn_caches=self.state.crossattn_caches,
            clip_fea=clip_fea,
        )


class BruxosBerniniKVTemporalProbe:
    """Node de diagnostico para a futura variante causal; nao renderiza video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("causal_kv_supported", "report")
    FUNCTION = "probe"
    CATEGORY = "Bruxos do VFX/Experimental"

    def probe(self, model):
        core = _unwrap_model(model)
        ok = bool(core and hasattr(core, "forward_block") and hasattr(core, "init_kv_caches"))
        if ok:
            return (True, "CausalWanModel detectado: KV temporal pode ser integrado ao sampler causal.")
        name = type(core).__name__ if core is not None else "desconhecido"
        return (False, f"Backend {name}: Bernini atual usa sampler normal; KV temporal nao pode ser aplicado com seguranca.")


NODE_CLASS_MAPPINGS = {"BruxosBerniniKVTemporalProbe": BruxosBerniniKVTemporalProbe}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosBerniniKVTemporalProbe": "Bernini KV Temporal Probe (Experimental)"}
