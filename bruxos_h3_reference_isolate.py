"""Isolamento unidirecional das referencias no self-attention do MiniMax H3.

Adaptado de ComfyUI-NynxzH3, Copyright (c) 2026 Nynxz, licenca MIT.
Veja THIRD_PARTY_NOTICES.md.

O alvo continua atendendo texto e referencias. Apenas o prefixo de texto e
referencias deixa de atender o alvo ruidoso. Isso troca uma atencao quadrada
por duas chamadas retangulares e economiza mais quando a referencia ocupa uma
parte grande da sequencia.
"""

from __future__ import annotations

import logging
import torch


PREFIX_KEY = "bruxos_h3_isolate_prefix"
SEQLEN_KEY = "bruxos_h3_isolate_seqlen"
TARGET_KINDS = ("audio", "video")


def _diffusion_model(model):
    inner = getattr(model, "model", None)
    return getattr(inner, "diffusion_model", None)


def _is_h3(model):
    dit = _diffusion_model(model)
    return dit is not None and type(dit).__name__ == "MiniMaxH3Model"


def _prefix_length(layout):
    segments = getattr(layout, "segments", None)
    if not segments:
        return None
    for start, _stop, kind in segments:
        if kind in TARGET_KINDS:
            return int(start)
    return None


def _has_visual_reference(layout):
    segments = getattr(layout, "segments", None) or []
    return any(kind in ("cond", "ref_img") for _, _, kind in segments)


def _split_attention(func, args, kwargs, prefix):
    q, k, v = args[0], args[1], args[2]
    rest = args[3:]
    prefix_out = func(q[:, :, :prefix], k[:, :, :prefix], v[:, :, :prefix], *rest, **kwargs)
    target_out = func(q[:, :, prefix:], k, v, *rest, **kwargs)
    return torch.cat([prefix_out, target_out], dim=1)


def _attention_override(previous, sigma_start, sigma_end, state):
    def override(func, *args, **kwargs):
        options = kwargs.get("transformer_options") or {}
        prefix = options.get(PREFIX_KEY)
        seqlen = options.get(SEQLEN_KEY)

        def passthrough():
            if previous is not None:
                return previous(func, *args, **kwargs)
            return func(*args, **kwargs)

        if not prefix or len(args) < 3 or not isinstance(args[0], torch.Tensor):
            return passthrough()
        # A mesma funcao atende tambem o token refiner. Isolamos somente a
        # chamada cujo comprimento e o pacote H3 completo.
        if args[0].ndim < 3 or args[0].shape[2] != seqlen or prefix >= seqlen:
            return passthrough()

        sigma = options.get("sigmas")
        if sigma is not None:
            value = float(sigma.flatten()[0].item())
            if not (sigma_end <= value <= sigma_start):
                return passthrough()

        state["split_calls"] = state.get("split_calls", 0) + 1
        return _split_attention(func, args, kwargs, int(prefix))

    return override


def _forward_wrapper(state):
    def wrapper(executor, x, timestep, context, transformer_options, **kwargs):
        payload = kwargs.get("minimax_payload") or {}
        layout = payload.get("layout")
        prefix = _prefix_length(layout)
        if prefix and _has_visual_reference(layout):
            seqlen = int(layout.seq_len)
            transformer_options[PREFIX_KEY] = prefix
            transformer_options[SEQLEN_KEY] = seqlen
            signature = (prefix, seqlen)
            if state.get("logged_signature") != signature:
                saved = prefix * (seqlen - prefix) / max(1, seqlen * seqlen)
                logging.info(
                    "[Bruxos H3 Reference Isolate] prefixo=%s total=%s; reducao teorica "
                    "da area de atencao=%.1f%%",
                    prefix, seqlen, saved * 100.0,
                )
                state["logged_signature"] = signature
        else:
            transformer_options.pop(PREFIX_KEY, None)
            transformer_options.pop(SEQLEN_KEY, None)
            if layout is not None and not state.get("warned_no_ref"):
                logging.warning(
                    "[Bruxos H3 Reference Isolate] nenhum cond/ref_img visual no layout; "
                    "node em passthrough."
                )
                state["warned_no_ref"] = True
        return executor(x, timestep, context, transformer_options, **kwargs)

    return wrapper


class BruxosH3ReferenceIsolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "start_percent": ("FLOAT", {
                "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Comeca depois da composicao inicial. 0.15 e um teste conservador; 0 economiza mais.",
            }),
            "end_percent": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "1.0 mantem o isolamento ate o final.",
            }),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "relatorio")
    FUNCTION = "apply"
    CATEGORY = "Bruxos do VFX/MiniMax H3/Otimizacao"
    DESCRIPTION = "O alvo continua vendo as referencias, mas elas deixam de atender o alvo ruidoso. Reduz o custo de atencao em workflows com referencias grandes."

    def apply(self, model, start_percent=0.15, end_percent=1.0):
        if not _is_h3(model):
            wired = type(_diffusion_model(model)).__name__
            raise ValueError(
                "Bruxos H3 Reference Isolate requer MiniMax H3; o modelo conectado e "
                f"{wired!r}."
            )
        if end_percent < start_percent:
            raise ValueError("end_percent precisa ser maior ou igual a start_percent.")

        import comfy.patcher_extension

        patched = model.clone()
        sampling = patched.get_model_object("model_sampling")
        state = {}

        options = patched.model_options["transformer_options"] = patched.model_options.get(
            "transformer_options", {}
        ).copy()
        previous = options.get("optimized_attention_override")
        options["optimized_attention_override"] = _attention_override(
            previous,
            float(sampling.percent_to_sigma(float(start_percent))),
            float(sampling.percent_to_sigma(float(end_percent))),
            state,
        )

        wrappers = options["wrappers"] = {
            kind: dict(keyed) for kind, keyed in options.get("wrappers", {}).items()
        }
        bucket = wrappers.setdefault(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
        key = "bruxos_h3_reference_isolate"
        bucket[key] = [*bucket.get(key, []), _forward_wrapper(state)]

        report = (
            "Reference Isolate instalado\n"
            f"intervalo: {start_percent:.2f} -> {end_percent:.2f} do denoise\n"
            "alvo: continua vendo texto + referencias completas\n"
            "referencias: deixam de consultar o alvo ruidoso durante o intervalo\n"
            "preset inicial recomendado: 0.15 -> 1.00; use 0.00 apenas depois do A/B\n"
            "o log mostrara prefixo, sequencia total e economia teorica ao executar"
        )
        return patched, report


NODE_CLASS_MAPPINGS = {"BruxosH3ReferenceIsolate": BruxosH3ReferenceIsolate}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3ReferenceIsolate": "H3 Reference Isolate (Bruxos)"
}

