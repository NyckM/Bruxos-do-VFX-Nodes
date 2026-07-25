# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Prompt Source (switch: manual / Qwen-VL / Florence2)
====================================================================
Um node so pra decidir DE ONDE vem o prompt do upscale/geracao:
  - manual   : usa o texto que voce digitar.
  - qwenvl   : auto-legenda com Qwen2.5-VL (reaproveita o BruxosQwenVLCaption).
  - florence2: auto-legenda com Florence-2 (carrega e roda os nodes instalados).

So o modo escolhido roda — os outros captioners nem carregam. Saida STRING
(pronta p/ CLIP Text Encode), com prefix/suffix opcionais.
"""

import logging

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Caption"

# reaproveita o caption Qwen ja existente no pacote
try:
    from .nodes import BruxosQwenVLCaption as _BX_QWEN
except Exception:  # pragma: no cover
    try:
        from nodes import BruxosQwenVLCaption as _BX_QWEN
    except Exception:
        _BX_QWEN = None

try:
    from .nodes import _bx_qwen_models_list  # se existir
except Exception:
    _bx_qwen_models_list = None

_QWEN_MODELS = None
try:
    # tenta puxar a mesma lista que o BruxosQwenVLCaption usa
    from . import nodes as _bxnodes
    _QWEN_MODELS = list(getattr(_bxnodes, "_BX_QWEN_MODELS", []))
except Exception:
    try:
        import nodes as _bxnodes
        _QWEN_MODELS = list(getattr(_bxnodes, "_BX_QWEN_MODELS", []))
    except Exception:
        _QWEN_MODELS = []
if not _QWEN_MODELS:
    _QWEN_MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]

_FLORENCE_MODELS = ["microsoft/Florence-2-large", "microsoft/Florence-2-base",
                    "microsoft/Florence-2-large-ft", "microsoft/Florence-2-base-ft"]
_FLORENCE_TASKS = ["more_detailed_caption", "detailed_caption", "caption"]


def _get_cls(name):
    try:
        import nodes as _core
        return getattr(_core, "NODE_CLASS_MAPPINGS", {}).get(name)
    except Exception:
        return None


def _call(cls, **kw):
    inst = cls()
    fn = getattr(inst, getattr(cls, "FUNCTION", "execute"))
    return fn(**kw)


def _first_string(out):
    """Extrai a primeira STRING de uma saida de node (tupla/dict/str)."""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        out = out.get("result", out)
    if isinstance(out, (tuple, list)):
        for v in out:
            if isinstance(v, str) and v.strip():
                return v
        # as vezes vem aninhado
        for v in out:
            s = _first_string(v)
            if s:
                return s
    return ""


# cache do modelo Florence carregado
_FL_CACHE = {}


def _florence_caption(images, model_name, task):
    dl = _get_cls("DownloadAndLoadFlorence2Model")
    run = _get_cls("Florence2Run")
    if dl is None or run is None:
        raise RuntimeError("nodes do Florence2 (comfyui-florence2) nao encontrados.")
    key = (model_name,)
    fl_model = _FL_CACHE.get(key)
    if fl_model is None:
        out = _call(dl, model=model_name, precision="fp16", attention="sdpa")
        fl_model = out[0] if isinstance(out, (tuple, list)) else out
        _FL_CACHE[key] = fl_model
    # 1o frame
    img = images[:1]
    # tenta a assinatura conhecida do Florence2Run; cai pra kwargs minimos
    tentativas = [
        dict(image=img, florence2_model=fl_model, text_input="", task=task,
             fill_mask=False, keep_model_loaded=True, max_new_tokens=1024,
             num_beams=3, do_sample=False, output_mask_select="", seed=1),
        dict(image=img, florence2_model=fl_model, text_input="", task=task),
    ]
    last = None
    for kw in tentativas:
        try:
            out = _call(run, **kw)
            s = _first_string(out)
            if s:
                return s
        except Exception as e:
            last = e
    raise RuntimeError(f"Florence2Run falhou: {last}")


class BruxosPromptSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["manual", "qwenvl", "florence2"], {"default": "manual",
                    "tooltip": "DE ONDE vem o prompt. manual = seu texto. qwenvl = auto-legenda Qwen2.5-VL. "
                               "florence2 = auto-legenda Florence-2. So o modo escolhido roda."}),
                "manual_prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Texto usado no modo manual (e como fallback se o captioner falhar)."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "[opcional] Se ligar o CLIP, a saida 'positive' ja sai como CONDITIONING (encodado) -> dispensa um CLIP Text Encode."}),
                "images": ("IMAGE", {"tooltip": "Frames pra auto-legenda (qwenvl/florence2). Ignorado no manual."}),
                "qwen_model": (_QWEN_MODELS, {"default": _QWEN_MODELS[0],
                    "tooltip": "[qwenvl] Modelo Qwen-VL (3B leve, 7B mais forte)."}),
                "florence_model": (_FLORENCE_MODELS, {"default": _FLORENCE_MODELS[0],
                    "tooltip": "[florence2] Modelo Florence-2."}),
                "florence_task": (_FLORENCE_TASKS, {"default": "more_detailed_caption",
                    "tooltip": "[florence2] Nivel de detalhe da legenda."}),
                "num_keyframes": ("INT", {"default": 6, "min": 1, "max": 32,
                    "tooltip": "[qwenvl] Quantos keyframes amostrar pra descrever o video inteiro num prompt."}),
                "max_tokens": ("INT", {"default": 220, "min": 16, "max": 2048,
                    "tooltip": "Tamanho maximo da legenda gerada."}),
                "prefix": ("STRING", {"default": "", "tooltip": "Texto colado ANTES da legenda (ex.: estilo)."}),
                "suffix": ("STRING", {"default": "", "tooltip": "Texto colado DEPOIS da legenda."}),
            },
        }

    RETURN_TYPES = ("STRING", "CONDITIONING")
    RETURN_NAMES = ("prompt", "positive")
    OUTPUT_TOOLTIPS = ("Prompt final (STRING).",
                       "CONDITIONING (so se o CLIP estiver ligado) — pronto pro sampler/upscale.")
    FUNCTION = "build"
    CATEGORY = CAT
    DESCRIPTION = (
        "Prompt Source (Bruxos): switch entre prompt MANUAL, auto-legenda QWEN-VL ou FLORENCE-2, "
        "num node so. So o modo escolhido carrega/roda o modelo. Saida STRING com prefix/suffix."
    )

    def build(self, mode, manual_prompt="", clip=None, images=None,
              qwen_model=None, florence_model="microsoft/Florence-2-large",
              florence_task="more_detailed_caption", num_keyframes=6, max_tokens=220,
              prefix="", suffix=""):
        cap = ""
        if mode == "manual":
            cap = manual_prompt or ""
        elif mode == "qwenvl":
            if images is None:
                print("[Bruxos Prompt Source] qwenvl sem 'images'; usando o prompt manual.", flush=True)
                cap = manual_prompt or ""
            elif _BX_QWEN is None:
                print("[Bruxos Prompt Source] BruxosQwenVLCaption indisponivel; usando manual.", flush=True)
                cap = manual_prompt or ""
            else:
                try:
                    out = _BX_QWEN().run(images=images, model_name=qwen_model,
                                         mode="keyframes_merge", num_keyframes=int(num_keyframes),
                                         max_new_tokens=int(max_tokens))
                    cap = _first_string(out) or manual_prompt or ""
                except Exception as e:
                    print(f"[Bruxos Prompt Source] Qwen-VL falhou ({e}); usando manual.", flush=True)
                    cap = manual_prompt or ""
        elif mode == "florence2":
            if images is None:
                print("[Bruxos Prompt Source] florence2 sem 'images'; usando manual.", flush=True)
                cap = manual_prompt or ""
            else:
                try:
                    cap = _florence_caption(images, florence_model, florence_task) or manual_prompt or ""
                except Exception as e:
                    print(f"[Bruxos Prompt Source] Florence2 falhou ({e}); usando manual.", flush=True)
                    cap = manual_prompt or ""

        final = (str(prefix) + (cap or "") + str(suffix)).strip()
        print(f"[Bruxos Prompt Source] modo={mode} -> {final[:120]!r}", flush=True)

        cond = None
        if clip is not None:
            try:
                cte = _get_cls("CLIPTextEncode")
                if cte is not None:
                    out = _call(cte, clip=clip, text=final)
                    cond = out[0] if isinstance(out, (tuple, list)) else out
                else:
                    tokens = clip.tokenize(final)
                    cond = [[clip.encode_from_tokens(tokens, return_pooled=False), {}]]
            except Exception as e:
                print(f"[Bruxos Prompt Source] encode CONDITIONING falhou ({e}); saida so STRING.", flush=True)
                cond = None
        return (final, cond)


NODE_CLASS_MAPPINGS = {"BruxosPromptSource": BruxosPromptSource}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosPromptSource": "Prompt Source: manual/Qwen/Florence (Bruxos)"}
