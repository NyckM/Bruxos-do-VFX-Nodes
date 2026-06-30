# -*- coding: utf-8 -*-
"""
Loader Tudo-em-1 (Bruxos) — carrega, num unico node:
  - modelo HIGH (safetensors via UNETLoader ou .gguf via UnetLoaderGGUF) + LoRA high
  - modelo LOW  (idem) + LoRA low
  - CLIP (safetensors via CLIPLoader ou .gguf via CLIPLoaderGGUF)
  - VAE
Reaproveita os nodes ja instalados (core + ComfyUI-GGUF), entao as listas de
arquivos e as opcoes ficam sempre iguais as dos loaders originais.
"""

import inspect
import json
import os
import struct

try:
    import folder_paths as _fp
except Exception:
    _fp = None

try:
    import nodes as _nodes
except Exception:
    _nodes = None


def _registry():
    return getattr(_nodes, "NODE_CLASS_MAPPINGS", {}) if _nodes else {}


def _get_cls(key):
    cls = _registry().get(key)
    if cls is None and _nodes is not None:
        cls = getattr(_nodes, key, None)
    return cls


def _opts(node_key, field, section="required"):
    """Le as opcoes (lista de arquivos / combos) direto do INPUT_TYPES do loader
    original, pra nunca ficar desatualizado em relacao a versao do ComfyUI."""
    cls = _get_cls(node_key)
    if cls is None:
        return []
    try:
        it = cls.INPUT_TYPES()
        entry = it.get(section, {}).get(field)
        if entry and isinstance(entry[0], (list, tuple)):
            return list(entry[0])
    except Exception:
        pass
    return []


def _dedupe(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _model_list():
    lst = _dedupe(_opts("UNETLoader", "unet_name") + _opts("UnetLoaderGGUF", "unet_name"))
    return lst if lst else ["None"]


def _clip_list():
    lst = _dedupe(_opts("CLIPLoader", "clip_name") + _opts("CLIPLoaderGGUF", "clip_name"))
    return lst if lst else ["None"]


def _vae_list():
    lst = _opts("VAELoader", "vae_name")
    return lst if lst else ["None"]


def _lora_list():
    return ["None"] + _opts("LoraLoaderModelOnly", "lora_name")


def _clip_types():
    t = _opts("CLIPLoader", "type")
    return t if t else ["wan"]


def _weight_dtypes():
    d = _opts("UNETLoader", "weight_dtype")
    return d if d else ["default"]


def _clip_devices():
    d = _opts("CLIPLoader", "device", section="optional")
    return d if d else ["default", "cpu"]


def _call(cls, **kwargs):
    """Instancia o loader e chama sua FUNCTION passando so os kwargs que ela aceita."""
    if cls is None:
        raise RuntimeError("loader nao encontrado")
    inst = cls()
    fn = getattr(inst, getattr(cls, "FUNCTION"))
    try:
        params = inspect.signature(fn).parameters
        accepted = {k: v for k, v in kwargs.items() if k in params}
    except (ValueError, TypeError):
        accepted = kwargs
    return fn(**accepted)


def _resolved_path(name, folder_keys):
    if _fp is None or not name:
        return None
    for k in folder_keys:
        try:
            p = _fp.get_full_path(k, name)
            if p:
                return p
        except Exception:
            pass
    return None


def _validate(name, folder_keys, kind):
    """Confere se o arquivo existe e tem formato valido. Erro claro em PT caso
    contrario (cobre o caso de node com valores antigos/desalinhados)."""
    if not name or name == "None":
        raise RuntimeError(f"[Bruxos Loader] nenhum {kind} selecionado.")
    p = _resolved_path(name, folder_keys)
    if not p or not os.path.isfile(p):
        raise RuntimeError(
            f"[Bruxos Loader] {kind} '{name}' nao encontrado. Em geral isso acontece "
            f"quando o node foi salvo numa versao anterior e os campos sairam de lugar: "
            f"APAGUE e RECRIE o node, e selecione os arquivos de novo."
        )
    low = name.lower()
    size = os.path.getsize(p)
    try:
        if low.endswith(".gguf"):
            with open(p, "rb") as f:
                if f.read(4) != b"GGUF":
                    raise ValueError("magic gguf invalido")
        elif low.endswith((".safetensors", ".sft")):
            with open(p, "rb") as f:
                head = f.read(8)
                if len(head) < 8:
                    raise ValueError("arquivo curto demais")
                n = struct.unpack("<Q", head)[0]
                if n <= 0 or 8 + n > size:
                    raise ValueError("tamanho de header invalido/incompleto")
                hdr = f.read(n)
                if len(hdr) < n:
                    raise ValueError("header truncado (download incompleto)")
                json.loads(hdr.decode("utf-8"))  # valida o JSON de verdade
    except Exception:
        raise RuntimeError(
            f"[Bruxos Loader] o arquivo de {kind} '{name}' ({os.path.basename(p)}, "
            f"{size} bytes) parece CORROMPIDO ou INCOMPLETO (header invalido). "
            f"Quase sempre e download pela metade: REBAIXE esse arquivo e tente de novo. "
            f"Se for um .gguf, escolha o loader/arquivo certo."
        )
    return p


def _default(seq, prefer=()):
    for p in prefer:
        if p in seq:
            return p
    return seq[0] if seq else "None"


def _is_gguf_path(path, name=""):
    """Detecta gguf pelos bytes magicos (mais confiavel que a extensao)."""
    if name and str(name).lower().endswith(".gguf"):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except Exception:
        return False


class BruxosWanAllInOneLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = _model_list()
        clips = _clip_list()
        vaes = _vae_list()
        loras = _lora_list()
        types = _clip_types()
        wdtypes = _weight_dtypes()
        devices = _clip_devices()
        return {
            "required": {
                "high_model": (models, {"tooltip": "Modelo de ruido ALTO (high noise). Aceita .safetensors ou .gguf."}),
                "high_lora": (loras, {"tooltip": "LoRA aplicada no modelo high. 'None' = sem LoRA."}),
                "high_lora_strength": ("FLOAT", {"default": 3.0, "min": -100.0, "max": 100.0, "step": 0.05, "tooltip": "Forca da LoRA high (seu fluxo usa ~3.0)."}),
                "low_model": (models, {"tooltip": "Modelo de ruido BAIXO (low noise). Aceita .safetensors ou .gguf."}),
                "low_lora": (loras, {"tooltip": "LoRA aplicada no modelo low. 'None' = sem LoRA."}),
                "low_lora_strength": ("FLOAT", {"default": 1.5, "min": -100.0, "max": 100.0, "step": 0.05, "tooltip": "Forca da LoRA low (seu fluxo usa ~1.5)."}),
                "clip_name": (clips, {"tooltip": "Text encoder / CLIP. Aceita .safetensors ou .gguf."}),
                "clip_type": (types, {"default": _default(types, ("wan",))}),
                "vae_name": (vaes, {"tooltip": "VAE."}),
            },
            "optional": {
                "weight_dtype": (wdtypes, {"default": _default(wdtypes, ("default",)),
                                           "tooltip": "Precisao do UNET safetensors (ignorado em .gguf)."}),
                "clip_device": (devices, {"default": _default(devices, ("default",)),
                                          "tooltip": "Dispositivo do CLIP (default/cpu)."}),
            },
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model_high", "model_low", "clip", "vae")
    FUNCTION = "load_all"
    CATEGORY = "Bruxos do VFX/Loaders"
    DESCRIPTION = ("Carrega tudo num node so: modelo high+LoRA, modelo low+LoRA, CLIP e VAE. "
                   "Entrada principal de modelos aceita safetensors OU gguf (detecta pela extensao "
                   "e usa o loader certo). Reaproveita os loaders ja instalados.")

    # ---- loaders internos ----
    def _load_unet(self, name, weight_dtype):
        p = _validate(name, ["diffusion_models", "unet", "unet_gguf", "diffusion_models_gguf"], "modelo")
        if _is_gguf_path(p, name):
            cls = _get_cls("UnetLoaderGGUF")
            if cls is None:
                raise RuntimeError("Pra carregar .gguf preciso do node ComfyUI-GGUF (UnetLoaderGGUF). "
                                   "Instale/atualize o ComfyUI-GGUF, ou selecione um .safetensors.")
            return _call(cls, unet_name=name)[0]
        cls = _get_cls("UNETLoader")
        if cls is None:
            raise RuntimeError("UNETLoader (core) nao encontrado.")
        return _call(cls, unet_name=name, weight_dtype=weight_dtype)[0]

    def _apply_lora(self, model, lora_name, strength):
        if not lora_name or lora_name == "None" or abs(float(strength)) < 1e-9:
            return model
        cls = _get_cls("LoraLoaderModelOnly")
        if cls is None:
            return model
        return _call(cls, model=model, lora_name=lora_name, strength_model=float(strength))[0]

    def _load_clip(self, clip_name, clip_type, clip_device):
        p = _validate(clip_name, ["text_encoders", "clip", "clip_gguf"], "CLIP/text encoder")
        if _is_gguf_path(p, clip_name):
            cls = _get_cls("CLIPLoaderGGUF")
            if cls is None:
                raise RuntimeError("Pra carregar CLIP .gguf preciso do node ComfyUI-GGUF (CLIPLoaderGGUF).")
            return _call(cls, clip_name=clip_name, type=clip_type, clip_type=clip_type)[0]
        cls = _get_cls("CLIPLoader")
        if cls is None:
            raise RuntimeError("CLIPLoader (core) nao encontrado.")
        return _call(cls, clip_name=clip_name, type=clip_type, device=clip_device)[0]

    def _load_vae(self, vae_name):
        _validate(vae_name, ["vae"], "VAE")
        cls = _get_cls("VAELoader")
        if cls is None:
            raise RuntimeError("VAELoader (core) nao encontrado.")
        return _call(cls, vae_name=vae_name)[0]

    def load_all(self, high_model, high_lora, high_lora_strength,
                 low_model, low_lora, low_lora_strength,
                 clip_name, clip_type, vae_name,
                 weight_dtype="default", clip_device="default"):
        model_high = self._apply_lora(self._load_unet(high_model, weight_dtype), high_lora, high_lora_strength)
        model_low = self._apply_lora(self._load_unet(low_model, weight_dtype), low_lora, low_lora_strength)
        clip = self._load_clip(clip_name, clip_type, clip_device)
        vae = self._load_vae(vae_name)
        return (model_high, model_low, clip, vae)


NODE_CLASS_MAPPINGS = {"BruxosWanAllInOneLoader": BruxosWanAllInOneLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosWanAllInOneLoader": "Loader Tudo-em-1 Wan (Bruxos)"}
