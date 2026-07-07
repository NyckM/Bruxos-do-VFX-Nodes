# -*- coding: utf-8 -*-
"""
Nodes de Tracking (Bruxos do VFX)
=================================
Camera / objeto / pontos com cara Bruxos: tooltip PT, categoria "Bruxos do VFX/
Tracking", timer automatico (selo/titulo). NAO reimplementa os modelos: PUXA os
trackers ja instalados (DROID-SLAM, DUSt3R, CoTracker...) via registro do ComfyUI
— mesmo padrao do node do Bernini. Se o motor nao estiver instalado, o node avisa
em portugues em vez de quebrar.

Fluxo:
    video -> [Bruxos Camera Tracker]  -> CAMERA_TRAJECTORY -> [Bruxos Tracking Export] -> arquivo p/ VFX
    video -> [Bruxos Point Tracker]   -> POINT_TRACKS      -> (uso interno / export)
    video -> [Bruxos Object Tracker]  -> OBJECT_TRACKS/mask -> (guiar efeito/mascara)
"""

import os
import json

try:
    import nodes as _comfy_nodes
except Exception:
    _comfy_nodes = None


# ---------------------------------------------------------------------------
# util: achar a classe de um node de terceiro pelo nome de registro
# ---------------------------------------------------------------------------
def _registry():
    return getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", {}) if _comfy_nodes else {}


def _get(*keys):
    """Retorna a 1a classe encontrada entre varias chaves candidatas (o autor
    pode ter registrado com nomes diferentes conforme a versao do pacote)."""
    reg = _registry()
    for k in keys:
        if k in reg:
            return reg[k], k
    return None, None


def _pt_faltando(nomes, pacote="pacote de tracking (nelsig_tracking_nodes)"):
    return (
        f"[Bruxos Tracking] Motor de tracking nao encontrado ({', '.join(nomes)}).\n"
        f"  -> Instale e ative o {pacote} e rode o instalar_modelos_tracking.\n"
        f"  -> Reinicie o ComfyUI. Este node so 'puxa' o tracker; ele nao vem embutido."
    )


class _AnyType(str):
    def __ne__(self, other):
        return False


ANY = _AnyType("*")


# ===========================================================================
# CAMERA
# ===========================================================================
class BruxosCameraTracker:
    # chaves candidatas do motor de camera, em ordem de preferencia
    _ALVOS = ["CameraTrackerAuto", "CameraTrackerDROID", "CameraTrackerDUSt3R", "CameraTrackerSLAM"]

    @classmethod
    def INPUT_TYPES(cls):
        cls_alvo, _ = _get(*cls._ALVOS)
        # herda os widgets do motor real, se disponivel (fica sempre igual ao original)
        if cls_alvo is not None:
            try:
                it = cls_alvo.INPUT_TYPES()
                it.setdefault("optional", {})
                it["optional"]["motor"] = (cls._ALVOS, {"default": cls._ALVOS[0],
                    "tooltip": "Qual tracker de camera usar (precisa estar instalado). Auto tenta o melhor disponivel."})
                return it
            except Exception:
                pass
        # fallback minimo quando o motor nao esta instalado (node ainda aparece)
        return {
            "required": {"images": ("IMAGE", {"tooltip": "Frames do video pra rastrear o movimento da camera."})},
            "optional": {
                "motor": (cls._ALVOS, {"default": cls._ALVOS[0], "tooltip": "Tracker de camera (instale o pacote de tracking)."}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64,
                    "tooltip": "Resolucao de processamento. Maior = mais preciso e mais lento."}),
            },
        }

    RETURN_TYPES = ("CAMERA_TRAJECTORY", "STRING")
    RETURN_NAMES = ("trajectory", "info")
    OUTPUT_TOOLTIPS = (
        "Trajetoria da camera (matrizes 4x4 por frame + intrinsics). Liga no Bruxos Tracking Export.",
        "Relatorio: quantos frames, motor usado, confianca.",
    )
    FUNCTION = "track"
    CATEGORY = "Bruxos do VFX/Tracking"
    DESCRIPTION = ("Rastreia o MOVIMENTO DA CAMERA no video (pra inserir CG/objeto na cena ou exportar "
                   "pro VFX). Puxa o tracker instalado (DROID-SLAM/DUSt3R); nao vem embutido.")

    def track(self, images=None, motor="CameraTrackerAuto", **kw):
        cls_alvo, achou = _get(motor, *self._ALVOS)
        if cls_alvo is None:
            msg = _pt_faltando(self._ALVOS)
            print(msg, flush=True)
            raise RuntimeError(msg)
        print(f"[Bruxos Camera Tracker] usando motor: {achou}", flush=True)
        inst = cls_alvo()
        fn = getattr(inst, getattr(cls_alvo, "FUNCTION", "track"))
        kw.pop("motor", None)
        out = fn(images=images, **kw) if images is not None else fn(**kw)
        # o motor devolve tupla; a trajetoria costuma ser o 1o item
        traj = out[0] if isinstance(out, (tuple, list)) else out
        n = 0
        try:
            m = traj.get("matrices") if isinstance(traj, dict) else None
            n = len(m) if m is not None else 0
        except Exception:
            pass
        info = f"motor={achou} | frames={n}"
        return (traj, info)


# ===========================================================================
# PONTOS
# ===========================================================================
class BruxosPointTracker:
    _ALVOS = ["PointTrackerCoTracker", "SpaTrackerSparse"]

    @classmethod
    def INPUT_TYPES(cls):
        cls_alvo, _ = _get(*cls._ALVOS)
        if cls_alvo is not None:
            try:
                it = cls_alvo.INPUT_TYPES()
                it.setdefault("optional", {})
                it["optional"]["motor"] = (cls._ALVOS, {"default": cls._ALVOS[0],
                    "tooltip": "Tracker de pontos a usar (precisa estar instalado)."})
                return it
            except Exception:
                pass
        return {
            "required": {"images": ("IMAGE", {"tooltip": "Frames do video."})},
            "optional": {"motor": (cls._ALVOS, {"default": cls._ALVOS[0],
                "tooltip": "Tracker de pontos (instale o pacote de tracking)."})},
        }

    RETURN_TYPES = ("POINT_TRACKS", "VISIBILITY", "STRING")
    RETURN_NAMES = ("tracks", "visibility", "info")
    OUTPUT_TOOLTIPS = (
        "Trajetorias 2D dos pontos por frame. Use p/ guiar efeito/mascara ou exportar.",
        "Visibilidade de cada ponto por frame (0/1).",
        "Relatorio do tracking.",
    )
    FUNCTION = "track"
    CATEGORY = "Bruxos do VFX/Tracking"
    DESCRIPTION = ("Rastreia PONTOS ao longo do video (CoTracker/SpaTracker). Puxa o tracker instalado; "
                   "nao vem embutido. Bom pra guiar efeito/mascara ou exportar trajetorias.")

    def track(self, images=None, motor="PointTrackerCoTracker", **kw):
        cls_alvo, achou = _get(motor, *self._ALVOS)
        if cls_alvo is None:
            msg = _pt_faltando(self._ALVOS)
            print(msg, flush=True)
            raise RuntimeError(msg)
        print(f"[Bruxos Point Tracker] usando motor: {achou}", flush=True)
        inst = cls_alvo()
        fn = getattr(inst, getattr(cls_alvo, "FUNCTION", "track"))
        kw.pop("motor", None)
        out = fn(images=images, **kw) if images is not None else fn(**kw)
        out = out if isinstance(out, (tuple, list)) else (out,)
        tracks = out[0] if len(out) > 0 else None
        vis = out[1] if len(out) > 1 else None

        # DIAGNOSTICO: imprime o formato real do que o motor devolveu, pra a gente
        # escrever o visualizador/export no formato certo (sem adivinhar).
        def _fmt(x, nome):
            try:
                import torch as _t
                if hasattr(x, "shape"):
                    extra = ""
                    try:
                        xf = x.float() if hasattr(x, "float") else x
                        extra = f" min={float(xf.min()):.3f} max={float(xf.max()):.3f}"
                    except Exception:
                        pass
                    return f"{nome}: {type(x).__name__} shape={tuple(x.shape)} dtype={getattr(x,'dtype','?')}{extra}"
                if isinstance(x, dict):
                    return f"{nome}: dict keys={list(x.keys())}"
                if isinstance(x, (list, tuple)):
                    return f"{nome}: {type(x).__name__} len={len(x)} [0]={type(x[0]).__name__ if x else '-'}"
                return f"{nome}: {type(x).__name__} = {repr(x)[:80]}"
            except Exception as e:
                return f"{nome}: <erro ao inspecionar: {e}>"

        print("[Bruxos Point Tracker] FORMATO DAS SAIDAS (mande isto pro Claude):", flush=True)
        print("   " + _fmt(tracks, "tracks"), flush=True)
        print("   " + _fmt(vis, "visibility"), flush=True)
        print(f"   (saidas totais do motor: {len(out)})", flush=True)

        return (tracks, vis, f"motor={achou}")


# ===========================================================================
# OBJETO
# ===========================================================================
class BruxosObjectTracker:
    _ALVOS = ["ObjectTrackerCoTracker", "SpaTrackerObject"]

    @classmethod
    def INPUT_TYPES(cls):
        cls_alvo, _ = _get(*cls._ALVOS)
        if cls_alvo is not None:
            try:
                it = cls_alvo.INPUT_TYPES()
                it.setdefault("optional", {})
                it["optional"]["motor"] = (cls._ALVOS, {"default": cls._ALVOS[0],
                    "tooltip": "Tracker de objeto a usar (precisa estar instalado)."})
                return it
            except Exception:
                pass
        return {
            "required": {"images": ("IMAGE", {"tooltip": "Frames do video."})},
            "optional": {"motor": (cls._ALVOS, {"default": cls._ALVOS[0],
                "tooltip": "Tracker de objeto (instale o pacote de tracking)."})},
        }

    RETURN_TYPES = ("OBJECT_TRACKS", "STRING")
    RETURN_NAMES = ("tracks", "info")
    OUTPUT_TOOLTIPS = ("Trajetoria do objeto rastreado por frame.", "Relatorio do tracking.")
    FUNCTION = "track"
    CATEGORY = "Bruxos do VFX/Tracking"
    DESCRIPTION = ("Rastreia um OBJETO ao longo do video. Puxa o tracker instalado; nao vem embutido.")

    def track(self, images=None, motor="ObjectTrackerCoTracker", **kw):
        cls_alvo, achou = _get(motor, *self._ALVOS)
        if cls_alvo is None:
            msg = _pt_faltando(self._ALVOS)
            print(msg, flush=True)
            raise RuntimeError(msg)
        print(f"[Bruxos Object Tracker] usando motor: {achou}", flush=True)
        inst = cls_alvo()
        fn = getattr(inst, getattr(cls_alvo, "FUNCTION", "track"))
        kw.pop("motor", None)
        out = fn(images=images, **kw) if images is not None else fn(**kw)
        out = out if isinstance(out, (tuple, list)) else (out,)
        return (out[0] if out else None, f"motor={achou}")


NODE_CLASS_MAPPINGS = {
    "BruxosCameraTracker": BruxosCameraTracker,
    "BruxosPointTracker": BruxosPointTracker,
    "BruxosObjectTracker": BruxosObjectTracker,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosCameraTracker": "Camera Tracker (Bruxos)",
    "BruxosPointTracker": "Point Tracker (Bruxos)",
    "BruxosObjectTracker": "Object Tracker (Bruxos)",
}
