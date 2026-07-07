# -*- coding: utf-8 -*-
"""Bruxos do VFX - Comparar Vídeos (A/B) estilo Deno.
Recebe dois lotes de imagens (A e B), codifica em mp4 temporario e o widget JS
mostra o player de comparacao (cortina / lado a lado / diferenca / alternar).
Saida 'output' = passa A adiante sem alteracao (full-res)."""

import os
import logging
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import folder_paths
    _HAS_FP = True
except Exception:
    folder_paths = None
    _HAS_FP = False


def _to_uint8(images):
    arr = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr  # [B,H,W,C]


def _encode_temp(images, fps, prefix):
    """Codifica o lote em mp4 no diretorio temp. Retorna ref p/ o /view."""
    if not _HAS_FP:
        raise RuntimeError("[Bruxos Compare] folder_paths indisponivel.")
    tmp = folder_paths.get_temp_directory()
    os.makedirs(tmp, exist_ok=True)
    import time
    name = f"{prefix}_{int(time.time()*1000)}.mp4"
    path = os.path.join(tmp, name)
    frames = _to_uint8(images)
    # libx264 (yuv420p) EXIGE largura e altura PARES. Se vier impar (ex.: 861x487
    # apos um resize/upscale), corta 1 px pra virar par -> evita
    # "width not divisible by 2" e "Could not open encoder".
    h0, w0 = frames.shape[1], frames.shape[2]
    h2, w2 = h0 - (h0 % 2), w0 - (w0 % 2)
    if (h2, w2) != (h0, w0):
        frames = frames[:, :h2, :w2, :]
        logging.info(f"[Bruxos Compare] dimensao impar {w0}x{h0} -> ajustada p/ {w2}x{h2} (par, exigencia do h264)")
    fps = float(fps) if fps and fps > 0 else 24.0

    # backend 1: imageio-ffmpeg (h264 yuv420p, compativel com <video>)
    try:
        import imageio
        with imageio.get_writer(path, fps=fps, codec="libx264",
                                quality=7, macro_block_size=None,
                                ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
            for f in frames:
                w.append_data(f)
        return {"filename": name, "subfolder": "", "type": "temp", "format": "video/mp4"}
    except Exception as e:
        logging.info(f"[Bruxos Compare] imageio falhou ({e}); tentando cv2")

    # backend 2: OpenCV
    try:
        import cv2
        h, wdt = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(path, fourcc, fps, (wdt, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        return {"filename": name, "subfolder": "", "type": "temp", "format": "video/mp4"}
    except Exception as e:
        raise RuntimeError(f"[Bruxos Compare] nao consegui codificar o video temporario: {e}")


class BruxosVideoCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_a": ("IMAGE", {"tooltip": "Vídeo A (ex.: original)."}),
                "video_b": ("IMAGE", {"tooltip": "Vídeo B (ex.: upscale)."}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.5,
                                  "tooltip": "FPS dos vídeos de preview."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "Bruxos do VFX/Video"
    DESCRIPTION = ("Compara dois vídeos A/B no próprio node (cortina, lado a lado, "
                   "diferença, alternar), estilo Deno. A saída 'output' passa A "
                   "adiante em resolução total. Tem botão para abrir no navegador.")

    def compare(self, video_a, video_b, fps=24.0):
        ref_a = _encode_temp(video_a, fps, "bruxos_cmp_a")
        ref_b = _encode_temp(video_b, fps, "bruxos_cmp_b")
        logging.info(f"[Bruxos Compare] A={ref_a['filename']} B={ref_b['filename']} fps={fps}")
        return {"ui": {"bruxos_compare": [{"a": ref_a, "b": ref_b, "fps": float(fps)}]},
                "result": (video_a,)}


NODE_CLASS_MAPPINGS = {"BruxosVideoCompare": BruxosVideoCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosVideoCompare": "Comparar Vídeos A/B (Bruxos)"}
