# -*- coding: utf-8 -*-
"""Bruxos do VFX - Comparar Vídeos (A/B) estilo Deno.
Recebe dois lotes de imagens (A e B), codifica em mp4 temporario e o widget JS
mostra o player de comparacao (cortina / lado a lado / diferenca / alternar).
Saida 'output' = passa A adiante sem alteracao (full-res).

Nota de memoria: a codificacao e feita FRAME A FRAME (streaming), sem nunca
materializar o video uint8 inteiro na RAM nem a copia float32 duplicada que o
'arr * 255.0' criava. Isso evita o estouro de RAM em videos grandes
(ex.: 121 frames a 2560x1408 gastavam ~15GB de pico; agora sao alguns MB por vez).
O video de PREVIEW ainda e reduzido para no maximo PREVIEW_MAXSIDE px no maior
lado -- o player de comparacao nao precisa de resolucao total, e isso deixa o
mp4 leve. A saida 'output' continua em resolucao total (passa A adiante)."""

import os
import time
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

# maior lado do video de PREVIEW (nao afeta a saida 'output', so o mp4 do player)
PREVIEW_MAXSIDE = 960


def _frame_count(images):
    try:
        return int(images.shape[0])
    except Exception:
        return len(images)


def _downscale_wh(w, h, maxside):
    """Calcula (w,h) reduzidos mantendo proporcao, com lados PARES (h264)."""
    m = max(w, h)
    if maxside and m > maxside:
        s = maxside / float(m)
        w = max(2, int(round(w * s)))
        h = max(2, int(round(h * s)))
    w -= w % 2
    h -= h % 2
    return max(2, w), max(2, h)


def _iter_uint8_frames(images, out_w, out_h):
    """Gera frames uint8 [H,W,C] um a um, convertendo e redimensionando so o
    frame corrente. Nunca segura o lote inteiro convertido na RAM.

    Redimensionamento: usa cv2 se disponivel (rapido, boa qualidade); senao cai
    para uma reamostragem por indices em numpy (sem dependencia extra)."""
    n = _frame_count(images)
    try:
        import cv2
        _has_cv2 = True
    except Exception:
        _has_cv2 = False

    for i in range(n):
        fr = images[i]
        # tensor torch -> numpy, so este frame
        if hasattr(fr, "detach"):
            fr = fr.detach().cpu().numpy()
        else:
            fr = np.asarray(fr)
        # float [0..1] -> uint8 [0..255], in-place-ish, sem copia global do lote
        fr = np.multiply(fr, 255.0, dtype=np.float32)
        np.clip(fr, 0, 255, out=fr)
        fr = fr.astype(np.uint8, copy=False)

        h0, w0 = fr.shape[0], fr.shape[1]
        if (w0, h0) != (out_w, out_h):
            if _has_cv2:
                interp = cv2.INTER_AREA if (out_w < w0 or out_h < h0) else cv2.INTER_LINEAR
                fr = cv2.resize(fr, (out_w, out_h), interpolation=interp)
            else:
                ys = (np.linspace(0, h0 - 1, out_h)).astype(np.int64)
                xs = (np.linspace(0, w0 - 1, out_w)).astype(np.int64)
                fr = fr[ys][:, xs]
        yield fr


def _encode_temp(images, fps, prefix, maxside=PREVIEW_MAXSIDE):
    """Codifica o lote em mp4 no diretorio temp, frame a frame. Retorna ref p/ /view."""
    if not _HAS_FP:
        raise RuntimeError("[Bruxos Compare] folder_paths indisponivel.")
    if _frame_count(images) == 0:
        raise RuntimeError("[Bruxos Compare] lote de imagens vazio.")

    tmp = folder_paths.get_temp_directory()
    os.makedirs(tmp, exist_ok=True)
    name = f"{prefix}_{int(time.time() * 1000)}.mp4"
    path = os.path.join(tmp, name)

    # dimensao de origem (so le a forma; nao converte nada ainda)
    h0, w0 = int(images.shape[1]), int(images.shape[2])
    out_w, out_h = _downscale_wh(w0, h0, maxside)
    if (out_w, out_h) != (w0, h0):
        logging.info(f"[Bruxos Compare] preview {w0}x{h0} -> {out_w}x{out_h} "
                     f"(reduzido p/ <= {maxside}px, lados pares)")
    fps = float(fps) if fps and fps > 0 else 24.0

    # backend 1: imageio-ffmpeg (h264 yuv420p, compativel com <video>)
    try:
        import imageio
        with imageio.get_writer(path, fps=fps, codec="libx264",
                                quality=7, macro_block_size=None,
                                ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
            for fr in _iter_uint8_frames(images, out_w, out_h):
                w.append_data(fr)
        return {"filename": name, "subfolder": "", "type": "temp", "format": "video/mp4"}
    except Exception as e:
        logging.info(f"[Bruxos Compare] imageio falhou ({e}); tentando cv2")

    # backend 2: OpenCV (tambem frame a frame)
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(path, fourcc, fps, (out_w, out_h))
        if not vw.isOpened():
            raise RuntimeError("VideoWriter nao abriu")
        for fr in _iter_uint8_frames(images, out_w, out_h):
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
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
                "preview_maxside": ("INT", {"default": PREVIEW_MAXSIDE, "min": 128, "max": 4096, "step": 32,
                                            "tooltip": "Maior lado do vídeo de preview (px). "
                                                       "Menor = mais leve. Não afeta a saída 'output'."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "Bruxos do VFX/Video"
    DESCRIPTION = ("Compara dois vídeos A/B no próprio node (cortina, lado a lado, "
                   "diferença, alternar), estilo Deno. A saída 'output' passa A "
                   "adiante em resolução total. Codificação frame a frame (baixa RAM). "
                   "Tem botão para abrir no navegador.")

    def compare(self, video_a, video_b, fps=24.0, preview_maxside=PREVIEW_MAXSIDE):
        ref_a = _encode_temp(video_a, fps, "bruxos_cmp_a", maxside=preview_maxside)
        ref_b = _encode_temp(video_b, fps, "bruxos_cmp_b", maxside=preview_maxside)
        logging.info(f"[Bruxos Compare] A={ref_a['filename']} B={ref_b['filename']} fps={fps}")
        return {"ui": {"bruxos_compare": [{"a": ref_a, "b": ref_b, "fps": float(fps)}]},
                "result": (video_a,)}


NODE_CLASS_MAPPINGS = {"BruxosVideoCompare": BruxosVideoCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosVideoCompare": "Comparar Vídeos A/B (Bruxos)"}
