# -*- coding: utf-8 -*-
"""
Bruxos do VFX - Nodes utilitarios (Fase 1).
Substituem nodes de terceiros por equivalentes proprios, pra reduzir
dependencias de instalacao. Operacoes puras de tensor/string (sem modelos).

Cobre: GrowMaskWithBlur, BlockifyMask, DrawMaskOnImage (mascaras),
ImageSmartSharpen+ (imagem), StringFunction/JoinStrings/StringReplace/
SomethingToString/ShowText (texto), SeedGenerator (seed),
VHS_LoadImagesPath / VHS_VideoInfo (IO).
"""

import os
import math
import logging

import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image, ImageSequence, ImageOps
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    import folder_paths
    _HAS_FP = True
except Exception:
    folder_paths = None
    _HAS_FP = False

CAT = "Bruxos do VFX/Utilidades"


# ---------------------------------------------------------------------------
# helpers de mascara/imagem
# ---------------------------------------------------------------------------
def _gauss_kernel1d(radius):
    if radius <= 0:
        return None
    sigma = max(0.1, radius / 2.0)
    size = int(max(1, round(radius))) * 2 + 1
    x = torch.arange(size, dtype=torch.float32) - size // 2
    k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    return (k / k.sum())


def _blur2d(x, radius):
    """x: [B,C,H,W] -> blur gaussiano separavel."""
    k = _gauss_kernel1d(radius)
    if k is None:
        return x
    C = x.shape[1]
    pad = k.numel() // 2
    kh = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1).to(x.device, x.dtype)
    kw = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1).to(x.device, x.dtype)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, kh, groups=C)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kw, groups=C)
    return x


def _grow(mask, expand):
    """mask [B,H,W]. expand>0 dilata, <0 erode (px)."""
    if expand == 0:
        return mask
    m = mask.unsqueeze(1)
    r = abs(int(expand))
    k = r * 2 + 1
    if expand > 0:
        m = F.max_pool2d(m, k, stride=1, padding=r)
    else:
        m = 1.0 - F.max_pool2d(1.0 - m, k, stride=1, padding=r)
    return m.squeeze(1)


# ---------------------------------------------------------------------------
# 1) Crescer + Borrar Mascara  (substitui GrowMaskWithBlur do kjnodes)
# ---------------------------------------------------------------------------
class BruxosGrowMaskBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mascara de entrada."}),
                "expand": ("INT", {"default": 0, "min": -512, "max": 512,
                                   "tooltip": "Cresce (>0) ou encolhe (<0) a mascara, em pixels."}),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 200.0, "step": 0.5,
                                          "tooltip": "Raio do desfoque das bordas (suaviza a transicao)."}),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Inverte a mascara."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = ("Cresce/encolhe e borra a mascara. Substitui o GrowMaskWithBlur "
                   "(kjnodes). Use 'expand' p/ engordar a area e 'blur_radius' p/ "
                   "suavizar as bordas antes de compor.")

    def run(self, mask, expand=0, blur_radius=0.0, invert=False):
        m = mask.float()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if invert:
            m = 1.0 - m
        m = _grow(m, expand)
        if blur_radius > 0:
            m = _blur2d(m.unsqueeze(1), blur_radius).squeeze(1)
        return (m.clamp(0, 1),)


# ---------------------------------------------------------------------------
# 2) Mascara em Blocos  (substitui BlockifyMask do kjnodes)
# ---------------------------------------------------------------------------
class BruxosBlockifyMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mascara de entrada."}),
                "block_size": ("INT", {"default": 64, "min": 1, "max": 1024,
                                       "tooltip": "Tamanho do bloco (px). A mascara vira uma grade de blocos."}),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Acima deste valor o bloco e considerado ativo."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = ("Alinha a mascara a uma grade de blocos (util pra casar com tiles do "
                   "upscaler). Substitui o BlockifyMask (kjnodes).")

    def run(self, mask, block_size=64, threshold=0.0):
        m = mask.float()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        x = m.unsqueeze(1)
        B, _, H, W = x.shape
        bs = max(1, int(block_size))
        ph = (bs - H % bs) % bs
        pw = (bs - W % bs) % bs
        x = F.pad(x, (0, pw, 0, ph))
        pooled = F.max_pool2d(x, bs, stride=bs)
        if threshold > 0:
            pooled = (pooled > threshold).float()
        up = F.interpolate(pooled, scale_factor=bs, mode="nearest")
        up = up[:, :, :H, :W]
        return (up.squeeze(1).clamp(0, 1),)


# ---------------------------------------------------------------------------
# 3) Desenhar Mascara na Imagem  (substitui DrawMaskOnImage)
# ---------------------------------------------------------------------------
class BruxosDrawMaskOnImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Opacidade da cor sobre a area da mascara."}),
                "red": ("INT", {"default": 255, "min": 0, "max": 255}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255}),
                "blue": ("INT", {"default": 0, "min": 0, "max": 255}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = ("Pinta a area da mascara sobre a imagem com uma cor/opacidade "
                   "(visualizacao). Substitui o DrawMaskOnImage.")

    def run(self, image, mask, opacity=0.5, red=255, green=0, blue=0):
        img = image.float()
        m = mask.float()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        # casa H,W da mascara com a imagem
        H, W = img.shape[1], img.shape[2]
        if m.shape[1] != H or m.shape[2] != W:
            m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear",
                              align_corners=False).squeeze(1)
        # casa batch
        if m.shape[0] == 1 and img.shape[0] > 1:
            m = m.repeat(img.shape[0], 1, 1)
        col = torch.tensor([red, green, blue], dtype=torch.float32, device=img.device) / 255.0
        a = (m * opacity).unsqueeze(-1)
        out = img * (1 - a) + col.view(1, 1, 1, 3) * a
        return (out.clamp(0, 1),)


# ---------------------------------------------------------------------------
# 4) Nitidez Inteligente  (substitui ImageSmartSharpen+ do essentials)
# ---------------------------------------------------------------------------
class BruxosSmartSharpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05,
                                     "tooltip": "Intensidade da nitidez."}),
                "radius": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 50.0, "step": 0.5,
                                     "tooltip": "Raio do detalhe realcado."}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Ignora diferencas menores que isso (evita realcar ruido)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = ("Nitidez por mascara de contraste (unsharp mask) com limiar pra nao "
                   "realcar ruido. Substitui o ImageSmartSharpen+ (essentials).")

    def run(self, image, amount=0.5, radius=2.0, threshold=0.0):
        x = image.float().permute(0, 3, 1, 2)
        blurred = _blur2d(x, radius)
        high = x - blurred
        if threshold > 0:
            high = torch.where(high.abs() < threshold, torch.zeros_like(high), high)
        out = (x + amount * high).clamp(0, 1)
        return (out.permute(0, 2, 3, 1),)


# ---------------------------------------------------------------------------
# 5) Texto  (substitui StringFunction/JoinStrings/StringReplace/SomethingToString)
# ---------------------------------------------------------------------------
class BruxosText:
    MODES = ["juntar", "substituir", "formatar", "maiuscula", "minuscula", "remover_espacos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (cls.MODES, {"default": "juntar",
                                     "tooltip": "juntar: a+sep+b+sep+c | substituir: troca b por c em a | "
                                                "formatar: usa a como modelo com {} preenchidos por b,c"}),
                "a": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "b": ("STRING", {"default": "", "multiline": True}),
                "c": ("STRING", {"default": "", "multiline": True}),
                "separator": ("STRING", {"default": " "}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = ("Operacoes de texto num node so: juntar, substituir, formatar, "
                   "maiuscula/minuscula, remover espacos. Substitui StringFunction, "
                   "JoinStrings, StringReplace e SomethingToString.")

    def run(self, mode, a, b="", c="", separator=" "):
        a = "" if a is None else str(a)
        b = "" if b is None else str(b)
        c = "" if c is None else str(c)
        if mode == "juntar":
            parts = [p for p in (a, b, c) if p != ""]
            out = separator.join(parts)
        elif mode == "substituir":
            out = a.replace(b, c)
        elif mode == "formatar":
            try:
                out = a.format(b, c)
            except Exception:
                out = a
        elif mode == "maiuscula":
            out = a.upper()
        elif mode == "minuscula":
            out = a.lower()
        elif mode == "remover_espacos":
            out = a.strip()
        else:
            out = a
        return (out,)


# ---------------------------------------------------------------------------
# 6) Mostrar Texto  (substitui ShowText|pysssss / easy showAnything)
# ---------------------------------------------------------------------------
class BruxosShowText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True})}}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT
    DESCRIPTION = "Mostra o texto recebido no proprio node. Substitui ShowText / showAnything."

    def run(self, text):
        vals = [str(t) for t in (text if isinstance(text, list) else [text])]
        return {"ui": {"text": vals}, "result": (vals,)}


# ---------------------------------------------------------------------------
# 7) Seed  (substitui SeedGenerator do Easy-Use)
# ---------------------------------------------------------------------------
class BruxosSeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                              "tooltip": "Semente. Use o controle do ComfyUI (fixo/aleatorio)."})}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = "Gera/segura uma seed (INT). Substitui o SeedGenerator (Easy-Use)."

    def run(self, seed):
        return (int(seed),)


# ---------------------------------------------------------------------------
# 8) Carregar Imagens da Pasta  (substitui VHS_LoadImagesPath)
# ---------------------------------------------------------------------------
class BruxosLoadImagesPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "tooltip": "Caminho da pasta com a sequencia de imagens."}),
            },
            "optional": {
                "pattern": ("STRING", {"default": "*", "tooltip": "Filtro (ex.: *.png)."}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 100000,
                                           "tooltip": "Maximo de imagens (0 = todas)."}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = ("Carrega uma sequencia de imagens de uma pasta como batch IMAGE. "
                   "Substitui o VHS_LoadImagesPath.")

    def run(self, directory, pattern="*", frame_load_cap=0, skip_first_frames=0, select_every_nth=1):
        import glob
        if not _HAS_PIL:
            raise RuntimeError("[Bruxos] Pillow nao disponivel.")
        d = str(directory).strip().strip('"')
        if not os.path.isdir(d):
            raise FileNotFoundError(f"[Bruxos] pasta nao encontrada: {d}")
        files = sorted(glob.glob(os.path.join(d, pattern)))
        files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))]
        files = files[skip_first_frames::max(1, select_every_nth)]
        if frame_load_cap > 0:
            files = files[:frame_load_cap]
        if not files:
            raise FileNotFoundError(f"[Bruxos] nenhuma imagem em {d} (pattern={pattern})")
        arrs = []
        for f in files:
            im = Image.open(f)
            im = ImageOps.exif_transpose(im).convert("RGB")
            arrs.append(np.asarray(im, dtype=np.float32) / 255.0)
        h = min(a.shape[0] for a in arrs)
        w = min(a.shape[1] for a in arrs)
        arrs = [a[:h, :w, :] for a in arrs]
        batch = torch.from_numpy(np.stack(arrs, 0))
        return (batch, len(files))


# ---------------------------------------------------------------------------
# 9) Info do Video  (substitui VHS_VideoInfo)
# ---------------------------------------------------------------------------
class BruxosVideoInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_path": ("STRING", {"default": "", "tooltip": "Caminho do video."})}}

    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "fps", "frame_count", "duration", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = "Le largura, altura, fps, total de frames e duracao de um video. Substitui o VHS_VideoInfo."

    def run(self, video_path):
        p = str(video_path).strip().strip('"')
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[Bruxos] video nao encontrado: {p}")
        w = h = fc = 0
        fps = 0.0
        try:
            import cv2
            cap = cv2.VideoCapture(p)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception:
            import imageio.v3 as iio
            meta = iio.immeta(p, plugin="pyav")
            fps = float(meta.get("fps", 0) or 0)
        dur = (fc / fps) if fps else 0.0
        import json as _j
        info = _j.dumps({"width": w, "height": h, "fps": round(fps, 4),
                         "frame_count": fc, "duration": round(dur, 4)}, ensure_ascii=False)
        return (w, h, fps, fc, dur, info)


NODE_CLASS_MAPPINGS = {
    "BruxosGrowMaskBlur": BruxosGrowMaskBlur,
    "BruxosBlockifyMask": BruxosBlockifyMask,
    "BruxosDrawMaskOnImage": BruxosDrawMaskOnImage,
    "BruxosSmartSharpen": BruxosSmartSharpen,
    "BruxosText": BruxosText,
    "BruxosShowText": BruxosShowText,
    "BruxosSeed": BruxosSeed,
    "BruxosLoadImagesPath": BruxosLoadImagesPath,
    "BruxosVideoInfo": BruxosVideoInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosGrowMaskBlur": "Crescer + Borrar Mascara (Bruxos)",
    "BruxosBlockifyMask": "Mascara em Blocos (Bruxos)",
    "BruxosDrawMaskOnImage": "Desenhar Mascara na Imagem (Bruxos)",
    "BruxosSmartSharpen": "Nitidez Inteligente (Bruxos)",
    "BruxosText": "Texto (Bruxos)",
    "BruxosShowText": "Mostrar Texto (Bruxos)",
    "BruxosSeed": "Seed (Bruxos)",
    "BruxosLoadImagesPath": "Carregar Imagens da Pasta (Bruxos)",
    "BruxosVideoInfo": "Info do Video (Bruxos)",
}
