# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Load Image com crop-box + fit (crop/stretch/pad)
================================================================
Node de carregar imagem com:
  - fit_mode: off / crop / stretch / pad(letterbox)
  - target_width/target_height: resolucao de saida (0 = mantem)
  - aspect: proporcao do box de corte (livre / 1:1 / 3:4 / 4:3 / 16:9 / 9:16)
  - crop_x/y/w/h: retangulo normalizado (0..1) dirigido pelo box arrastavel (JS)

As funcoes _bx_apply_fit / _bx_resize_img sao reaproveitadas pelo Load Video.
"""

import os
import json
import logging

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

try:
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Loaders"

ASPECTS = ["livre", "1:1", "3:4", "4:3", "16:9", "9:16", "2:3", "3:2"]
FIT_MODES = ["off (original)", "crop", "stretch", "pad (letterbox)"]

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def _input_dir():
    if folder_paths is not None:
        try:
            return folder_paths.get_input_directory()
        except Exception:
            pass
    return os.getcwd()


def _list_input_images():
    d = _input_dir()
    out = []
    try:
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(d, f)):
                out.append(f)
    except Exception:
        pass
    return out


def _resolve_image_path(image, image_path):
    if image_path and str(image_path).strip():
        p = str(image_path).strip().strip('"')
        if os.path.isfile(p):
            return p
    # nome no diretorio input (pode ter subpasta "sub/arquivo.png")
    cand = os.path.join(_input_dir(), image)
    if os.path.isfile(cand):
        return cand
    raise RuntimeError(f"[Bruxos Load Image] imagem nao encontrada: {image!r} / {image_path!r}")


def _read_image_rgba(path):
    """Retorna (rgb float[H,W,3] 0..1, mask float[H,W] 0..1). Usa PIL; cai pra cv2."""
    try:
        from PIL import Image, ImageOps
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode == "P":
            img = img.convert("RGBA")
        has_alpha = img.mode in ("RGBA", "LA")
        rgb = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        if has_alpha:
            a = np.array(img.convert("RGBA"))[..., 3].astype(np.float32) / 255.0
            mask = 1.0 - a  # convencao comfy: mask = area transparente
        else:
            mask = np.zeros(rgb.shape[:2], dtype=np.float32)
        return rgb, mask
    except Exception:
        pass
    # fallback cv2
    import cv2
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"[Bruxos Load Image] falha ao ler {path}")
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        mask = np.zeros(raw.shape[:2], dtype=np.float32)
    elif raw.shape[2] == 4:
        a = raw[..., 3].astype(np.float32) / 255.0
        mask = 1.0 - a
        raw = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2RGB)
    else:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        mask = np.zeros(raw.shape[:2], dtype=np.float32)
    return raw.astype(np.float32) / 255.0, mask


# ---------------------------------------------------------------------------
# Fit / crop / resize (compartilhado com o Load Video)
# ---------------------------------------------------------------------------
def _bx_resize_img(t, tw, th):
    """t: torch [B,H,W,C] 0..1 -> redimensiona (stretch) pra (th,tw)."""
    B, H, W, C = t.shape
    if tw <= 0 or th <= 0 or (tw == W and th == H):
        return t
    x = t.permute(0, 3, 1, 2)
    mode = "area" if (tw < W or th < H) else "bicubic"
    if mode == "bicubic":
        x = F.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False)
    else:
        x = F.interpolate(x, size=(th, tw), mode="area")
    return x.permute(0, 2, 3, 1).clamp(0, 1)


def _bx_resize_mask(m, tw, th):
    B, H, W = m.shape
    if tw <= 0 or th <= 0 or (tw == W and th == H):
        return m
    x = m.unsqueeze(1)
    x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    return x.squeeze(1).clamp(0, 1)


def _target_from_one(W, H, tw, th):
    """Resolve o alvo quando so um lado e dado (mantem proporcao)."""
    if tw > 0 and th > 0:
        return tw, th
    if tw > 0:
        return tw, max(1, round(H * tw / W))
    if th > 0:
        return max(1, round(W * th / H)), th
    return W, H


def _bx_apply_fit(img, mask, fit_mode, cx, cy, cw, ch, tw, th):
    """
    img: [B,H,W,3] 0..1 ; mask: [B,H,W] 0..1.
    fit_mode: 'off'|'crop'|'stretch'|'pad'. cx..ch: retangulo normalizado 0..1.
    tw,th: alvo em px (0 = livre). Retorna (img, mask).
    """
    B, H, W, _ = img.shape
    fm = fit_mode.split()[0]  # "off" / "crop" / "stretch" / "pad"

    if fm == "off":
        return img, mask

    if fm == "crop":
        x0 = int(round(max(0.0, min(1.0, cx)) * W))
        y0 = int(round(max(0.0, min(1.0, cy)) * H))
        cwp = int(round(max(0.01, min(1.0, cw)) * W))
        chp = int(round(max(0.01, min(1.0, ch)) * H))
        x0 = max(0, min(W - 1, x0)); y0 = max(0, min(H - 1, y0))
        cwp = max(1, min(W - x0, cwp)); chp = max(1, min(H - y0, chp))
        img = img[:, y0:y0 + chp, x0:x0 + cwp, :]
        mask = mask[:, y0:y0 + chp, x0:x0 + cwp]
        if tw > 0 or th > 0:
            ntw, nth = _target_from_one(cwp, chp, tw, th)
            img = _bx_resize_img(img, ntw, nth)
            mask = _bx_resize_mask(mask, ntw, nth)
        return img, mask

    if fm == "stretch":
        if tw > 0 or th > 0:
            ntw, nth = _target_from_one(W, H, tw, th)
            img = _bx_resize_img(img, ntw, nth)
            mask = _bx_resize_mask(mask, ntw, nth)
        return img, mask

    if fm == "pad":
        if tw <= 0 and th <= 0:
            return img, mask
        ntw, nth = _target_from_one(W, H, tw, th)
        scale = min(ntw / W, nth / H)
        rw, rh = max(1, round(W * scale)), max(1, round(H * scale))
        img_r = _bx_resize_img(img, rw, rh)
        mask_r = _bx_resize_mask(mask, rw, rh)
        canvas = torch.zeros((B, nth, ntw, 3), dtype=img.dtype)
        mcanvas = torch.ones((B, nth, ntw), dtype=mask.dtype)  # padding = mascarado
        ox, oy = (ntw - rw) // 2, (nth - rh) // 2
        canvas[:, oy:oy + rh, ox:ox + rw, :] = img_r
        mcanvas[:, oy:oy + rh, ox:ox + rw] = mask_r
        return canvas, mcanvas

    return img, mask


# ===========================================================================
# NODE: Load Image (Bruxos)
# ===========================================================================
class BruxosLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_input_images()
        inputs = {
            "required": {
                "image": (files if files else ["(coloque imagens em ComfyUI/input)"],
                          {"tooltip": "Imagem da pasta ComfyUI/input. Use o botao de upload."}),
            },
            "optional": {
                # ---- secao FIT (o JS desenha as divisorias) ----
                "fit_mode": (FIT_MODES, {"default": "off (original)",
                    "tooltip": "off = original. crop = corta pelo box. stretch = estica pro alvo. pad = encaixa com bordas pretas."}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Largura de saida (px). 0 = mantem. Se so um lado, mantem proporcao."}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Altura de saida (px). 0 = mantem."}),
                # ---- secao CROP-BOX ----
                "aspect": (ASPECTS, {"default": "livre",
                    "tooltip": "Proporcao travada do box de corte. 'livre' = arrasta a vontade."}),
                "crop_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Canto esquerdo do box (0..1). Movido pelo box arrastavel."}),
                "crop_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Topo do box (0..1)."}),
                "crop_w": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.001,
                    "tooltip": "Largura do box (0..1)."}),
                "crop_h": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.001,
                    "tooltip": "Altura do box (0..1)."}),
                "image_path": ("STRING", {"default": "",
                    "tooltip": "Caminho absoluto (tem prioridade sobre o seletor)."}),
                # ---------------------------------------------------------
                # APPEND-ONLY: widget NOVO vai no FIM. O ComfyUI casa os
                # widgets_values salvos por ORDEM, nao por nome.
                # ---------------------------------------------------------
                "girar": (["off", "90 (horario)", "-90 (anti-horario)", "180"],
                    {"default": "off", "tooltip":
                    "Gira a imagem ANTES do fit/crop: girar troca largura por altura, e o box "
                    "de corte precisa ser calculado ja na orientacao final.\n\n"
                    "Serve para foto de celular que abre deitada -- o arquivo guarda o quadro "
                    "na horizontal mais uma FLAG EXIF de rotacao, e quem ignora a flag entrega "
                    "virado.\n\n"
                    "A MASCARA gira junto. '-180' nao existe: e o mesmo que 180."}),
                "flip_horizontal": ("BOOLEAN", {"default": False,
                    "tooltip": "Espelha a imagem da esquerda para a direita. A mascara acompanha."}),
                "flip_vertical": ("BOOLEAN", {"default": False,
                    "tooltip": "Espelha a imagem de cima para baixo. A mascara acompanha."}),
            },
        }
        # O frontend do ComfyUI cria o controle Hide/Show advanced inputs a
        # partir deste metadado. Isso funciona tanto no renderer LiteGraph
        # (Nodes 1.0) quanto no Vue (Nodes 2.0), sem inserir um widget extra em
        # widgets_values e, portanto, sem deslocar valores de workflows salvos.
        for spec in inputs["optional"].values():
            if len(spec) > 1 and isinstance(spec[1], dict):
                spec[1]["advanced"] = True
        return inputs

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "load"
    CATEGORY = CAT
    DESCRIPTION = ("Carrega imagem com box de corte arrastavel (proporcoes 1:1, 3:4, 16:9, 9:16...), "
                   "e modo fit: crop / stretch / pad. Preview em tempo real do recorte no node.")

    def load(self, image, fit_mode="off (original)", target_width=0, target_height=0,
             aspect="livre", crop_x=0.0, crop_y=0.0, crop_w=1.0, crop_h=1.0, image_path="",
             girar="off", flip_horizontal=False, flip_vertical=False):
        if not _OK:
            raise RuntimeError("[Bruxos Load Image] torch indisponivel neste build.")
        path = _resolve_image_path(image, image_path)
        rgb, mask = _read_image_rgba(path)
        img_t = torch.from_numpy(rgb).unsqueeze(0)          # [1,H,W,3]
        mask_t = torch.from_numpy(mask).unsqueeze(0)         # [1,H,W]

        # ---- GIRO: antes do fit/crop, e a mascara vai junto ----
        # rot90 k=1 e ANTI-HORARIO, entao horario e k=-1. Os rotulos dizem a
        # direcao por extenso porque "90" sozinho e ambiguo entre ferramentas.
        _g = str(girar or "off")
        if not _g.startswith("off"):
            _k = -1 if _g.startswith("90") else (1 if _g.startswith("-90") else 2)
            antes = (int(img_t.shape[1]), int(img_t.shape[2]))
            img_t = torch.rot90(img_t, _k, (1, 2)).contiguous()
            mask_t = torch.rot90(mask_t, _k, (1, 2)).contiguous()
            print(f"[Bruxos Load Image] girar {_g}: "
                  f"{antes[1]}x{antes[0]} -> {img_t.shape[2]}x{img_t.shape[1]}", flush=True)

        # ---- FLIP: ainda antes do crop/fit, exatamente como no preview ----
        if bool(flip_horizontal):
            img_t = torch.flip(img_t, dims=(2,)).contiguous()
            mask_t = torch.flip(mask_t, dims=(2,)).contiguous()
        if bool(flip_vertical):
            img_t = torch.flip(img_t, dims=(1,)).contiguous()
            mask_t = torch.flip(mask_t, dims=(1,)).contiguous()

        img_t, mask_t = _bx_apply_fit(
            img_t, mask_t, fit_mode,
            float(crop_x), float(crop_y), float(crop_w), float(crop_h),
            int(target_width), int(target_height),
        )
        H, W = int(img_t.shape[1]), int(img_t.shape[2])
        return (img_t, mask_t, W, H)

    @classmethod
    def IS_CHANGED(cls, image, image_path="", **kw):
        try:
            p = _resolve_image_path(image, image_path)
            return os.path.getmtime(p)
        except Exception:
            return float("nan")


NODE_CLASS_MAPPINGS = {"BruxosLoadImage": BruxosLoadImage}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosLoadImage": "Load Image + Crop (Bruxos)"}
