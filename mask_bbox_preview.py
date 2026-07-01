# -*- coding: utf-8 -*-
"""Bruxos do VFX - Prever BBox da Máscara.
Desenha a caixa (bbox) que o modo `mask_mode=bbox` do Bernini Infinity usaria,
com a MESMA conta (_normalize_mask -> _grow_blur_mask -> _mask_bbox). Serve pra
regular mask_grow/mask_pad e ver o % da área ANTES de rodar a geração."""

import logging

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# reaproveita os helpers reais do node (mesma conta do bbox)
try:
    from .nodes import _normalize_mask, _grow_blur_mask, _mask_bbox
    _HAS_HELPERS = True
except Exception as e:  # pragma: no cover
    logging.info(f"[Bruxos BBoxPreview] sem helpers do nodes.py ({e}); usando fallback")
    _HAS_HELPERS = False

ROXO = (0.659, 0.333, 0.969)   # #a855f7
VERDE = (0.133, 0.773, 0.369)  # #22c55e


def _fallback_norm(mask):
    m = mask
    if m.dim() == 4:
        m = m[..., :3].amax(dim=-1)
    elif m.dim() == 2:
        m = m.unsqueeze(0)
    return m.float().clamp(0.0, 1.0)


def _fallback_bbox(m, pad, stride, W, H, thr=0.02):
    any2d = (m.amax(dim=0) > thr)
    rows = torch.where(any2d.any(dim=1))[0]
    cols = torch.where(any2d.any(dim=0))[0]
    if rows.numel() == 0 or cols.numel() == 0:
        return 0, 0, int(W), int(H)
    y0 = int(rows.min()); y1 = int(rows.max()) + 1
    x0 = int(cols.min()); x1 = int(cols.max()) + 1
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(int(W), x1 + pad); y1 = min(int(H), y1 + pad)
    x0 -= x0 % stride; y0 -= y0 % stride
    if x1 % stride: x1 = min(int(W), x1 + (stride - x1 % stride))
    if y1 % stride: y1 = min(int(H), y1 + (stride - y1 % stride))
    return x0, y0, x1, y1


class BruxosMaskBBoxPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "region_mask": ("MASK,IMAGE", {"tooltip": "A máscara (branco = dentro). Aceita MASK ou máscara colorida (IMAGE)."}),
                "mask_grow": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1, "tooltip": "Mesmo valor do node: dilata (+) ou contrai (-) antes de medir a caixa."}),
                "mask_pad": ("INT", {"default": 16, "min": 0, "max": 1024, "step": 16, "tooltip": "Mesma folga do bbox ao redor da caixa (em pixels)."}),
                "stride": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1, "tooltip": "Alinhamento da caixa (múltiplo). O node usa 16."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Imagem/vídeo de fundo pra desenhar por cima (ex.: o source_video). Se vazio, usa fundo preto no tamanho da máscara."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "info")
    FUNCTION = "preview"
    CATEGORY = "Bruxos do VFX/Video"
    DESCRIPTION = ("Mostra a caixa (bbox) que o modo bbox recortaria, com a mesma conta do node. "
                   "Retângulo verde = recorte; roxo = máscara. A saída 'info' traz coordenadas, tamanho e % da área.")

    def preview(self, region_mask, mask_grow=0, mask_pad=16, stride=16, image=None):
        if not _HAS_TORCH:
            raise RuntimeError("[Bruxos BBoxPreview] torch indisponível.")
        norm = _normalize_mask if _HAS_HELPERS else _fallback_norm
        bbox = _mask_bbox if _HAS_HELPERS else _fallback_bbox

        m = norm(region_mask)                       # [T,H,W] 0..1
        if _HAS_HELPERS:
            m = _grow_blur_mask(m, int(mask_grow), 0)
        # resolução de referência
        if image is not None:
            H, W = int(image.shape[1]), int(image.shape[2])
            if m.shape[1] != H or m.shape[2] != W:
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(1).clamp(0, 1)
        else:
            H, W = int(m.shape[1]), int(m.shape[2])

        x0, y0, x1, y1 = bbox(m, int(mask_pad), int(stride), W, H)
        cw, ch = x1 - x0, y1 - y0
        area = 100.0 * (cw * ch) / max(1, W * H)
        info = f"bbox=({x0},{y0})-({x1},{y1}) | {cw}x{ch} de {W}x{H} | ~{area:.0f}% da área"
        logging.info(f"[Bruxos BBoxPreview] {info}")

        # fundo
        if image is not None:
            base = image.detach().clone().float()
            if base.shape[0] < m.shape[0]:
                base = base.repeat((m.shape[0] + base.shape[0] - 1)//base.shape[0], 1, 1, 1)[:m.shape[0]]
        else:
            base = torch.zeros((m.shape[0], H, W, 3), dtype=torch.float32)

        T = base.shape[0]
        mm = m[:T] if m.shape[0] >= T else m[:1].repeat(T, 1, 1)

        # tinta roxa na máscara (alpha 35%)
        purple = torch.tensor(ROXO, dtype=base.dtype).view(1, 1, 1, 3)
        a = (mm.unsqueeze(-1) * 0.35)
        base = base * (1 - a) + purple * a

        # retângulo verde
        green = torch.tensor(VERDE, dtype=base.dtype)
        th = max(2, W // 320)
        yb0, yb1 = max(0, y0), min(H, y1)
        xb0, xb1 = max(0, x0), min(W, x1)
        base[:, yb0:yb0+th, xb0:xb1, :] = green
        base[:, max(0, yb1-th):yb1, xb0:xb1, :] = green
        base[:, yb0:yb1, xb0:xb0+th, :] = green
        base[:, yb0:yb1, max(0, xb1-th):xb1, :] = green

        return (base.clamp(0, 1), info)


NODE_CLASS_MAPPINGS = {"BruxosMaskBBoxPreview": BruxosMaskBBoxPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosMaskBBoxPreview": "Prever BBox da Máscara (Bruxos)"}
