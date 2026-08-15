# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Tracked Crop & Stitch (SAM3 / qualquer MASK trackeada)
=====================================================================
Recorta uma janela que SEGUE o objeto trackeado (por frame). Dois modos:

  * FIXO   : janela de tamanho fixo que so acompanha a POSICAO (o objeto fica
             do tamanho que aparece — pequeno quando longe).
  * ZOOM   : a janela acompanha TAMBEM o TAMANHO do bbox (aperta quando o objeto
             esta longe) e cada frame e ampliado pra uma resolucao de saida
             uniforme -> o objeto fica GRANDE e do mesmo tamanho o tempo todo.
             Ideal pra dar upscale e depois colar de volta na posicao/tamanho
             que estava.

Fluxo:  SAM3 (mask) + video -> [Tracked Crop] -> recorte uniforme -> upscale ->
        [Tracked Stitch] -> video final (cada frame reencaixado no lugar/escala).
"""

import time
import math
import logging

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from .nodes import (
        _normalize_mask as _bx_normalize_mask,
        _grow_blur_mask as _bx_grow_blur,
        _rect_feather_mask as _bx_rect_feather,
    )
    _HAS = True
except Exception:  # pragma: no cover
    try:
        from mocha_tiled_nodes import _bx_normalize_mask, _bx_grow_blur, _bx_rect_feather
        _HAS = True
    except Exception as e:
        logging.info(f"[Bruxos Tracked] helpers indisponiveis ({e}); fallback local")
        _HAS = False

        def _bx_normalize_mask(mask):
            if mask is None:
                return None
            m = mask
            if m.dim() == 4:
                m = m[..., :3].amax(dim=-1)
            elif m.dim() == 2:
                m = m.unsqueeze(0)
            return m.float().clamp(0.0, 1.0)

        def _bx_grow_blur(m, grow=0, blur=0):
            x = m.unsqueeze(1)
            grow = int(grow)
            if grow > 0:
                x = torch.nn.functional.max_pool2d(x, grow * 2 + 1, stride=1, padding=grow)
            elif grow < 0:
                g = -grow
                x = -torch.nn.functional.max_pool2d(-x, g * 2 + 1, stride=1, padding=g)
            blur = int(blur)
            if blur > 0:
                k = blur * 2 + 1
                coords = torch.arange(k, dtype=torch.float32, device=m.device) - blur
                sigma = blur * 0.5 + 1e-6
                g1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma)); g1d = g1d / g1d.sum()
                x = torch.nn.functional.conv2d(x, g1d.view(1, 1, k, 1), padding=(blur, 0))
                x = torch.nn.functional.conv2d(x, g1d.view(1, 1, 1, k), padding=(0, blur))
            return x.squeeze(1).clamp(0.0, 1.0)

        def _bx_rect_feather(n, ch, cw, feather, device=None):
            device = device or torch.device("cpu")
            ys = torch.arange(ch, dtype=torch.float32, device=device).view(ch, 1)
            xs = torch.arange(cw, dtype=torch.float32, device=device).view(1, cw)
            f = max(1, int(feather))
            edge = torch.minimum(torch.minimum(ys, (ch - 1) - ys), torch.minimum(xs, (cw - 1) - xs))
            return (edge / float(f)).clamp(0, 1).unsqueeze(0).expand(n, ch, cw).contiguous()

CAT = "Bruxos do VFX/Tracking"


def _fmt_t(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{s - 60 * int(s // 60):04.1f}s"


def _norm_mask_full(mask, T, H, W):
    if mask is None:
        return None
    m = _bx_normalize_mask(mask)
    if int(m.shape[1]) != H or int(m.shape[2]) != W:
        m = torch.nn.functional.interpolate(m.unsqueeze(1), size=(H, W),
                                            mode="bilinear", align_corners=False).squeeze(1).clamp(0, 1)
    if int(m.shape[0]) < T:
        m = torch.cat([m, m[-1:].repeat(T - int(m.shape[0]), 1, 1)], dim=0)
    return m[:T].clamp(0, 1)


def _frame_bbox(m2d, thr=0.02):
    ys = torch.where(m2d.amax(dim=1) > thr)[0]
    xs = torch.where(m2d.amax(dim=0) > thr)[0]
    if ys.numel() == 0 or xs.numel() == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _smooth_series(vals, win):
    win = max(1, int(win))
    if win <= 1 or len(vals) <= 2:
        return list(vals)
    half = win // 2
    out = []
    n = len(vals)
    for i in range(n):
        a = max(0, i - half); b = min(n, i + half + 1)
        out.append(sum(vals[a:b]) / (b - a))
    return out


def _resize_bhwc(t, tw, th):
    """t [n,h,w,c] -> [n,th,tw,c] bilinear."""
    if int(t.shape[1]) == th and int(t.shape[2]) == tw:
        return t
    x = t.permute(0, 3, 1, 2)
    x = torch.nn.functional.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).clamp(0, 1)


# =============================================================================
class BruxosTrackedCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video de entrada completo."}),
                "mask": ("MASK,IMAGE", {"tooltip": "Mascara TRACKEADA do objeto (SAM3/SCAIL...). Deve variar por frame conforme o objeto se move."}),
            },
            "optional": {
                "size_mode": (["zoom (acompanha o tamanho)", "fixo (so posicao)"], {"default": "zoom (acompanha o tamanho)",
                    "tooltip": "ZOOM: a janela aperta no bbox por frame e amplia pro tamanho de saida -> objeto sempre GRANDE (mesmo longe). FIXO: janela de tamanho fixo, so segue a posicao (objeto fica pequeno quando longe)."}),
                "out_width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 16,
                    "tooltip": "[zoom] Largura da saida uniforme (pra onde cada recorte apertado e ampliado). [fixo] ignorado se crop_size>0."}),
                "out_height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 16,
                    "tooltip": "[zoom] Altura da saida uniforme."}),
                "padding": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Folga ao redor do bbox, como FRACAO do lado do bbox (0.35 = +35% de contexto). No zoom, controla quanto do fundo entra junto do objeto."}),
                "crop_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16,
                    "tooltip": "[fixo] Tamanho fixo da janela (px). 0 = auto (maior bbox + padding). No zoom nao e usado."}),
                "smooth": ("INT", {"default": 7, "min": 0, "max": 121, "step": 1,
                    "tooltip": "Suavizacao temporal (frames) do centro E do tamanho da janela. Reduz tremido/zoom nervoso quando o tracking oscila. 5-15 e bom."}),
                "align": ("INT", {"default": 16, "min": 2, "max": 64, "step": 2,
                    "tooltip": "Alinha o tamanho da saida a um multiplo disto (Wan VAE precisa de 16)."}),
                "mask_grow": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1,
                    "tooltip": "Dilata (+) / contrai (-) a mascara antes de medir o bbox."}),
                # APPEND-ONLY: funcoes Face Refine, sem deslocar workflows antigos.
                "center_smooth": ("INT", {"default": 21, "min": 0, "max": 241, "step": 2,
                    "tooltip": "Suavizacao separada do CENTRO. 21 em 24fps reduz tremor sem atrasar demais o rosto."}),
                "size_smooth": ("INT", {"default": 51, "min": 0, "max": 301, "step": 2,
                    "tooltip": "Suavizacao do TAMANHO. Deve ser maior para o crop nao respirar/pulsar."}),
                "canvas_mode": (["manual", "auto_no_downscale", "auto_capped_768"], {"default": "manual",
                    "tooltip": "Auto escolhe canvas pelo maior crop. no_downscale nunca descarta detalhe; capped limita o custo a 768."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "TRACK_CROP", "STRING")
    RETURN_NAMES = ("cropped_video", "cropped_mask", "crop_data", "info")
    OUTPUT_TOOLTIPS = (
        "Recorte uniforme (no zoom, o objeto fica sempre grande/centralizado). Ligue no upscale.",
        "Mascara recortada alinhada.",
        "Geometria por-frame (posicao E tamanho) + video original. Ligue no Tracked Stitch.",
        "Relatorio.",
    )
    FUNCTION = "crop"
    CATEGORY = CAT
    DESCRIPTION = (
        "Tracked Crop (Bruxos): recorta seguindo o objeto trackeado (SAM3). No modo ZOOM a janela "
        "acompanha o TAMANHO do bbox (aperta quando o objeto esta longe) e amplia pra saida uniforme "
        "-> o objeto fica grande o tempo todo. O Tracked Stitch reencaixa na posicao e escala originais."
    )

    def crop(self, images, mask, size_mode="zoom (acompanha o tamanho)",
             out_width=512, out_height=512, padding=0.35, crop_size=0,
             smooth=7, align=16, mask_grow=0, center_smooth=21,
             size_smooth=51, canvas_mode="manual"):
        if torch is None:
            raise RuntimeError("[Bruxos Tracked] torch indisponivel.")
        t0 = time.time()
        T = int(images.shape[0]); H = int(images.shape[1]); W = int(images.shape[2])
        m = _norm_mask_full(mask, T, H, W)
        if m is None:
            m = torch.ones((T, H, W), dtype=torch.float32)
        if int(mask_grow) != 0:
            m = _bx_grow_blur(m, int(mask_grow), 0)

        zoom = str(size_mode).startswith("zoom")

        def _al(v):
            a = max(2, int(align))
            return int(max(a, round(float(v) / a) * a))

        # bbox por frame -> centro + tamanho (carrega o ultimo se sumir)
        cxs, cys, bws, bhs, detected = [], [], [], [], []
        last = (W / 2.0, H / 2.0, W * 0.3, H * 0.3)
        found = 0
        for t in range(T):
            bb = _frame_bbox(m[t])
            if bb is None:
                cx, cy, bw, bh = last
                detected.append(0.0)
            else:
                x0, y0, x1, y1 = bb
                bw, bh = x1 - x0, y1 - y0
                cx, cy = x0 + bw / 2.0, y0 + bh / 2.0
                last = (cx, cy, bw, bh); found += 1
                detected.append(1.0)
            cxs.append(cx); cys.append(cy); bws.append(max(4, bw)); bhs.append(max(4, bh))

        cs = int(center_smooth) if int(center_smooth) > 0 else int(smooth)
        ss = int(size_smooth) if int(size_smooth) > 0 else int(smooth)
        cxs = _smooth_series(cxs, cs); cys = _smooth_series(cys, cs)
        bws = _smooth_series(bws, ss); bhs = _smooth_series(bhs, ss)
        face_heights = list(bhs)

        windows = []          # (x0,y0,w,h) por frame na resolucao ORIGINAL
        crops = []; mcrops = []

        if zoom:
            out_w, out_h = _al(out_width), _al(out_height)
            ar = out_w / float(out_h)
            pad = float(padding)
            raw_dims = []
            for t in range(T):
                bw = bws[t] * (1.0 + 2 * pad); bh = bhs[t] * (1.0 + 2 * pad)
                rw = max(bw, bh * ar); rh = rw / ar
                raw_dims.append((min(rw, W), min(rh, H)))
            if str(canvas_mode) != "manual":
                need_w = max(v[0] for v in raw_dims); need_h = max(v[1] for v in raw_dims)
                cw = max(need_w, need_h * ar); ch = cw / ar
                if str(canvas_mode) == "auto_capped_768":
                    scale = min(1.0, 768.0 / max(cw, ch))
                    cw *= scale; ch *= scale
                a = max(2, int(align))
                # Auto nunca arredonda para baixo: isso introduziria um
                # downscale pequeno mesmo no modo explicitamente no-downscale.
                out_w = int(math.ceil(cw / a) * a)
                out_h = int(math.ceil(ch / a) * a)
            for t in range(T):
                bw = bws[t] * (1.0 + 2 * pad)
                bh = bhs[t] * (1.0 + 2 * pad)
                # janela na proporcao da saida, cobrindo o bbox+padding
                w = max(bw, bh * ar)
                h = w / ar
                w = min(w, W); h = min(h, H)
                x0 = int(round(cxs[t] - w / 2.0)); y0 = int(round(cys[t] - h / 2.0))
                wi = int(round(w)); hi = int(round(h))
                x0 = max(0, min(W - wi, x0)); y0 = max(0, min(H - hi, y0))
                wi = max(2, min(W - x0, wi)); hi = max(2, min(H - y0, hi))
                windows.append((x0, y0, wi, hi))
                sub = images[t:t + 1, y0:y0 + hi, x0:x0 + wi, :]
                msub = m[t:t + 1, y0:y0 + hi, x0:x0 + wi]
                crops.append(_resize_bhwc(sub, out_w, out_h)[0])
                mcrops.append(_resize_bhwc(msub.unsqueeze(-1), out_w, out_h)[0, :, :, 0])
        else:
            # FIXO: janela unica, so posicao
            if int(crop_size) > 0:
                cw = ch = _al(crop_size)
            else:
                cw = ch = _al(max(max(bws) + 2 * 0.2 * max(bws), max(bhs) + 2 * 0.2 * max(bhs)))
            cw = min(cw, W); ch = min(ch, H)
            out_w, out_h = cw, ch
            for t in range(T):
                x0 = int(round(cxs[t] - cw / 2.0)); y0 = int(round(cys[t] - ch / 2.0))
                x0 = max(0, min(W - cw, x0)); y0 = max(0, min(H - ch, y0))
                windows.append((x0, y0, cw, ch))
                crops.append(images[t, y0:y0 + ch, x0:x0 + cw, :])
                mcrops.append(m[t, y0:y0 + ch, x0:x0 + cw])

        cropped = torch.stack(crops, 0).contiguous()
        cropped_mask = torch.stack(mcrops, 0).contiguous()

        crop_data = {
            "orig_video": images.cpu(),
            "windows": windows,            # (x0,y0,w,h) por frame (res original)
            "out_w": int(out_w), "out_h": int(out_h),
            "H": int(H), "W": int(W), "T": int(T),
            "mask_crop": cropped_mask.cpu(),
            "zoom": bool(zoom),
            "face_heights": face_heights,
            "detected": detected,
            "center_smooth": cs, "size_smooth": ss,
        }
        magnification = min(out_h / max(float(v[3]), 1.0) for v in windows)
        info = (f"{'ZOOM' if zoom else 'FIXO'} -> saida {out_w}x{out_h} x{T}f | "
                f"{found}/{T} detectados | face {min(face_heights):.0f}-{max(face_heights):.0f}px | "
                f"magnificacao min={magnification:.2f}x | centro={cs} tamanho={ss} | "
                f"canvas={canvas_mode} | {_fmt_t(time.time() - t0)}")
        print(f"[Bruxos Tracked Crop] {info}", flush=True)
        return (cropped, cropped_mask, crop_data, info)


# =============================================================================
class BruxosTrackedStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_data": ("TRACK_CROP", {"tooltip": "Saida 'crop_data' do Tracked Crop (Bruxos)."}),
                "edited_crop": ("IMAGE", {"tooltip": "O recorte JA processado (upscalado). Reencaixado na posicao e escala de cada frame."}),
            },
            "optional": {
                "compose": (["rectangle", "silhouette"], {"default": "rectangle",
                    "tooltip": "rectangle = retangulo da janela com feather (recomendado). silhouette = usa a mascara do objeto como alpha (so o sujeito volta)."}),
                "feather": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Suavidade da borda ao colar (px, na resolucao original). Maior = transicao mais macia."}),
                "colour_match": ("BOOLEAN", {"default": True,
                    "tooltip": "Casa media e contraste do crop refinado com a regiao original antes da colagem."}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    OUTPUT_TOOLTIPS = ("Video final: recorte colado na posicao E escala de cada frame.", "Relatorio.")
    FUNCTION = "stitch"
    CATEGORY = CAT
    DESCRIPTION = (
        "Tracked Stitch (Bruxos): reencaixa o recorte (upscalado) na posicao E no tamanho que ele "
        "ocupava em cada frame (desfaz o zoom), com feather. Corrige contagem de frames."
    )

    def stitch(self, crop_data, edited_crop, compose="rectangle", feather=24,
               colour_match=True, blend=1.0):
        if torch is None:
            raise RuntimeError("[Bruxos Tracked] torch indisponivel.")
        t0 = time.time()
        H, W, T = crop_data["H"], crop_data["W"], crop_data["T"]
        windows = crop_data["windows"]
        orig = crop_data["orig_video"].float().clamp(0, 1)

        edited = edited_crop.float().clamp(0, 1)
        Te = int(edited.shape[0])
        if Te > T:
            edited = edited[:T]
        elif Te < T:
            edited = torch.cat([edited, edited[-1:].repeat(T - Te, 1, 1, 1)], dim=0)

        mask_all = crop_data.get("mask_crop", None)
        final = orig.clone()
        for t in range(T):
            x0, y0, w, h = windows[t]
            # reduz o frame editado (tamanho de saida) de volta ao tamanho da janela
            piece = _resize_bhwc(edited[t:t + 1], w, h)[0]
            if compose == "silhouette" and mask_all is not None:
                a = mask_all[t] if int(mask_all.shape[0]) > t else mask_all[-1]
                a = _resize_bhwc(a.view(1, a.shape[0], a.shape[1], 1), w, h)[0, :, :, 0]
                if int(feather) > 0:
                    a = _bx_grow_blur(a.unsqueeze(0), 0, int(feather))[0]
                a = a.unsqueeze(-1)
            else:
                a = _bx_rect_feather(1, h, w, max(1, int(feather)))[0].unsqueeze(-1)
            region = final[t, y0:y0 + h, x0:x0 + w, :]
            if bool(colour_match):
                # Match suave por canal; clamp do ganho evita amplificar ruido.
                dims = (0, 1)
                pm, ps = piece.mean(dims, keepdim=True), piece.std(dims, keepdim=True).clamp_min(1e-4)
                rm, rs = region.mean(dims, keepdim=True), region.std(dims, keepdim=True).clamp_min(1e-4)
                gain = (rs / ps).clamp(0.5, 2.0)
                piece = ((piece - pm) * gain + rm).clamp(0, 1)
            a = a * float(blend)
            final[t, y0:y0 + h, x0:x0 + w, :] = region * (1.0 - a) + piece * a

        info = (f"stitch {T}f em {W}x{H} | compose={compose} feather={int(feather)}px | "
                f"colour_match={bool(colour_match)} blend={float(blend):.2f} | {_fmt_t(time.time() - t0)}")
        print(f"[Bruxos Tracked Stitch] {info}", flush=True)
        return (final.clamp(0, 1), info)


NODE_CLASS_MAPPINGS = {
    "BruxosTrackedCrop": BruxosTrackedCrop,
    "BruxosTrackedStitch": BruxosTrackedStitch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosTrackedCrop": "Tracked Crop / SAM3 (Bruxos)",
    "BruxosTrackedStitch": "Tracked Stitch (Bruxos)",
}
