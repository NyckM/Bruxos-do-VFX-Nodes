# -*- coding: utf-8 -*-
"""Merge de video em tiles sem dupla exposicao na faixa de overlap."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    from .bruxos_video_tiler import ler_config
except Exception:  # pragma: no cover - permite teste direto do arquivo
    from bruxos_video_tiler import ler_config


CAT = "Bruxos do VFX/Tiler"


def _unwrap(value, default):
    while isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return default if value is None else value


def _tile_tensor(value):
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise ValueError("[Bruxos Seam Merge] tile invalido; esperava tensor IMAGE.")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError("[Bruxos Seam Merge] cada tile precisa ser [F,H,W,C].")
    return value


def _gray(image):
    rgb = image[..., :3].float()
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _edge(gray):
    gx = F.pad((gray[..., 1:] - gray[..., :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    return gx + gy


def _shift_frame(frame, dx, dy):
    """Desloca IMAGE [H,W,C], preenchendo as bordas por replicacao."""
    h, w = int(frame.shape[0]), int(frame.shape[1])
    pad = max(1, abs(int(dx)), abs(int(dy)))
    x = frame.permute(2, 0, 1).unsqueeze(0)
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    y0, x0 = pad - int(dy), pad - int(dx)
    return x[:, :, y0:y0 + h, x0:x0 + w].squeeze(0).permute(1, 2, 0)


def _phase_shift(reference, moving, max_shift):
    """Retorna o deslocamento (dx,dy) a aplicar em moving para casar reference."""
    a, b = _gray(reference), _gray(moving)
    if min(int(a.shape[-2]), int(a.shape[-1])) < 8:
        return 0, 0, 0.0
    # Bordas carregam geometria e ignoram boa parte da diferenca de cor/material.
    a, b = _edge(a), _edge(b)
    if float(a.std()) < 1e-5 or float(b.std()) < 1e-5:
        return 0, 0, 0.0
    # FFT pequena e estavel: overlaps grandes sao reduzidos somente para estimar translacao.
    h, w = int(a.shape[-2]), int(a.shape[-1])
    scale = min(1.0, 256.0 / max(h, w))
    if scale < 1.0:
        nh, nw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))
        a = F.interpolate(a[None, None], size=(nh, nw), mode="area")[0, 0]
        b = F.interpolate(b[None, None], size=(nh, nw), mode="area")[0, 0]
    else:
        nh, nw = h, w
    a, b = a - a.mean(), b - b.mean()
    wy = torch.hann_window(nh, device=a.device, dtype=a.dtype).view(nh, 1)
    wx = torch.hann_window(nw, device=a.device, dtype=a.dtype).view(1, nw)
    window = wy * wx
    fa, fb = torch.fft.fft2(a * window), torch.fft.fft2(b * window)
    cross = fa * torch.conj(fb)
    cross = cross / cross.abs().clamp(min=1e-7)
    corr = torch.fft.ifft2(cross).real
    flat = int(torch.argmax(corr).item())
    py, px = divmod(flat, nw)
    if px > nw // 2:
        px -= nw
    if py > nh // 2:
        py -= nh
    dx = int(round(px / scale))
    dy = int(round(py / scale))
    limit = max(0, int(max_shift))
    dx, dy = max(-limit, min(limit, dx)), max(-limit, min(limit, dy))
    confidence = float(corr.max().item() / (corr.abs().mean().item() + 1e-7))
    return dx, dy, confidence


def _min_vertical_seam(cost):
    """Programacao dinamica; devolve x por linha em um overlap HxW."""
    c = cost.detach().float().cpu()
    h, w = int(c.shape[0]), int(c.shape[1])
    if w <= 1:
        return torch.zeros(h, dtype=torch.long)
    center = (w - 1) * 0.5
    prior = ((torch.arange(w).float() - center) / max(1.0, center)) ** 2
    c = c + prior.view(1, w) * max(1e-5, float(c.mean())) * 0.04
    dp = c[0].clone()
    parent = torch.zeros((h, w), dtype=torch.int8)
    inf = torch.tensor(float("inf"))
    for y in range(1, h):
        padded = F.pad(dp, (1, 1), value=float(inf))
        candidates = torch.stack((padded[:-2], padded[1:-1], padded[2:]), 0)
        best, which = candidates.min(0)
        dp = c[y] + best
        parent[y] = which.to(torch.int8) - 1
    seam = torch.empty(h, dtype=torch.long)
    x = int(torch.argmin(dp).item())
    seam[-1] = x
    for y in range(h - 1, 0, -1):
        x += int(parent[y, x].item())
        x = max(0, min(w - 1, x))
        seam[y - 1] = x
    return seam


def _pair_alpha(cost, current_is_positive, vertical, seam_width):
    if not vertical:
        return _pair_alpha(cost.t(), current_is_positive, True, seam_width).t()
    h, w = int(cost.shape[0]), int(cost.shape[1])
    seam = _min_vertical_seam(cost).to(cost.device).view(h, 1).float()
    xx = torch.arange(w, device=cost.device).view(1, w).float()
    signed = xx - seam if current_is_positive else seam - xx
    width = max(1.0, float(seam_width))
    alpha = (signed / width + 0.5).clamp(0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _cost_map(existing, incoming):
    ga, gb = _gray(existing), _gray(incoming)
    luma = (ga - gb).abs().mean(0)
    edges = (_edge(ga) - _edge(gb)).abs().mean(0)
    cost = luma + 0.65 * edges
    return F.avg_pool2d(cost[None, None], 3, stride=1, padding=1)[0, 0]


def _interpolate_shifts(frame_count, sample_frames, sample_values, temporal_lock):
    if not sample_values:
        return [(0, 0)] * frame_count
    raw = []
    for frame in range(frame_count):
        right = 0
        while right < len(sample_frames) and sample_frames[right] < frame:
            right += 1
        if right <= 0:
            value = sample_values[0]
        elif right >= len(sample_frames):
            value = sample_values[-1]
        else:
            f0, f1 = sample_frames[right - 1], sample_frames[right]
            t = (frame - f0) / max(1, f1 - f0)
            a, b = sample_values[right - 1], sample_values[right]
            value = (a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t)
        raw.append(value)
    lock = max(0.0, min(0.999, float(temporal_lock)))
    smooth, state = [], raw[0]
    for value in raw:
        state = (lock * state[0] + (1 - lock) * value[0],
                 lock * state[1] + (1 - lock) * value[1])
        smooth.append((int(round(state[0])), int(round(state[1]))))
    return smooth


class BruxosSeamAwareVideoTileMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_config": ("TILE_CONFIG", {"forceInput": True}),
                "tiles": ("IMAGE", {"tooltip": "Lista de tiles gerados, na ordem do slicer."}),
                "alignment": (["translation", "off"], {"default": "translation"}),
                "max_shift": ("INT", {"default": 24, "min": 0, "max": 128, "step": 1}),
                "seam_width": ("INT", {"default": 16, "min": 0, "max": 128, "step": 1}),
                "temporal_lock": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 0.999, "step": 0.01}),
                "alignment_samples": ("INT", {"default": 5, "min": 1, "max": 33, "step": 1}),
                "color_match_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05}),
                "merge_device": (["cpu", "auto", "cuda"], {"default": "cpu"}),
            },
        }

    INPUT_IS_LIST = (False, True)
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "merge"
    CATEGORY = CAT
    DESCRIPTION = (
        "Alinha translacao no overlap, encontra uma linha de menor erro e mistura apenas uma faixa estreita. "
        "Evita a dupla exposicao do merge por media ponderada."
    )

    def merge(self, tile_config, tiles, alignment="translation", max_shift=24, seam_width=16,
              temporal_lock=0.90, alignment_samples=5, color_match_strength=0.15,
              merge_device="cpu"):
        cfg = tile_config[0] if isinstance(tile_config, list) and tile_config else tile_config
        values = list(tiles) if isinstance(tiles, (list, tuple)) else [tiles]
        tl = [_tile_tensor(v) for v in values]
        W, H, _lw, _lh, _ox, _oy, _m, specs = ler_config(cfg)
        if len(tl) != len(specs):
            raise ValueError(f"[Bruxos Seam Merge] chegaram {len(tl)} tiles; esperados {len(specs)}.")
        batch = int(tl[0].shape[0])
        channels = int(tl[0].shape[-1])
        if any(int(t.shape[0]) != batch for t in tl):
            raise ValueError("[Bruxos Seam Merge] todos os tiles precisam ter a mesma quantidade de frames.")
        mode = str(_unwrap(alignment, "translation"))
        max_shift = int(_unwrap(max_shift, 24))
        seam_width = int(_unwrap(seam_width, 16))
        temporal_lock = float(_unwrap(temporal_lock, 0.90))
        sample_count = max(1, int(_unwrap(alignment_samples, 5)))
        color_strength = float(_unwrap(color_match_strength, 0.15))
        device_name = str(_unwrap(merge_device, "cpu"))
        if device_name == "cuda" and torch.cuda.is_available():
            dev = torch.device("cuda:0")
        elif device_name == "auto":
            dev = tl[0].device
        else:
            dev = torch.device("cpu")

        dtype = tl[0].dtype
        canvas = torch.zeros((batch, H, W, channels), dtype=dtype, device=dev)
        covered = torch.zeros((H, W), dtype=torch.bool, device=dev)
        ordered = sorted(enumerate(specs), key=lambda item: item[1]["ordem"])
        placed = []
        report = []
        sample_frames = sorted(set(int(round(v)) for v in torch.linspace(0, batch - 1, min(batch, sample_count)).tolist()))

        for order_index, (tile_index, spec) in enumerate(ordered):
            tile = tl[tile_index]
            x, y, w, h = (int(spec[k]) for k in ("x", "y", "w", "h"))
            if tuple(tile.shape[1:3]) != (h, w):
                tile = F.interpolate(tile.permute(0, 3, 1, 2).float(), size=(h, w),
                                     mode="bicubic", align_corners=False).permute(0, 2, 3, 1).to(dtype)
            if order_index == 0:
                canvas[:, y:y + h, x:x + w] = tile.to(dev)
                covered[y:y + h, x:x + w] = True
                placed.append(spec)
                report.append(f"tile {tile_index + 1}: base")
                continue

            pairs = []
            for previous in placed:
                ix0, iy0 = max(x, previous["x"]), max(y, previous["y"])
                ix1 = min(x + w, previous["x"] + previous["w"])
                iy1 = min(y + h, previous["y"] + previous["h"])
                if ix1 > ix0 and iy1 > iy0:
                    pairs.append((previous, ix0, iy0, ix1, iy1))

            sampled_shifts = []
            if mode == "translation" and pairs and max_shift > 0:
                for frame in sample_frames:
                    estimates = []
                    for _previous, ix0, iy0, ix1, iy1 in pairs:
                        ref = canvas[frame, iy0:iy1, ix0:ix1]
                        mov = tile[frame, iy0 - y:iy1 - y, ix0 - x:ix1 - x].to(dev)
                        dx, dy, confidence = _phase_shift(ref, mov, max_shift)
                        if confidence >= 2.0:
                            estimates.append((dx, dy))
                    if estimates:
                        sampled_shifts.append((
                            int(torch.tensor([v[0] for v in estimates]).median().item()),
                            int(torch.tensor([v[1] for v in estimates]).median().item()),
                        ))
                    else:
                        sampled_shifts.append((0, 0))
            else:
                sampled_shifts = [(0, 0)] * len(sample_frames)
            shifts = _interpolate_shifts(batch, sample_frames, sampled_shifts, temporal_lock)

            # Custo temporal medio e mascara de ownership fixa no tempo.
            alpha = torch.ones((h, w), dtype=torch.float32, device=dev)
            for previous, ix0, iy0, ix1, iy1 in pairs:
                existing_samples, incoming_samples = [], []
                for frame in sample_frames:
                    warped = _shift_frame(tile[frame].to(dev), *shifts[frame])
                    existing_samples.append(canvas[frame, iy0:iy1, ix0:ix1])
                    incoming_samples.append(warped[iy0 - y:iy1 - y, ix0 - x:ix1 - x])
                existing_stack = torch.stack(existing_samples)
                incoming_stack = torch.stack(incoming_samples)
                cost = _cost_map(existing_stack, incoming_stack)
                dcx = (x + w * 0.5) - (previous["x"] + previous["w"] * 0.5)
                dcy = (y + h * 0.5) - (previous["y"] + previous["h"] * 0.5)
                vertical = abs(dcx) >= abs(dcy)
                current_positive = dcx >= 0 if vertical else dcy >= 0
                pair_alpha = _pair_alpha(cost, current_positive, vertical, seam_width)
                ry0, ry1, rx0, rx1 = iy0 - y, iy1 - y, ix0 - x, ix1 - x
                alpha[ry0:ry1, rx0:rx1] = torch.minimum(alpha[ry0:ry1, rx0:rx1], pair_alpha)

            # Monta em blocos para nao criar outra copia completa do video do tile.
            for start in range(0, batch, 4):
                end = min(batch, start + 4)
                warped_frames = torch.stack([
                    _shift_frame(tile[frame].to(dev), *shifts[frame]) for frame in range(start, end)
                ])
                region = canvas[start:end, y:y + h, x:x + w]
                cov = covered[y:y + h, x:x + w]
                if color_strength > 0 and bool(cov.any()):
                    mask = cov.view(1, h, w, 1).expand(end - start, h, w, 1)
                    count = mask.sum(dim=(1, 2), keepdim=True).clamp(min=1)
                    mean_ref = (region.float() * mask).sum(dim=(1, 2), keepdim=True) / count
                    mean_new = (warped_frames.float() * mask).sum(dim=(1, 2), keepdim=True) / count
                    warped_frames = (warped_frames.float() + color_strength * (mean_ref - mean_new)).clamp(0, 1).to(dtype)
                a = torch.where(cov, alpha, torch.ones_like(alpha)).view(1, h, w, 1)
                canvas[start:end, y:y + h, x:x + w] = (
                    region.float() * (1.0 - a) + warped_frames.float() * a
                ).to(dtype)

            covered[y:y + h, x:x + w] = True
            placed.append(spec)
            shift_values = shifts if shifts else [(0, 0)]
            dxs, dys = [v[0] for v in shift_values], [v[1] for v in shift_values]
            report.append(
                f"tile {tile_index + 1}: shift x {min(dxs)}..{max(dxs)}, "
                f"y {min(dys)}..{max(dys)}, {len(pairs)} overlap(s)"
            )

        if not bool(covered.all()):
            missing = int((~covered).sum().item())
            raise ValueError(f"[Bruxos Seam Merge] layout deixou {missing} pixels sem cobertura.")
        info = " | ".join(report)
        print(f"[Bruxos Seam Merge] {batch} frames {W}x{H} | {info}", flush=True)
        return canvas.cpu(), info


NODE_CLASS_MAPPINGS = {
    "BruxosSeamAwareVideoTileMerge": BruxosSeamAwareVideoTileMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosSeamAwareVideoTileMerge": "Seam-Aware Video Tile Merge (Bruxos)",
}

