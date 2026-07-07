# -*- coding: utf-8 -*-
"""
Point Tracker Visualizer (Bruxos)
=================================
Destino VISUAL das saidas do Point Tracker: recebe os frames + tracks (+ visibility
opcional) e devolve os frames com os pontos/rastros desenhados. Ligue a saida IMAGE
num Preview pra VER o tracking.

Tolerante ao formato do tracks (o CoTracker e cia variam):
  aceita shape [B,T,N,2] ou [T,N,2]; coords em PIXEL ou NORMALIZADAS (auto-detecta).
visibility opcional: [B,T,N] ou [T,N] (0/1 ou bool) -> ponto oculto fica esmaecido.
"""

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import cv2
except Exception:
    cv2 = None


def _np(x):
    if x is None:
        return None
    if torch is not None and hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _extract_tracks_vis(tr):
    """Aceita POINT_TRACKS (tensor) OU OBJECT_TRACKS (dict com 'tracks'/'visibility').
    Retorna (tracks_array, visibility_ou_None)."""
    vis = None
    if isinstance(tr, dict):
        vis = tr.get("visibility", None)
        tr = tr.get("tracks", tr.get("object_tracks", None))
        if tr is None:
            raise ValueError("dict de tracks sem chave 'tracks'.")
    return tr, vis


def _norm_tracks(tr):
    """-> array float [T, N, 2]."""
    tr = _np(tr).astype(np.float32)
    if tr.ndim == 4:      # [B,T,N,2] -> pega batch 0
        tr = tr[0]
    if tr.ndim == 2:      # [N,2] (1 frame) -> [1,N,2]
        tr = tr[None]
    if tr.ndim != 3 or tr.shape[-1] < 2:
        raise ValueError(f"tracks com shape inesperado: {tr.shape}")
    return tr[..., :2]


def _norm_vis(vis, T, N):
    if vis is None:
        return None
    v = _np(vis).astype(np.float32)
    if v.ndim == 3:
        v = v[0]
    if v.ndim == 1:
        v = v[None]
    # tenta alinhar em [T, N]
    if v.shape == (T, N):
        return v
    if v.shape == (N, T):
        return v.T
    return None  # formato nao reconhecido -> ignora visibilidade


def _color(i, n):
    # cor por ponto (HSV -> BGR), espalhada
    import colorsys
    h = (i / max(1, n)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR p/ cv2


class BruxosPointVisualizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Os MESMOS frames que entraram no tracker."}),
                "tracks": ("POINT_TRACKS,OBJECT_TRACKS", {"tooltip": "Saida 'tracks' do Point Tracker OU 'object_tracks' do Object Tracker (Bruxos)."}),
            },
            "optional": {
                "visibility": ("VISIBILITY", {"tooltip": "Saida 'visibility' (opcional): pontos ocultos ficam esmaecidos."}),
                "raio": ("INT", {"default": 4, "min": 1, "max": 30, "tooltip": "Tamanho da bolinha de cada ponto."}),
                "rastro": ("INT", {"default": 8, "min": 0, "max": 60, "tooltip": "Quantos frames de rastro desenhar atras de cada ponto (0 = sem rastro)."}),
                "so_visiveis": ("BOOLEAN", {"default": False, "tooltip": "Se ligado, nao desenha pontos marcados como ocultos (precisa de visibility)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    OUTPUT_TOOLTIPS = ("Frames com os pontos/rastros desenhados. Ligue num Preview Image.",
                       "Relatorio: nº de frames, nº de pontos, se coords eram pixel ou normalizadas.")
    FUNCTION = "draw"
    CATEGORY = "Bruxos do VFX/Tracking"
    DESCRIPTION = ("Desenha os pontos rastreados (tracks) sobre o video, pra VER o tracking. "
                   "Recebe as saidas do Point Tracker (Bruxos) e devolve IMAGE pro Preview.")

    def draw(self, images, tracks, visibility=None, raio=4, rastro=8, so_visiveis=False):
        if cv2 is None:
            raise RuntimeError("opencv (cv2) indisponivel; instale opencv-python.")
        imgs = _np(images)  # [T,H,W,C] float 0..1
        T, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[2]
        tracks, vis_from_dict = _extract_tracks_vis(tracks)   # abre OBJECT_TRACKS se for dict
        if visibility is None:
            visibility = vis_from_dict
        tr = _norm_tracks(tracks)          # [Tt, N, 2]
        Tt, N = tr.shape[0], tr.shape[1]

        # coords normalizadas? (todos valores <= ~1.5) -> multiplica por W/H
        mx = float(np.nanmax(np.abs(tr))) if tr.size else 0.0
        normalizado = mx <= 1.5
        if normalizado:
            tr = tr.copy()
            tr[..., 0] *= W
            tr[..., 1] *= H

        vis = _norm_vis(visibility, Tt, N)

        out = (imgs[..., :3] * 255.0).clip(0, 255).astype(np.uint8).copy()
        nframes = min(T, Tt)
        for t in range(nframes):
            frame = out[t]
            for i in range(N):
                x, y = tr[t, i, 0], tr[t, i, 1]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                visivel = True if vis is None else bool(vis[t, i] > 0.5)
                if so_visiveis and not visivel:
                    continue
                col = _color(i, N)
                # rastro (frames anteriores)
                if rastro > 0:
                    for k in range(1, rastro + 1):
                        tp = t - k
                        if tp < 0:
                            break
                        xp, yp = tr[tp, i, 0], tr[tp, i, 1]
                        if not (np.isfinite(xp) and np.isfinite(yp)):
                            continue
                        cv2.line(frame, (int(xp), int(yp)), (int(x), int(y)), col, 1, cv2.LINE_AA)
                        x, y = xp, yp
                    x, y = tr[t, i, 0], tr[t, i, 1]
                # ponto (esmaecido se oculto)
                if visivel:
                    cv2.circle(frame, (int(x), int(y)), raio, col, -1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (int(x), int(y)), raio, col, 1, cv2.LINE_AA)

        out_f = out.astype(np.float32) / 255.0
        result = torch.from_numpy(out_f) if torch is not None else out_f
        info = (f"frames={nframes} pontos={N} coords={'normalizadas' if normalizado else 'pixel'} "
                f"visibility={'sim' if vis is not None else 'nao'}")
        print(f"[Bruxos Point Visualizer] {info}", flush=True)
        return (result, info)


NODE_CLASS_MAPPINGS = {"BruxosPointVisualizer": BruxosPointVisualizer}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosPointVisualizer": "Point Visualizer (Bruxos)"}
