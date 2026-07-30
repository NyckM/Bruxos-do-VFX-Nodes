# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — MoCha TILED (crop & stitch, espacial + temporal)
================================================================
Otimizacao de ladrilho pro MoCha (Wan / WanVideoWrapper do Kijai).

POR QUE crop & stitch, e nao um guider de grade (como o wan_tiled/ltx):
  O node MoCha (BruxosMochaEmbeds) so faz o ENCODE -- ele monta os
  WANVIDIMAGE_EMBEDS (video + mascara + refs concatenados) que voce liga no
  WanVideo Sampler do WRAPPER do Kijai. O sampler nao e o SamplerCustom do
  Comfy, entao nao da pra pendurar um CFGGuider de ladrilho nele com seguranca.
  Alem disso, o MoCha concatena as REFS de identidade no latente -- cortar
  isso numa grade quebraria o rosto entre ladrilhos e degradaria a identidade.

  A forma certa (e a mesma que o ComfyUI-Inpaint-CropAndStitch usa) e o par
  CROP -> [seu MochaEmbeds + Sampler + decode roda no recorte] -> STITCH:

     input_video ─┐
     mask ────────┤
     ref1/ref2 ───┤
                  ▼
        [Mocha BBox Crop (Bruxos)]  --> cropped_video, cropped_mask, ref1, ref2, CROP_DATA
                  │  (o recorte tem SO a regiao da mascara + contexto,
                  │   em resolucao cheia; as refs passam INTEIRAS)
                  ▼
        [Mocha Embeds (Bruxos)] -> [WanVideo Sampler] -> [WanVideo Decode]
                  │
                  ▼
        [Mocha BBox Stitch (Bruxos)]  <-- edited_crop, CROP_DATA
                  ▼
              video final (cola o recorte editado de volta no frame, com feather)

GANHOS:
  * ESPACIAL (bbox): o MoCha roda so na regiao do sujeito (rosto/personagem),
    em resolucao cheia. Se o sujeito e pequeno/medio no frame, isso e MUITO
    mais rapido e usa menos VRAM -- o resto do frame fica intacto, sem seam.
  * TEMPORAL: o Crop tambem aceita uma FAIXA DE FRAMES (frame_start/frame_count).
    Pra video longo, recorte um bloco no tempo, rode o MoCha nele e costure de
    volta (com overlap/feather temporal no Stitch). Rode um bloco por vez.

  O modo GRADE espacial NxM nao entra aqui de proposito: pro MoCha ele quebra a
  identidade (refs cortadas) e o sujeito costuma caber num bbox unico, que e
  mais rapido e sem emenda.

Categoria: Bruxos do VFX/Mocha
"""

import time
import logging

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

# Reaproveita helpers do pacote (mesma conta do Bernini/Mocha).
try:
    from .nodes import (
        _mask_bbox as _bx_mask_bbox,
        _rect_feather_mask as _bx_rect_feather,
        _normalize_mask as _bx_normalize_mask,
        _grow_blur_mask as _bx_grow_blur,
        _align_up_4n1 as _bx_align_up_4n1,
    )
    _HAS_HELPERS = True
except Exception as e:  # pragma: no cover
    logging.info(f"[Bruxos Mocha Tiled] helpers do nodes.py indisponiveis ({e}); usando fallback")
    _HAS_HELPERS = False

    def _bx_align_up_4n1(n):
        n = int(n)
        if n < 1:
            return 1
        r = (n - 1) % 4
        return n if r == 0 else n + (4 - r)

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
            k = grow * 2 + 1
            x = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=1, padding=grow)
        elif grow < 0:
            g = -grow
            k = g * 2 + 1
            x = -torch.nn.functional.max_pool2d(-x, kernel_size=k, stride=1, padding=g)
        blur = int(blur)
        if blur > 0:
            k = blur * 2 + 1
            coords = torch.arange(k, dtype=torch.float32, device=m.device) - blur
            sigma = blur * 0.5 + 1e-6
            g1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
            g1d = (g1d / g1d.sum())
            x = torch.nn.functional.conv2d(x, g1d.view(1, 1, k, 1), padding=(blur, 0))
            x = torch.nn.functional.conv2d(x, g1d.view(1, 1, 1, k), padding=(0, blur))
        return x.squeeze(1).clamp(0.0, 1.0)

    def _bx_mask_bbox(m, pad, stride, W, H, thr=0.02):
        any2d = (m.amax(dim=0) > thr)
        rows = torch.where(any2d.any(dim=1))[0]
        cols = torch.where(any2d.any(dim=0))[0]
        if rows.numel() == 0 or cols.numel() == 0:
            return 0, 0, int(W), int(H)
        y0 = int(rows.min()); y1 = int(rows.max()) + 1
        x0 = int(cols.min()); x1 = int(cols.max()) + 1
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(int(W), x1 + pad); y1 = min(int(H), y1 + pad)
        x0 -= x0 % stride
        y0 -= y0 % stride
        if x1 % stride:
            x1 = min(int(W), x1 + (stride - x1 % stride))
        if y1 % stride:
            y1 = min(int(H), y1 + (stride - y1 % stride))
        if x1 - x0 < stride:
            x1 = min(int(W), x0 + stride)
        if y1 - y0 < stride:
            y1 = min(int(H), y0 + stride)
        return x0, y0, x1, y1

    def _bx_rect_feather(n, ch, cw, feather, device=None):
        device = device or torch.device("cpu")
        ys = torch.arange(ch, dtype=torch.float32, device=device).view(ch, 1)
        xs = torch.arange(cw, dtype=torch.float32, device=device).view(1, cw)
        f = max(1, int(feather))
        dist_y = torch.minimum(ys, (ch - 1) - ys)
        dist_x = torch.minimum(xs, (cw - 1) - xs)
        edge = torch.minimum(dist_y, dist_x)
        alpha = (edge / float(f)).clamp(0.0, 1.0)
        return alpha.unsqueeze(0).expand(n, ch, cw).contiguous()


CAT = "Bruxos do VFX/Mocha"


def _fmt_t(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{s - 60 * int(s // 60):04.1f}s"


def _snap_span_aligned(lo, hi, limit, stride):
    """Garante que (hi-lo) seja multiplo de `stride` E caiba em [0, limit].

    O _mask_bbox clampa em W/H, mas se o PROPRIO frame nao for multiplo de
    stride (ex.: altura 486) e o bbox encostar na borda, o span sai
    desalinhado -> o Wan VAE rejeita. Aqui:
      1) arredonda o span PRA CIMA (multiplo de stride);
      2) se estourar `limit`, empurra o inicio pra tras;
      3) se ainda estourar (frame menor que um bloco alinhado), arredonda o
         span PRA BAIXO ate o maior multiplo que cabe -- perde no maximo
         `stride-1` px de CONTEXTO (nunca o nucleo da mascara, que fica no
         miolo gracas ao context_pad).
    """
    lo = int(lo); hi = int(hi); limit = int(limit); stride = max(1, int(stride))
    span = hi - lo
    if span <= 0:
        span = stride
    span_al = ((span + stride - 1) // stride) * stride
    hi_new = lo + span_al
    if hi_new > limit:
        hi_new = limit
        lo = max(0, hi_new - span_al)
    if hi_new - lo != span_al:
        # nao coube: arredonda pra baixo ate caber dentro de [0, limit]
        avail = hi_new - lo
        span_dn = max(stride, (avail // stride) * stride)
        if span_dn > limit:                     # frame menor que 1 bloco
            span_dn = (limit // stride) * stride or limit
        lo = max(0, hi_new - span_dn)
        hi_new = lo + span_dn
        if hi_new > limit:
            hi_new = limit
            lo = max(0, hi_new - span_dn)
    return int(lo), int(hi_new)


def _norm_mask_full(mask, T, H, W):
    """Normaliza mascara (MASK ou IMAGE colorida) -> [T,H,W] no tamanho do video."""
    if mask is None:
        return None
    m = _bx_normalize_mask(mask)                 # [Tm,Hm,Wm]
    if int(m.shape[1]) != H or int(m.shape[2]) != W:
        m = torch.nn.functional.interpolate(
            m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1).clamp(0, 1)
    if int(m.shape[0]) < T:                       # repete o ultimo frame
        m = torch.cat([m, m[-1:].repeat(T - int(m.shape[0]), 1, 1)], dim=0)
    return m[:T].clamp(0, 1)


# =============================================================================
# Mocha BBox Crop (Bruxos) — recorta a regiao da mascara (+contexto) e, opcional,
# uma faixa de frames. As refs passam INTEIRAS.
# =============================================================================
class BruxosMochaBBoxCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_video": ("IMAGE", {"tooltip": "Video de entrada completo (o footage a editar)."}),
                "mask": ("MASK,IMAGE", {"tooltip": "Mascara da regiao a editar. Aceita MASK ou IMAGE colorida (SAM3/SCAIL/FaceFusion)."}),
                "ref1": ("IMAGE", {"tooltip": "Referencia 1 (identidade). PASSA INTEIRA pro Mocha Embeds -- nao e recortada."}),
            },
            "optional": {
                "ref2": ("IMAGE", {"tooltip": "Referencia 2 (opcional). Tambem passa inteira."}),
                "mask_frame_mode": (["uniao (todos os frames)", "primeiro frame"], {"default": "uniao (todos os frames)",
                    "tooltip": "Como reduzir uma mascara por-frame ao bbox unico: 'uniao' cobre o sujeito onde quer que ele passe (recomendado p/ sujeito em movimento); 'primeiro frame' usa so o frame 0."}),
                "mask_grow": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1, "tooltip": "Dilata (+) / contrai (-) a mascara ANTES de medir o bbox, em pixels."}),
                "context_pad": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 16, "tooltip": "Folga ao redor do bbox da mascara, em pixels. O MoCha ve esse fundo pra casar cor/textura na borda. 64-128 e um bom comeco."}),
                "align": ("INT", {"default": 16, "min": 8, "max": 64, "step": 8, "tooltip": "Alinha o recorte a um multiplo disto (o Wan VAE precisa de largura/altura multiplas de 16). Deixe 16."}),
                "frame_start": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "[temporal] Primeiro frame do BLOCO a recortar. Use com frame_count pra processar video longo em blocos."}),
                "frame_count": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "[temporal] Quantos frames recortar a partir de frame_start. 0 = do frame_start ate o fim (video inteiro no tempo)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "MOCHA_CROP", "STRING")
    RETURN_NAMES = ("cropped_video", "cropped_mask", "ref1", "ref2", "crop_data", "info")
    OUTPUT_TOOLTIPS = (
        "Recorte do video (so a regiao da mascara + contexto, e a faixa de frames escolhida). Ligue no 'input_video' do Mocha Embeds (Bruxos).",
        "Recorte da mascara, alinhado ao video recortado. Ligue no 'mask' do Mocha Embeds.",
        "ref1 INTEIRA (passthrough). Ligue no ref1 do Mocha Embeds.",
        "ref2 INTEIRA (passthrough, pode estar vazia). Ligue no ref2 do Mocha Embeds.",
        "Geometria do recorte + video/mascara originais. Ligue no Mocha BBox Stitch pra colar de volta.",
        "Relatorio do recorte.",
    )
    FUNCTION = "crop"
    CATEGORY = CAT
    DESCRIPTION = (
        "Mocha BBox Crop (Bruxos): recorta a regiao da mascara (com contexto) e, opcional, uma faixa de "
        "frames, pra rodar o MoCha SO no sujeito em resolucao cheia (mais rapido, menos VRAM, sem seam). "
        "As refs de identidade passam INTEIRAS. Ligue a saida no Mocha Embeds -> Sampler -> Decode -> "
        "Mocha BBox Stitch. Se a mascara for vazia, cai pro frame inteiro."
    )

    def crop(self, input_video, mask, ref1, ref2=None,
             mask_frame_mode="uniao (todos os frames)", mask_grow=0,
             context_pad=64, align=16, frame_start=0, frame_count=0):
        if torch is None:
            raise RuntimeError("[Bruxos Mocha Tiled] torch indisponivel.")
        t0 = time.time()

        T_full = int(input_video.shape[0])
        H, W = int(input_video.shape[1]), int(input_video.shape[2])

        # ---- FAIXA TEMPORAL (opcional) ----
        fs = max(0, min(int(frame_start), T_full - 1))
        fc = int(frame_count)
        fe = T_full if fc <= 0 else min(T_full, fs + fc)
        if fe <= fs:
            fe = min(T_full, fs + 1)
        video_t = input_video[fs:fe]
        Tt = int(video_t.shape[0])

        # ---- MASCARA no tamanho do video, fatiada no tempo igual ----
        m_full = _norm_mask_full(mask, T_full, H, W)
        if m_full is None:
            m_t = torch.ones((Tt, H, W), dtype=torch.float32)
        else:
            m_t = m_full[fs:fe]

        # reduz a UM plano p/ medir o bbox (uniao ou primeiro frame)
        if str(mask_frame_mode).startswith("primeiro"):
            m_plane = m_t[:1]
        else:
            m_plane = m_t.amax(dim=0, keepdim=True)
        if int(mask_grow) != 0:
            m_plane = _bx_grow_blur(m_plane, int(mask_grow), 0)

        # ---- BBOX (alinhado ao multiplo de `align`) + contexto ----
        stride = max(8, int(align))
        empty_mask = (m_full is None) or (float(m_plane.max()) < 0.02)
        if empty_mask:
            x0, y0, x1, y1 = 0, 0, W, H
            bbox_msg = "mascara vazia -> frame inteiro"
        else:
            x0, y0, x1, y1 = _bx_mask_bbox(m_plane, int(context_pad), stride, W, H)
            # garante recorte alinhado mesmo se W/H do frame nao forem mult. de
            # stride e o bbox encostar na borda (senao o Wan VAE rejeita).
            x0, x1 = _snap_span_aligned(x0, x1, W, stride)
            y0, y1 = _snap_span_aligned(y0, y1, H, stride)
            bbox_msg = f"bbox ({x0},{y0})-({x1},{y1})"

        cropped_video = video_t[:, y0:y1, x0:x1, :].contiguous()
        cropped_mask = m_t[:, y0:y1, x0:x1].contiguous()
        cw, ch = x1 - x0, y1 - y0
        area_pct = 100.0 * (cw * ch) / (W * H)

        # dados p/ o stitch: guardamos o ORIGINAL (video completo no tempo do
        # bloco) e a geometria. O stitch cola so a regiao do bbox de volta.
        crop_data = {
            "orig_video": video_t.cpu(),          # [Tt,H,W,3] (bloco temporal, res cheia)
            "orig_mask_crop": cropped_mask.cpu(),  # [Tt,ch,cw] mascara na regiao (p/ silhouette)
            "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
            "H": int(H), "W": int(W),
            "frame_start": int(fs), "frame_end": int(fe), "T_full": int(T_full),
            "full_video": None,   # preenchido so se quiser reconstruir o video inteiro no tempo
        }
        # guarda o video completo (todos os frames) so por referencia do stitch
        # temporal -- barato em CPU e evita o usuario ter que religar o original.
        crop_data["full_video"] = input_video.cpu()

        info = (f"crop {cw}x{ch} de {W}x{H} (~{area_pct:.0f}% da area) | {bbox_msg} | "
                f"frames {fs}..{fe - 1} de {T_full} | ctx={int(context_pad)}px align={stride} | "
                f"{_fmt_t(time.time() - t0)}")
        print(f"[Bruxos Mocha Crop] {info}", flush=True)

        # ref2 passa como veio (None se nao ligado) -- devolver tensor vazio
        # quebraria o MochaEmbeds, que faz `if ref2 is not None: encode(ref2)`.
        return (cropped_video, cropped_mask, ref1, ref2, crop_data, info)


# =============================================================================
# Mocha BBox Stitch (Bruxos) — cola o recorte editado de volta no frame inteiro.
# =============================================================================
class BruxosMochaBBoxStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_data": ("MOCHA_CROP", {"tooltip": "Saida 'crop_data' do Mocha BBox Crop (Bruxos)."}),
                "edited_crop": ("IMAGE", {"tooltip": "O recorte JA editado (saida do WanVideo Decode depois do MoCha rodar no recorte)."}),
            },
            "optional": {
                "compose": (["rectangle", "silhouette"], {"default": "rectangle", "tooltip": "Como colar de volta. rectangle = retangulo do bbox com feather nas bordas (sem linha de contorno, recomendado). silhouette = usa a mascara do sujeito como alpha."}),
                "feather": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1, "tooltip": "Suavidade da borda ao colar (em pixels). Maior = transicao mais macia com o fundo original."}),
                "temporal_feather": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1, "tooltip": "[temporal] Se voce recortou um BLOCO de frames, suaviza a emenda temporal nas pontas do bloco (frames de fade no comeco/fim). 0 = corte seco."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    OUTPUT_TOOLTIPS = (
        "Video final: o recorte editado colado de volta no frame inteiro (e no tempo inteiro, se foi um bloco).",
        "Relatorio do stitch.",
    )
    FUNCTION = "stitch"
    CATEGORY = CAT
    DESCRIPTION = (
        "Mocha BBox Stitch (Bruxos): cola o recorte editado de volta no frame original com feather "
        "(sem emenda dura). Se o crop foi um bloco temporal, recompoe o video inteiro no tempo, com "
        "fade opcional nas pontas do bloco. Corrige diferenca de contagem de frames do 4n+1 automaticamente."
    )

    def stitch(self, crop_data, edited_crop, compose="rectangle", feather=24, temporal_feather=0):
        if torch is None:
            raise RuntimeError("[Bruxos Mocha Tiled] torch indisponivel.")
        t0 = time.time()

        x0, y0, x1, y1 = crop_data["x0"], crop_data["y0"], crop_data["x1"], crop_data["y1"]
        H, W = crop_data["H"], crop_data["W"]
        cw, ch = x1 - x0, y1 - y0
        fs, fe, T_full = crop_data["frame_start"], crop_data["frame_end"], crop_data["T_full"]

        orig_block = crop_data["orig_video"].float().clamp(0, 1)   # [Tt,H,W,3]
        Tt = int(orig_block.shape[0])

        edited = edited_crop.float().clamp(0, 1)                    # [Te,ech,ecw,3]
        Te = int(edited.shape[0])

        # ---- 4n+1: o decode pode devolver mais/menos frames que o bloco ----
        if Te > Tt:
            edited = edited[:Tt]
        elif Te < Tt:
            edited = torch.cat([edited, edited[-1:].repeat(Tt - Te, 1, 1, 1)], dim=0)

        # ---- resolucao: o recorte editado deve ter a res do bbox; se diferir,
        # reamostra pra caber exatamente (evita erro de indice) ----
        if int(edited.shape[1]) != ch or int(edited.shape[2]) != cw:
            e = edited.permute(0, 3, 1, 2)
            e = torch.nn.functional.interpolate(e, size=(ch, cw), mode="bilinear", align_corners=False)
            edited = e.permute(0, 2, 3, 1).clamp(0, 1)

        # ---- alpha de composicao (rectangle feather ou silhouette) ----
        if compose == "silhouette" and "orig_mask_crop" in crop_data and crop_data["orig_mask_crop"] is not None:
            alpha = crop_data["orig_mask_crop"].float().clamp(0, 1)   # [Tt,ch,cw]
            if int(alpha.shape[0]) != Tt:
                # casa o tempo
                if int(alpha.shape[0]) > Tt:
                    alpha = alpha[:Tt]
                else:
                    alpha = torch.cat([alpha, alpha[-1:].repeat(Tt - int(alpha.shape[0]), 1, 1)], dim=0)
            if int(feather) > 0:
                alpha = _bx_grow_blur(alpha, 0, int(feather))
            alpha = alpha.unsqueeze(-1)                               # [Tt,ch,cw,1]
        else:
            alpha = _bx_rect_feather(Tt, ch, cw, max(1, int(feather))).unsqueeze(-1)

        # ---- cola no bloco ----
        out_block = orig_block.clone()
        region = out_block[:, y0:y1, x0:x1, :]
        out_block[:, y0:y1, x0:x1, :] = region * (1.0 - alpha) + edited * alpha

        # ---- recompoe no TEMPO (se foi um bloco de um video maior) ----
        full = crop_data.get("full_video", None)
        if full is not None and (fs > 0 or fe < T_full):
            final = full.float().clamp(0, 1).clone()                 # [T_full,H,W,3]
            tf = int(temporal_feather)
            if tf > 0 and Tt > 2 * tf:
                # fade nas pontas do bloco: mistura com o original nas bordas
                wt = torch.ones(Tt, dtype=torch.float32)
                ramp = torch.linspace(0.0, 1.0, tf + 2)[1:-1]
                if fs > 0:
                    wt[:tf] = ramp
                if fe < T_full:
                    wt[-tf:] = torch.flip(ramp, dims=[0])
                wt = wt.view(Tt, 1, 1, 1)
                final[fs:fe] = final[fs:fe] * (1.0 - wt) + out_block * wt
            else:
                final[fs:fe] = out_block
            temporal_msg = f"bloco {fs}..{fe - 1} recomposto em {T_full}f (tfeather={tf})"
        else:
            final = out_block
            temporal_msg = "sem recomposicao temporal (bloco = video inteiro)"

        info = (f"stitch {cw}x{ch} @ ({x0},{y0}) em {W}x{H} | compose={compose} feather={int(feather)}px | "
                f"{temporal_msg} | {_fmt_t(time.time() - t0)}")
        print(f"[Bruxos Mocha Stitch] {info}", flush=True)
        return (final.clamp(0, 1), info)


NODE_CLASS_MAPPINGS = {
    "BruxosMochaBBoxCrop": BruxosMochaBBoxCrop,
    "BruxosMochaBBoxStitch": BruxosMochaBBoxStitch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosMochaBBoxCrop": "Mocha BBox Crop (Bruxos)",
    "BruxosMochaBBoxStitch": "Mocha BBox Stitch (Bruxos)",
}
