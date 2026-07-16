# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Bernini Infinity TILED (espacial, em pixels)
============================================================
Roda o Bernini Infinity POR LADRILHO pra alcancar resolucoes maiores em
QUALQUER funcao (remover, modificar, gerar, refinar) — o jeito que funciona
de verdade em V2V com referencia.

Por que este funciona (e o tiling no latente mosaicou):
  - Cada ladrilho recebe o PROPRIO PEDACO do video-fonte como conditioning.
    A posicao nao se perde porque o conteudo do ladrilho E a posicao: o modelo
    ve "um video pequeno completo" (o canto dele) e edita esse video.
  - CONSISTENCIA entre ladrilhos ("costura viva"): o ladrilho atual recebe, na
    faixa de sobreposicao, o resultado JA GERADO dos vizinhos (esquerda/cima/
    canto) colado na fonte, e a mascara e ZERADA ali -> o modelo trata como
    "ja pronto, case com isso". + fade complementar na montagem final.
  - PULA ladrilhos vazios: em remocao (inpaint), ladrilhos onde a mascara nao
    toca nem sao renderizados (saem da fonte) -> remocao em shot grande fica
    MAIS RAPIDA, nao mais lenta.

Custo honesto: N ladrilhos = N renders completos do Bernini (cada um menor).
Nao e "mais rapido" no caso geral: e "cabe na VRAM e sem mosaico".

Arquitetura inspirada no comfyUI-TiledWan (Baverne, GPL-3.0) — reimplementada
do zero para o pipeline Bernini (nenhum codigo copiado); creditado no README.
"""

import time
import logging

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Bernini Infinity do proprio pacote (o motor que roda cada ladrilho)
try:
    from .nodes import BerniniInfinity as _BERNINI
except Exception:
    try:
        from nodes import BerniniInfinity as _BERNINI  # execucao fora do pacote
    except Exception:
        _BERNINI = None

try:
    import comfy.samplers as _cs
    _SAMPLERS = list(getattr(_cs, "SAMPLER_NAMES", ["res_multistep", "euler"]))
    _SCHEDULERS = list(getattr(_cs, "SCHEDULER_NAMES", ["simple"]))
except Exception:
    _SAMPLERS = ["res_multistep", "euler"]
    _SCHEDULERS = ["simple"]

CAT = "Bruxos do VFX/Tiles"


# ----------------------------------------------------------------------------
# geometria (pixels): canvas preenchido ate multiplo de 16, ladrilhos UNIFORMES
# ----------------------------------------------------------------------------
def _ceil_div(a, d):
    return ((int(a) + d - 1) // d) * d


def _pad_replicate(img, Wp, Hp):
    H, W = int(img.shape[1]), int(img.shape[2])
    if (Hp, Wp) == (H, W):
        return img
    x = img.permute(0, 3, 1, 2)
    x = torch.nn.functional.pad(x, (0, Wp - W, 0, Hp - H), mode="replicate")
    return x.permute(0, 2, 3, 1).contiguous()


def _plan(Wp, Hp, cols, rows, ov, div=16):
    """Ladrilhos uniformes (multiplos de div) cobrindo o canvas, com sobreposicao."""
    cols, rows, ov = max(1, int(cols)), max(1, int(rows)), max(0, int(ov))
    tw = min(Wp, _ceil_div(-(-Wp // cols) + (2 * ov if cols > 1 else 0), div))
    th = min(Hp, _ceil_div(-(-Hp // rows) + (2 * ov if rows > 1 else 0), div))
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = min(max(0, (Wp * c) // cols - (ov if c > 0 else 0)), Wp - tw)
            y0 = min(max(0, (Hp * r) // rows - (ov if r > 0 else 0)), Hp - th)
            tiles.append({"r": r, "c": c, "x0": int(x0), "y0": int(y0),
                          "x1": int(x0 + tw), "y1": int(y0 + th)})
    return tiles, int(tw), int(th)


def _inter(a, b):
    x0, y0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
    x1, y1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
    return (x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None


def _ramp(n, up, down, device):
    w = torch.ones(n, device=device)
    up, down = min(int(up), n), min(int(down), n)
    if up > 0:
        w[:up] = torch.linspace(0.0, 1.0, up + 2, device=device)[1:-1]
    if down > 0:
        w[-down:] = torch.linspace(1.0, 0.0, down + 2, device=device)[1:-1]
    return w


class BruxosBerniniInfinityTiled:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "Positivo (do Prompt Guide). Vale pra TODOS os ladrilhos."}),
                "negative": ("CONDITIONING", {"tooltip": "Negativo."}),
                "high_model": ("MODEL", {"tooltip": "Modelo HIGH noise (mesmo do Bernini Infinity)."}),
                "low_model": ("MODEL", {"tooltip": "Modelo LOW noise."}),
                "vae": ("VAE", {"tooltip": "VAE de VIDEO do Wan."}),
                "source_video": ("IMAGE", {"tooltip": "O video-fonte. Sera redimensionado pra width x height e cortado em ladrilhos."}),
                "width": ("INT", {"default": 1664, "min": 64, "max": 8192, "step": 16, "tooltip": "Largura FINAL do resultado (a resolucao maior que voce quer). A fonte e redimensionada pra ca antes de cortar."}),
                "height": ("INT", {"default": 960, "min": 64, "max": 8192, "step": 16, "tooltip": "Altura FINAL do resultado."}),
                "tile_count_width": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "Colunas de ladrilho. Cada ladrilho roda um Bernini COMPLETO no seu pedaco. Dimensione pra cada ladrilho ficar perto de 832x480 (o doce do Wan). 1x1 = sem ladrilho."}),
                "tile_count_height": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "Linhas de ladrilho."}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 16, "tooltip": "Sobreposicao entre ladrilhos, em PIXELS. E onde a costura viva cola o vizinho ja gerado e o fade mistura. 64-96 e um bom comeco."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Semente base. Cada ladrilho usa seed + indice (evita padrao repetido)."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100, "tooltip": "Steps totais (igual ao Bernini Infinity). Com LoRA LightX2V: 6."}),
                "split_step": ("INT", {"default": 4, "min": 1, "max": 99, "tooltip": "Quantos steps no HIGH (o resto vai pro LOW). Com LightX2V: 4."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "CFG. Com LightX2V use 1.0."}),
                "sampler_name": (_SAMPLERS, {"tooltip": "Algoritmo de amostragem (ex.: res_multistep, euler)."}),
                "scheduler": (_SCHEDULERS, {"tooltip": "Scheduler (ex.: simple)."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Forca da modificacao POR ladrilho. 1.0 = gerar/remover/modificar de verdade. 0.3 = so refinar (upscale)."}),
                "mask_mode": (["off", "inpaint"], {"default": "off", "tooltip": "off = modifica o shot todo (guiado pelo prompt). inpaint = so a area da region_mask muda (remocao/edicao local). Com inpaint + pular_tiles_vazios, ladrilhos fora da mascara NEM RENDERIZAM (mais rapido)."}),
                "costura_viva": ("BOOLEAN", {"default": True, "tooltip": "LIGADO (recomendado): cada ladrilho recebe o resultado JA GERADO dos vizinhos na faixa de sobreposicao (com a mascara zerada ali) -> os ladrilhos casam entre si, sem drift de cor/conteudo. DESLIGADO: ladrilhos independentes (so o fade disfarca)."}),
                "pular_tiles_vazios": ("BOOLEAN", {"default": True, "tooltip": "[inpaint] Ladrilhos onde a mascara nao toca saem direto da fonte, sem renderizar. Remocao em shot grande fica MAIS RAPIDA."}),
            },
            "optional": {
                "region_mask": ("MASK,IMAGE", {"tooltip": "Mascara (p/ mask_mode=inpaint). Aceita MASK ou IMAGE colorida (SAM3/SCAIL)."}),
                "reference_video": ("IMAGE", {"tooltip": "Video de referencia (repassado a cada ladrilho)."}),
                "mode": (["context_window", "sequential"], {"default": "context_window", "tooltip": "Modo temporal do Bernini DENTRO de cada ladrilho (igual ao node normal)."}),
                "chunk_size": ("INT", {"default": 121, "min": 1, "max": 1024, "tooltip": "Frames por janela/chunk dentro de cada ladrilho."}),
                "overlap_frames": ("INT", {"default": 8, "min": 0, "max": 128, "tooltip": "Sobreposicao TEMPORAL (frames) dentro de cada ladrilho."}),
                "mask_grow": ("INT", {"default": 20, "min": -256, "max": 256, "tooltip": "[inpaint] Dilata a mascara (igual ao Bernini)."}),
                "mask_blur": ("INT", {"default": 6, "min": 0, "max": 256, "tooltip": "[inpaint] Suaviza a borda da mascara e a emenda da costura viva."}),
                "limpar_vram": (["off", "leve", "agressivo"], {"default": "leve", "tooltip": "Limpeza de memoria entre ladrilhos e dentro do Bernini (com o guard de re-stage)."}),
                "monitor_memoria": ("BOOLEAN", {"default": False, "tooltip": "Relatorio de RAM/VRAM por ladrilho no console."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "total_frames", "info")
    OUTPUT_TOOLTIPS = (
        "O video final na resolucao width x height, montado dos ladrilhos com fade (sem emenda).",
        "Numero de frames do resultado.",
        "Relatorio: grade, tamanho dos ladrilhos, quais renderizaram/pularam, tempo por ladrilho.",
    )
    FUNCTION = "render_tiled"
    CATEGORY = CAT
    DESCRIPTION = (
        "Bernini Infinity TILED: roda o Bernini COMPLETO por ladrilho (fonte recortada em pixels) "
        "pra alcancar resolucoes maiores em QUALQUER funcao (remover, modificar, gerar, refinar). "
        "A posicao nunca se perde (cada ladrilho ve o proprio pedaco da fonte) e a 'costura viva' "
        "cola o vizinho ja gerado na sobreposicao (mascara zerada ali) -> ladrilhos casam, sem drift. "
        "Em inpaint, ladrilhos fora da mascara nem renderizam. Custo: N ladrilhos = N renders (cada um menor). "
        "Arquitetura inspirada no TiledWan (reimplementada do zero p/ o Bernini)."
    )

    # ------------------------------------------------------------------ utils
    def _resize(self, video, W, H):
        x = video[..., :3].permute(0, 3, 1, 2).float()
        x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).clamp(0, 1)

    def _norm_mask(self, m, T, W, H):
        if m is None:
            return None
        mm = m
        if mm.dim() == 4:                      # IMAGE colorida -> intensidade
            mm = mm[..., :3].amax(dim=-1)
        elif mm.dim() == 2:
            mm = mm.unsqueeze(0)
        mm = mm.float().clamp(0, 1)
        if int(mm.shape[1]) != H or int(mm.shape[2]) != W:
            mm = torch.nn.functional.interpolate(mm.unsqueeze(1), size=(H, W), mode="bilinear",
                                                 align_corners=False).squeeze(1)
        if int(mm.shape[0]) < T:               # repete o ultimo frame
            mm = torch.cat([mm, mm[-1:].repeat(T - int(mm.shape[0]), 1, 1)], dim=0)
        return mm[:T].clamp(0, 1)

    # ------------------------------------------------------------------ main
    def render_tiled(self, positive, negative, high_model, low_model, vae, source_video,
                     width, height, tile_count_width, tile_count_height, tile_overlap,
                     seed, steps, split_step, cfg, sampler_name, scheduler, denoise,
                     mask_mode, costura_viva, pular_tiles_vazios,
                     region_mask=None, reference_video=None,
                     mode="context_window", chunk_size=121, overlap_frames=8,
                     mask_grow=20, mask_blur=6,
                     limpar_vram="leve", monitor_memoria=False, **kwargs):
        if not _HAS_TORCH:
            raise RuntimeError("[Bernini Tiled] torch indisponivel.")
        if _BERNINI is None:
            raise RuntimeError("[Bernini Tiled] nao achei o Bernini Infinity no pacote (nodes.py). "
                               "Este node roda o Bernini por ladrilho; instale o pacote completo.")

        t_start = time.time()
        W, H = int(width), int(height)
        cols, rows = int(tile_count_width), int(tile_count_height)
        ov = int(tile_overlap)

        # fonte na resolucao final + canvas multiplo de 16
        src = self._resize(source_video, W, H)
        T = int(src.shape[0])
        Wp, Hp = _ceil_div(W, 16), _ceil_div(H, 16)
        src = _pad_replicate(src, Wp, Hp)

        umask = self._norm_mask(region_mask, T, W, H)
        if umask is not None and (Wp, Hp) != (W, H):
            umask = torch.nn.functional.pad(umask.unsqueeze(1), (0, Wp - W, 0, Hp - H),
                                            mode="replicate").squeeze(1)
        if mask_mode == "inpaint" and umask is None:
            print("[Bernini Tiled] mask_mode=inpaint sem region_mask -> caindo pra 'off'.", flush=True)
            mask_mode = "off"

        tiles, tw, th = _plan(Wp, Hp, cols, rows, ov, 16)
        n_tiles = len(tiles)
        print(f"[Bernini Tiled] {W}x{H} x{T}f | grade {cols}x{rows} = {n_tiles} ladrilho(s) de {tw}x{th} "
              f"| sobreposicao {ov}px | costura_viva={'on' if costura_viva else 'off'}", flush=True)

        bern = _BERNINI()
        outs = [None] * n_tiles
        rendered, skipped = 0, 0
        log = []

        for i, t in enumerate(tiles):
            x0, y0, x1, y1 = t["x0"], t["y0"], t["x1"], t["y1"]
            src_tile = src[:, y0:y1, x0:x1, :].clone()

            # mascara por ladrilho: a do usuario (inpaint) ou tudo-branco (off)
            if mask_mode == "inpaint":
                m_tile = umask[:, y0:y1, x0:x1].clone()
            else:
                m_tile = torch.ones((T, th, tw), dtype=torch.float32)

            # ---- COSTURA VIVA: cola vizinhos ja gerados e zera a mascara ali ----
            if costura_viva:
                for (dr, dc) in ((0, -1), (-1, 0), (-1, -1)):     # esq, cima, canto
                    rr, cc = t["r"] + dr, t["c"] + dc
                    if rr < 0 or cc < 0:
                        continue
                    j = rr * cols + cc
                    if j >= n_tiles or outs[j] is None:
                        continue
                    nb = tiles[j]
                    it = _inter(t, nb)
                    if it is None:
                        continue
                    ix0, iy0, ix1, iy1 = it
                    strip = outs[j][:, iy0 - nb["y0"]:iy1 - nb["y0"], ix0 - nb["x0"]:ix1 - nb["x0"], :]
                    src_tile[:, iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0, :] = strip
                    m_tile[:, iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = 0.0

            # ---- pular ladrilho vazio (inpaint): mascara do usuario nao toca ----
            if (mask_mode == "inpaint" and pular_tiles_vazios
                    and float(umask[:, y0:y1, x0:x1].max()) < 0.02):
                outs[i] = src_tile
                skipped += 1
                log.append(f"tile {i + 1}/{n_tiles} (L{t['r']}C{t['c']}): PULADO (mascara vazia)")
                print(f"[Bernini Tiled] {log[-1]}", flush=True)
                continue

            # ---- renderiza o ladrilho com o Bernini COMPLETO ----
            eff_mode = "inpaint" if (mask_mode == "inpaint" or costura_viva) else "off"
            tt0 = time.time()
            print(f"[Bernini Tiled] tile {i + 1}/{n_tiles} (L{t['r']}C{t['c']}) em ({x0},{y0}) "
                  f"{tw}x{th} | mask={eff_mode} ...", flush=True)
            imgs, _lat, _tf = bern.render(
                positive=positive, negative=negative,
                high_model=high_model, low_model=low_model, vae=vae,
                source_video=src_tile,
                width=tw, height=th,
                seed=int(seed) + i, steps=int(steps), split_step=int(split_step),
                cfg=float(cfg), sampler_name=sampler_name, scheduler=scheduler,
                denoise=float(denoise),
                chunk_size=int(chunk_size), overlap=int(overlap_frames),
                max_frames=0, tail_memory=True, tail_frames=5,
                decode_tiled=False, decode_chunk=0, vary_seed_per_chunk=False,
                ref_max_size=848, mode=mode, context_jitter=True,
                mask_mode=eff_mode,
                mask_grow=int(mask_grow) if mask_mode == "inpaint" else 0,
                mask_blur=int(mask_blur),
                mask_pad=16, bbox_compose="rectangle", resize_mode="stretch",
                limpar_vram=limpar_vram, monitor_memoria=bool(monitor_memoria),
                guidance_mode="off",
                region_mask=m_tile if eff_mode == "inpaint" else None,
                reference_video=reference_video,
                **kwargs,
            )
            out = imgs.float().clamp(0, 1)
            if int(out.shape[0]) > T:
                out = out[:T]
            elif int(out.shape[0]) < T:
                out = torch.cat([out, out[-1:].repeat(T - int(out.shape[0]), 1, 1, 1)], dim=0)
            if int(out.shape[1]) != th or int(out.shape[2]) != tw:
                out = self._resize(out, tw, th)
            outs[i] = out.cpu()
            rendered += 1
            dt = time.time() - tt0
            log.append(f"tile {i + 1}/{n_tiles} (L{t['r']}C{t['c']}): ok em {dt:.0f}s")
            print(f"[Bernini Tiled] {log[-1]}", flush=True)

        # ---- montagem final: fade complementar (pesos somam 1) -----------------
        acc = torch.zeros((T, Hp, Wp, 3), dtype=torch.float32)
        wsum = torch.zeros((T, Hp, Wp, 1), dtype=torch.float32)
        for i, t in enumerate(tiles):
            x0, y0 = t["x0"], t["y0"]
            fl = fr = ft = fb = 0
            for j, nb in enumerate(tiles):
                if j == i:
                    continue
                it = _inter(t, nb)
                if it is None:
                    continue
                ix0, iy0, ix1, iy1 = it
                if nb["c"] < t["c"] and iy1 > iy0:
                    fl = max(fl, ix1 - x0)
                if nb["c"] > t["c"] and iy1 > iy0:
                    fr = max(fr, t["x1"] - ix0)
                if nb["r"] < t["r"] and ix1 > ix0:
                    ft = max(ft, iy1 - y0)
                if nb["r"] > t["r"] and ix1 > ix0:
                    fb = max(fb, t["y1"] - iy0)
            wx = _ramp(tw, fl, fr, "cpu")
            wy = _ramp(th, ft, fb, "cpu")
            wmap = (wy.view(th, 1) * wx.view(1, tw)).view(1, th, tw, 1)
            acc[:, y0:t["y1"], x0:t["x1"], :] += outs[i] * wmap
            wsum[:, y0:t["y1"], x0:t["x1"], :] += wmap
        final = (acc / wsum.clamp(min=1e-6))[:, :H, :W, :].clamp(0, 1)

        total_dt = time.time() - t_start
        info = (f"{W}x{H} x{T}f | {n_tiles} ladrilho(s) {tw}x{th} (grade {cols}x{rows}, ov {ov}px) | "
                f"renderizados {rendered}, pulados {skipped} | costura_viva={'on' if costura_viva else 'off'} | "
                f"total {total_dt / 60:.1f}min")
        print(f"[Bernini Tiled] DONE: {info}", flush=True)
        return (final, int(T), info)


NODE_CLASS_MAPPINGS = {"BruxosBerniniInfinityTiled": BruxosBerniniInfinityTiled}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosBerniniInfinityTiled": "Bernini Infinity Tiled (Bruxos)"}
