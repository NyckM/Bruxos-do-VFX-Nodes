# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Ultimate Upscale Video (facil + rapido)
=======================================================
Um node so que junta o que a workflow gigante fazia em ~90 nodes:
  ESRGAN (pre-upscale opcional) -> resize pro alvo -> [por LOTE de frames:
  UltimateSDUpscale (No Upscale) — tiles espaciais otimizados] -> junta.

Diferente do BruxosWanTiledUpscale (que refazia o sampler ladrilhado na mao e
ficava lento com DynamicVRAM), aqui a otimizacao e o PROPRIO UltimateSDUpscale
instalado — o mesmo que voce usava e era rapido. So embrulhamos ele + ESRGAN +
o batching temporal num node, pra sumir com a confusao de subgraphs/for-loops.

Requer o node UltimateSDUpscale instalado (comfyui_ultimatesdupscale) — voce ja tem.
"""

import time
import logging

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

# helpers reaproveitados do outro node de upscale
try:
    from .bruxos_wan_upscale import (
        _get_cls, _call, _first, _esrgan, _resize_bhwc, _crossfade_pair,
    )
    from .nodes import _mem_cleanup as _bx_mem_cleanup, _merge_linear_overlap as _bx_merge_overlap
    _HAS = True
except Exception:  # pragma: no cover
    try:
        from bruxos_wan_upscale import _get_cls, _call, _first, _esrgan, _resize_bhwc, _crossfade_pair
        from nodes import _mem_cleanup as _bx_mem_cleanup, _merge_linear_overlap as _bx_merge_overlap
        _HAS = True
    except Exception as e:
        logging.warning(f"[Bruxos Ultimate Upscale] helpers indisponiveis: {e}")
        _HAS = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Upscale"

try:
    import comfy.samplers as _cs
    _SAMPLERS = list(getattr(_cs, "SAMPLER_NAMES", ["euler"]))
    _SCHEDULERS = list(getattr(_cs, "SCHEDULER_NAMES", ["simple"]))
except Exception:
    _SAMPLERS = ["euler", "res_2s", "dpmpp_2m"]
    _SCHEDULERS = ["simple", "beta", "normal"]

# nome do node UltimateSD (variantes por versao)
_USDU_NAMES = ["UltimateSDUpscaleNoUpscale", "UltimateSDUpscale"]
_SEAM_MODES = ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"]


def _fmt_t(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{s - 60 * int(s // 60):04.1f}s"


def _get_usdu():
    for n in _USDU_NAMES:
        c = _get_cls(n)
        if c is not None:
            return c, n
    return None, None


class BruxosUltimateUpscaleVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frames do video a dar upscale."}),
                "model": ("MODEL", {"tooltip": "Modelo Wan (com LoRA/ModelSamplingSD3 ja aplicados)."}),
                "positive": ("CONDITIONING", {"tooltip": "Prompt positivo."}),
                "negative": ("CONDITIONING", {"tooltip": "Prompt negativo."}),
                "vae": ("VAE", {"tooltip": "VAE do Wan."}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "[opcional] ESRGAN pro pre-upscale (ex.: 4x_UniversalUpscaler). Sem ele, so faz resize lanczos + refino."}),
                # ---- RESOLUCAO ----
                "target_width": ("INT", {"default": 1920, "min": 0, "max": 16384, "step": 8,
                    "tooltip": "Largura FINAL. 0 = mantem o que sair do ESRGAN."}),
                "target_height": ("INT", {"default": 1088, "min": 0, "max": 16384, "step": 8,
                    "tooltip": "Altura FINAL. 0 = mantem."}),
                # ---- REFINO (UltimateSD) ----
                "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Forca do refino. 0.15-0.25 adiciona detalhe sem reinventar."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100, "tooltip": "Passos. Com LoRA 4-steps use ~4-8."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "CFG. LoRA destilado = 1.0."}),
                "sampler_name": (_SAMPLERS, {"default": ("res_2s" if "res_2s" in _SAMPLERS else _SAMPLERS[0])}),
                "scheduler": (_SCHEDULERS, {"default": ("beta57" if "beta57" in _SCHEDULERS else _SCHEDULERS[0])}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # ---- TILE espacial (UltimateSD) ----
                "tile_width": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 64,
                    "tooltip": "Largura do tile do UltimateSD (px na resolucao final). Maior = menos tiles, mais VRAM."}),
                "tile_height": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 64,
                    "tooltip": "Altura do tile do UltimateSD."}),
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 256, "tooltip": "Suaviza a emenda dos tiles."}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 256, "tooltip": "Contexto ao redor de cada tile."}),
                "seam_fix_mode": (_SEAM_MODES, {"default": "None", "tooltip": "Correcao de emenda extra. 'None' costuma bastar com mask_blur."}),
                # ---- BATCHING temporal (o que os subgraphs faziam) ----
                "batch_size": ("INT", {"default": 81, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Quantos frames processar por vez (evita OOM em video longo). 0 = todos de uma vez. Era o 'Frames per Iteration' do fluxo antigo."}),
                "batch_overlap": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Frames de sobreposicao entre lotes p/ crossfade (suaviza a emenda temporal). 0 = corte seco (mais rapido)."}),
                "limpar_vram": ("BOOLEAN", {"default": True, "tooltip": "Limpa a VRAM entre os lotes."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    OUTPUT_TOOLTIPS = ("Frames com upscale.", "Resumo.")
    FUNCTION = "upscale"
    CATEGORY = CAT
    DESCRIPTION = (
        "Ultimate Upscale Video (Bruxos): ESRGAN + resize + UltimateSDUpscale (tiles otimizados) + "
        "batching temporal, tudo num node. Usa o UltimateSDUpscale instalado (rapido, o mesmo do seu "
        "fluxo antigo), sem a confusao de subgraphs/for-loops. Precisa do comfyui_ultimatesdupscale."
    )

    def upscale(self, images, model, positive, negative, vae,
                upscale_model=None, target_width=1920, target_height=1088,
                denoise=0.2, steps=8, cfg=1.0, sampler_name=None, scheduler=None, seed=0,
                tile_width=1024, tile_height=1024, mask_blur=16, tile_padding=32,
                seam_fix_mode="None", batch_size=81, batch_overlap=0, limpar_vram=True):
        if torch is None or not _HAS:
            raise RuntimeError("[Bruxos Ultimate Upscale] torch/helpers indisponiveis.")
        usdu_cls, usdu_name = _get_usdu()
        if usdu_cls is None:
            raise RuntimeError("[Bruxos Ultimate Upscale] UltimateSDUpscale nao encontrado. "
                               "Instale o comfyui_ultimatesdupscale (ComfyUI-Manager).")
        t0 = time.time()
        imgs = images.float().clamp(0, 1)
        B = int(imgs.shape[0])

        # 1) pre-upscale ESRGAN + resize pro alvo
        if upscale_model is not None:
            imgs = _esrgan(imgs, upscale_model).float().clamp(0, 1)
        tw = int(target_width) if int(target_width) > 0 else int(imgs.shape[2])
        th = int(target_height) if int(target_height) > 0 else int(imgs.shape[1])
        imgs = _resize_bhwc(imgs, tw, th)
        print(f"[Bruxos Ultimate Upscale] {B}f -> {tw}x{th} | tile {tile_width}x{tile_height} | "
              f"denoise {denoise} | lote {batch_size}f (ov {batch_overlap}) | via {usdu_name}", flush=True)

        # parametros fixos do UltimateSD (o resto vem dos widgets)
        base_kw = dict(
            model=model, positive=positive, negative=negative, vae=vae,
            upscale_model=upscale_model,          # ignorado pelo NoUpscale; alguns aceitam
            upscale_by=1.0,                        # ja resolvido no resize acima
            seed=int(seed), steps=int(steps), cfg=float(cfg),
            sampler_name=sampler_name, scheduler=scheduler, denoise=float(denoise),
            mode_type="Linear", tile_width=int(tile_width), tile_height=int(tile_height),
            mask_blur=int(mask_blur), tile_padding=int(tile_padding),
            seam_fix_mode=str(seam_fix_mode), seam_fix_denoise=1.0, seam_fix_width=64,
            seam_fix_mask_blur=8, seam_fix_padding=16,
            force_uniform_tiles=True, tiled_decode=False,
        )

        # descobre exatamente quais parametros ESTE UltimateSD aceita
        accepted = set()
        try:
            it = usdu_cls.INPUT_TYPES()
            for sec in ("required", "optional"):
                accepted |= set(it.get(sec, {}).keys())
        except Exception:
            accepted = set(base_kw.keys()) | {"image"}
        img_key = "image" if "image" in accepted else ("upscaled_image" if "upscaled_image" in accepted else "image")

        def _usdu(batch):
            kw = {k: v for k, v in base_kw.items() if k in accepted}
            kw[img_key] = batch
            out = _call(usdu_cls, **kw)
            return _first(out).float().clamp(0, 1)

        # 2) batching temporal
        L = int(imgs.shape[0])
        bs = int(batch_size)
        ov = max(0, int(batch_overlap))
        if bs <= 0 or bs >= L:
            final = _usdu(imgs).cpu()
        else:
            if ov >= bs:
                ov = max(0, bs // 2)
            fstep = max(1, bs - ov)
            stitched = None
            bi = 0
            for start in range(0, L, fstep):
                end = min(start + bs, L)
                blk = imgs[start:end]
                if int(blk.shape[0]) <= 0:
                    break
                res = _usdu(blk).cpu()
                bi += 1
                if stitched is None:
                    stitched = res
                elif ov > 0:
                    stitched = _bx_merge_overlap(stitched, res, ov)
                else:
                    stitched = torch.cat([stitched, res], dim=0)
                print(f"[Bruxos Ultimate Upscale] lote {bi}: frames {start}..{end-1} ok", flush=True)
                _bx_mem_cleanup(limpar_vram)
                if end >= L:
                    break
            final = stitched if stitched is not None else imgs.cpu()

        info = (f"{tw}x{th} x{int(final.shape[0])}f | tile {tile_width}x{tile_height} | "
                f"denoise {denoise} | lote {batch_size} | {usdu_name} | {_fmt_t(time.time() - t0)}")
        print(f"[Bruxos Ultimate Upscale] DONE: {info}", flush=True)
        return (final.clamp(0, 1), info)


NODE_CLASS_MAPPINGS = {"BruxosUltimateUpscaleVideo": BruxosUltimateUpscaleVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosUltimateUpscaleVideo": "Ultimate Upscale Video (Bruxos)"}
