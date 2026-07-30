# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Wan Tiled Upscale (substitui o MM_Upscale gigante num node)
===========================================================================
Refino/upscale de VIDEO com o Wan (img2img de baixo denoise), ladrilhado no
ESPACO (tiles fundidos a cada passo, via o guider do wan_tiled) e janelado no
TEMPO (processa N frames por vez com crossfade) -> substitui, num node so, o
UltimateSDUpscale + os 3 subgraphs de batch/blend + os for-loops.

Pipeline por dentro:
  imagens -> (upscale ESRGAN opcional) -> resize p/ alvo -> [por janela temporal:
    VAE encode -> sampler ladrilhado (denoise) -> VAE decode -> crossfade] -> saida

Sem dependencia de node de terceiros pra logica (usa so o guider Bruxos +
nucleo do ComfyUI: SamplerCustomAdvanced/RandomNoise/BasicScheduler). O ESRGAN
e opcional e usa o ImageUpscaleWithModel do core so se um upscale_model for ligado.
"""

import os
import logging

import numpy as np

try:
    import torch
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Upscale"

# ---- helpers do nodes.py (encode/decode/sigmas/merge) ----------------------
try:
    from .nodes import (
        _encode_video as _bx_encode_video,
        _decode_video as _bx_decode_video,
        _align_up_4n1 as _bx_align_up_4n1,
        _mirror_pad_frames as _bx_mirror_pad,
        _merge_linear_overlap as _bx_merge_overlap,
        _mem_cleanup as _bx_mem_cleanup,
        BasicScheduler as _bx_BasicScheduler,
        KSamplerSelect as _bx_KSamplerSelect,
    )
    _HAS_HELPERS = True
except Exception:  # pragma: no cover
    try:
        from nodes import (
            _encode_video as _bx_encode_video,
            _decode_video as _bx_decode_video,
            _align_up_4n1 as _bx_align_up_4n1,
            _mirror_pad_frames as _bx_mirror_pad,
            _merge_linear_overlap as _bx_merge_overlap,
            _mem_cleanup as _bx_mem_cleanup,
            BasicScheduler as _bx_BasicScheduler,
            KSamplerSelect as _bx_KSamplerSelect,
        )
        _HAS_HELPERS = True
    except Exception as e:
        logging.warning(f"[Bruxos Wan Upscale] helpers do nodes.py indisponiveis: {e}")
        _bx_encode_video = _bx_decode_video = _bx_align_up_4n1 = None
        _bx_mirror_pad = _bx_merge_overlap = _bx_mem_cleanup = None
        _bx_BasicScheduler = _bx_KSamplerSelect = None
        _HAS_HELPERS = False

# ---- guider de ladrilho espacial (wan_tiled) -------------------------------
try:
    from .wan_tiled import _WanTiledGuider as _BX_GUIDER
except Exception:
    try:
        from wan_tiled import _WanTiledGuider as _BX_GUIDER
    except Exception:
        _BX_GUIDER = None

try:
    import comfy.samplers as _cs
    _SAMPLERS = list(getattr(_cs, "SAMPLER_NAMES", ["euler"]))
    _SCHEDULERS = list(getattr(_cs, "SCHEDULER_NAMES", ["simple"]))
except Exception:
    _SAMPLERS = ["euler", "res_multistep", "dpmpp_2m"]
    _SCHEDULERS = ["simple", "beta", "normal"]

try:
    import comfy.utils as _cu
except Exception:
    _cu = None


def _get_cls(name):
    """Pega uma classe de node do core pelo nome (ex.: SamplerCustomAdvanced)."""
    try:
        import nodes as _core
        m = getattr(_core, "NODE_CLASS_MAPPINGS", {})
        if name in m:
            return m[name]
    except Exception:
        pass
    return None


def _call(cls, **kw):
    """Instancia e chama a FUNCTION do node com kwargs. Retorna a saida crua."""
    inst = cls()
    fn = getattr(inst, getattr(cls, "FUNCTION", "execute"), None)
    if fn is None:
        raise RuntimeError(f"node {cls} sem FUNCTION")
    return fn(**kw)


def _first(out):
    """Extrai a 1a saida de um node. Cobre tupla/lista, o NodeOutput V3 do
    ComfyUI 0.28 (que expoe .args), e objetos indexaveis."""
    if isinstance(out, (tuple, list)):
        return out[0]
    args = getattr(out, "args", None)
    if isinstance(args, (tuple, list)) and len(args):
        return args[0]
    try:
        return out[0]
    except Exception:
        return out


# ---------------------------------------------------------------------------
def _resize_bhwc(images, tw, th):
    """images [B,H,W,3] 0..1 -> [B,th,tw,3] lanczos (via common_upscale)."""
    B, H, W, C = images.shape
    if tw <= 0 or th <= 0 or (tw == W and th == H):
        return images
    x = images.permute(0, 3, 1, 2)
    if _cu is not None:
        x = _cu.common_upscale(x, tw, th, "lanczos", "disabled")
    else:
        x = torch.nn.functional.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False)
    return x.permute(0, 2, 3, 1).clamp(0, 1)


def _esrgan(images, upscale_model):
    """Roda o ImageUpscaleWithModel do core, se disponivel. Retorna [B,H',W',3]."""
    cls = _get_cls("ImageUpscaleWithModel")
    if cls is None or upscale_model is None:
        return images
    out = _call(cls, upscale_model=upscale_model, image=images)
    return _first(out)


def _save_png_seq(images, folder, start_idx):
    """Salva [B,H,W,3] 0..1 como PNGs numerados. Retorna nova contagem."""
    os.makedirs(folder, exist_ok=True)
    arr = (images.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    try:
        from PIL import Image
        for i in range(arr.shape[0]):
            Image.fromarray(arr[i]).save(os.path.join(folder, f"frame_{start_idx + i:06d}.png"))
    except Exception:
        import cv2
        for i in range(arr.shape[0]):
            cv2.imwrite(os.path.join(folder, f"frame_{start_idx + i:06d}.png"),
                        cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR))
    return start_idx + arr.shape[0]


def _crossfade_pair(a, b):
    """a,b: [n,H,W,3] mesmo n -> crossfade linear a->b (borda entre janelas)."""
    n = int(a.shape[0])
    if n == 0:
        return b
    w = torch.linspace(1.0, 0.0, steps=n).view(n, 1, 1, 1).to(a.dtype)
    return (a * w + b * (1.0 - w)).clamp(0, 1)


def _load_png_seq(folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
    if not files:
        raise RuntimeError(f"[Bruxos Wan Upscale] nada em {folder} pra recarregar.")
    frames = []
    try:
        from PIL import Image
        for f in files:
            frames.append(np.array(Image.open(os.path.join(folder, f)).convert("RGB")))
    except Exception:
        import cv2
        for f in files:
            im = cv2.imread(os.path.join(folder, f))
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    arr = np.stack(frames, 0).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


# ===========================================================================
class BruxosWanTiledUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frames do video a dar upscale (do Load Video/Image)."}),
                "model": ("MODEL", {"tooltip": "Wan (com ModelSamplingSD3/LoRAs ja aplicados antes, se quiser)."}),
                "positive": ("CONDITIONING", {"tooltip": "Prompt positivo (do CLIP Text Encode / Prompt Source)."}),
                "negative": ("CONDITIONING", {"tooltip": "Prompt negativo."}),
                "vae": ("VAE", {"tooltip": "VAE do Wan."}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "[opcional] Modelo ESRGAN (ex.: 4x_UniversalUpscaler) pro pre-upscale. Sem ele, so faz resize lanczos + refino."}),
                # ---- RESOLUCAO (voce controla) ----
                "target_width": ("INT", {"default": 1920, "min": 0, "max": 16384, "step": 8,
                    "tooltip": "Largura FINAL. 0 = mantem o que sair do ESRGAN/entrada."}),
                "target_height": ("INT", {"default": 1088, "min": 0, "max": 16384, "step": 8,
                    "tooltip": "Altura FINAL. 0 = mantem."}),
                # ---- REFINO (img2img) ----
                "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Forca do refino. 0.15-0.25 = adiciona detalhe SEM reinventar. Alto demais muda o conteudo."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Passos totais do schedule. Com LoRA 4-steps use ~4-8."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "CFG. Com LoRA destilado (LightX2V/4steps) use 1.0."}),
                "sampler_name": (_SAMPLERS, {"default": ("res_2s" if "res_2s" in _SAMPLERS else _SAMPLERS[0]),
                    "tooltip": "Algoritmo de amostragem."}),
                "scheduler": (_SCHEDULERS, {"default": ("beta57" if "beta57" in _SCHEDULERS else _SCHEDULERS[0]),
                    "tooltip": "Agenda de sigmas."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # ---- LADRILHO ESPACIAL ----
                "tile_w": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Colunas de ladrilho. 1 = nao corta na horizontal."}),
                "tile_h": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Linhas de ladrilho. 2x2 = 4 tiles (menos VRAM, costura fundida a cada passo)."}),
                "tile_overlap": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Sobreposicao dos tiles em LATENTE (1~8px)."}),
                # ---- JANELA TEMPORAL ----
                "chunk_size": ("INT", {"default": 81, "min": 0, "max": 1024, "step": 1,
                    "tooltip": "Frames por janela temporal (substitui o batch de 81). 0 = video inteiro de uma vez. Use 4n+1 (49/81). Menor = menos VRAM, mais janelas."}),
                "overlap_frames": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Frames de sobreposicao entre janelas pro crossfade (sem emenda). 8-16 e bom."}),
                # ---- ARMAZENAMENTO: memoria vs disco ----
                "armazenamento": (["memoria", "disco"], {"default": "memoria",
                    "tooltip": "MEMORIA: guarda tudo na RAM e devolve o video inteiro pela saida (rapido; ideal p/ videos curtos/medios). DISCO: grava cada janela como PNG numerado numa pasta e SO no fim recarrega — a RAM/VRAM nunca segura o video todo durante o processo (ideal p/ videos longos em 4K, e ainda deixa a sequencia salva)."}),
                "disco_prefix": ("STRING", {"default": "Bruxos/upscale",
                    "tooltip": "[disco] Subpasta/prefixo dentro de ComfyUI/output pra sequencia PNG."}),
                # ---- retoques opcionais ----
                "grain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.005,
                    "tooltip": "Ruido fino ANTES do refino (ajuda o modelo a criar textura). 0.01 e sutil. 0 = off."}),
                "limpar_vram": ("BOOLEAN", {"default": True,
                    "tooltip": "Esvazia a VRAM entre tiles/janelas. Deixe ligado."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    OUTPUT_TOOLTIPS = ("Frames com upscale.", "Resumo do processo.")
    FUNCTION = "upscale"
    CATEGORY = CAT
    DESCRIPTION = (
        "Wan Tiled Upscale (Bruxos): refino/upscale de video num node so. Pre-upscale ESRGAN opcional, "
        "resize pro alvo, e refino img2img LADRILHADO no espaco (tiles fundidos a cada passo) e JANELADO "
        "no tempo (substitui o batch de 81f + blends). Escolha memoria ou disco. Sem for-loops, sem "
        "salvar/recarregar manual, sem MathExpression."
    )

    def upscale(self, images, model, positive, negative, vae,
                upscale_model=None, target_width=1920, target_height=1088,
                denoise=0.2, steps=6, cfg=1.0, sampler_name=None, scheduler=None, seed=0,
                tile_w=2, tile_h=2, tile_overlap=8,
                chunk_size=81, overlap_frames=8,
                armazenamento="memoria", disco_prefix="Bruxos/upscale",
                grain=0.0, limpar_vram=True):
        if not _OK or not _HAS_HELPERS:
            raise RuntimeError("[Bruxos Wan Upscale] torch/helpers indisponiveis neste build.")
        if _BX_GUIDER is None:
            raise RuntimeError("[Bruxos Wan Upscale] guider do wan_tiled nao importou.")
        sca = _get_cls("SamplerCustomAdvanced")
        rnd = _get_cls("RandomNoise")
        if sca is None or rnd is None:
            raise RuntimeError("[Bruxos Wan Upscale] SamplerCustomAdvanced/RandomNoise (core) nao encontrados.")

        import time
        t0 = time.time()
        imgs = images.float().clamp(0, 1)
        B = int(imgs.shape[0])

        # 1) pre-upscale ESRGAN (opcional) + resize pro alvo
        if upscale_model is not None:
            imgs = _esrgan(imgs, upscale_model).float().clamp(0, 1)
        tw = int(target_width) if int(target_width) > 0 else int(imgs.shape[2])
        th = int(target_height) if int(target_height) > 0 else int(imgs.shape[1])
        imgs = _resize_bhwc(imgs, tw, th)
        print(f"[Bruxos Wan Upscale] {B}f -> alvo {tw}x{th} | tiles {tile_w}x{tile_h} ov{tile_overlap} | "
              f"chunk {chunk_size}f ov {overlap_frames} | denoise {denoise} | armazenamento={armazenamento}",
              flush=True)

        if grain and float(grain) > 0:
            g = torch.randn_like(imgs) * float(grain)
            imgs = (imgs + g).clamp(0, 1)

        # 2) sampler + guider (ladrilho espacial)
        sampler = _bx_KSamplerSelect.execute(sampler_name).args[0]
        guider = _BX_GUIDER(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(float(cfg))
        guider.set_tiling(int(tile_h), int(tile_w), int(tile_overlap), "hann", bool(limpar_vram), False)

        def _refina_janela(win_imgs, seed_):
            lat = _bx_encode_video(vae, win_imgs)
            sigmas = _bx_BasicScheduler.execute(model, scheduler, int(steps), float(denoise)).args[0]
            noise = _first(_call(rnd, noise_seed=int(seed_)))
            out = _call(sca, noise=noise, guider=guider, sampler=sampler, sigmas=sigmas,
                        latent_image={"samples": lat})
            samples = _first(out)["samples"]
            dec = _bx_decode_video(vae, samples, False).float().clamp(0, 1)
            return dec

        # 3) janela temporal (memoria OU disco)
        L = int(imgs.shape[0])
        use_disk = (armazenamento == "disco")
        out_folder = None
        if use_disk:
            try:
                import folder_paths
                base = folder_paths.get_output_directory()
            except Exception:
                base = os.getcwd()
            out_folder = os.path.join(base, *disco_prefix.replace("\\", "/").split("/"))
            os.makedirs(out_folder, exist_ok=True)

        cs = int(chunk_size)
        ov = max(0, int(overlap_frames))
        # blindagem: overlap NUNCA pode ser >= chunk (senao fstep=1 -> janelas
        # demais, lentissimo). Clampa pra no maximo metade do chunk.
        if cs > 0 and ov >= cs:
            ov_novo = max(0, min(ov, cs // 2))
            print(f"[Bruxos Wan Upscale] overlap_frames {ov} >= chunk {cs} -> ajustado p/ {ov_novo} "
                  f"(senao processaria 1 frame por vez).", flush=True)
            ov = ov_novo
        if cs <= 0 or cs >= L:
            # video inteiro de uma vez
            aligned = _bx_align_up_4n1(L)
            win = imgs if aligned == L else _bx_mirror_pad(imgs, aligned)
            dec = _refina_janela(win, seed)[:L].cpu()
            if use_disk:
                _save_png_seq(dec, out_folder, 0)
            final = dec
        elif use_disk:
            # DISCO (streaming): segura so a cauda (ov frames) na RAM; grava o resto.
            fstep = max(1, cs - ov)
            prev_tail = None
            disk_count = 0
            bi = 0
            for start in range(0, L, fstep):
                end = min(start + cs, L)
                blk = imgs[start:end]
                blk_len = int(blk.shape[0])
                if blk_len <= 0:
                    break
                aligned = _bx_align_up_4n1(blk_len)
                win = blk if aligned == blk_len else _bx_mirror_pad(blk, aligned)
                dec = _refina_janela(win, int(seed) + bi)[:blk_len].cpu()
                bi += 1
                head = 0
                if prev_tail is not None:
                    k = min(ov, int(prev_tail.shape[0]), blk_len)
                    if k > 0:
                        disk_count = _save_png_seq(_crossfade_pair(prev_tail[:k], dec[:k]),
                                                   out_folder, disk_count)
                        head = k
                if end >= L:
                    disk_count = _save_png_seq(dec[head:], out_folder, disk_count)
                    prev_tail = None
                    print(f"[Bruxos Wan Upscale][disco] janela {bi}: {start}..{end-1} ok (final)", flush=True)
                    break
                keep = min(ov, blk_len - head) if ov > 0 else 0
                cut = blk_len - keep
                disk_count = _save_png_seq(dec[head:cut], out_folder, disk_count)
                prev_tail = dec[cut:] if keep > 0 else None
                print(f"[Bruxos Wan Upscale][disco] janela {bi}: {start}..{end-1} ok", flush=True)
                del dec
                _bx_mem_cleanup(limpar_vram)
            final = _load_png_seq(out_folder)
        else:
            # MEMORIA: acumula o video inteiro com crossfade.
            fstep = max(1, cs - ov)
            stitched = None
            bi = 0
            for start in range(0, L, fstep):
                end = min(start + cs, L)
                blk = imgs[start:end]
                blk_len = int(blk.shape[0])
                if blk_len <= 0:
                    break
                aligned = _bx_align_up_4n1(blk_len)
                win = blk if aligned == blk_len else _bx_mirror_pad(blk, aligned)
                dec = _refina_janela(win, int(seed) + bi)[:blk_len].cpu()
                bi += 1
                stitched = dec if stitched is None else _bx_merge_overlap(stitched, dec, ov)
                print(f"[Bruxos Wan Upscale] janela {bi}: frames {start}..{end-1} ({blk_len}f) ok", flush=True)
                _bx_mem_cleanup(limpar_vram)
                if end >= L:
                    break
            final = stitched if stitched is not None else imgs.cpu()

        dt = time.time() - t0
        info = (f"{tw}x{th} x{int(final.shape[0])}f | tiles {tile_w}x{tile_h} | chunk {chunk_size} | "
                f"denoise {denoise} | {armazenamento}" + (f" -> {out_folder}" if use_disk else "") +
                f" | {dt/60:.1f}min")
        print(f"[Bruxos Wan Upscale] DONE: {info}", flush=True)
        return (final.clamp(0, 1), info)


NODE_CLASS_MAPPINGS = {"BruxosWanTiledUpscale": BruxosWanTiledUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosWanTiledUpscale": "Wan Tiled Upscale (Bruxos)"}
