# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — LTX Tiled Sampler (step-fused)
===============================================
Guider de ladrilho PRA LTX (nao pro Wan/Bernini): corta o LATENTE (5D:
batch,canais,frames,H,W) em ladrilhos espaciais e funde as PREDICOES DE RUIDO
a cada passo de denoise, com janela complementar (soma 1 na sobreposicao) --
o mesmo "step-fused tiled sampler" do wan_tiled.py (Bruxos), so que adaptado
pro LTX em vez do Wan.

Por que step-fused (nao "N renders completos por ladrilho"):
  A LTXVTiledSampler oficial da Lightricks (tiled_sampler.py) roda um
  SamplerCustomAdvanced.sample() INTEIRO por ladrilho (cada ladrilho e um
  video pequeno independente do inicio ao fim) e so funde os latentes DEPOIS
  de cada um estar 100% denoised. Isso funciona, mas cada ladrilho pode
  divergir em cor/conteudo porque nunca "viu" os vizinhos durante o processo.
  Aqui a fusao acontece A CADA PASSO: o modelo preve o ruido em cada ladrilho
  e as predicoes sao fundidas antes do passo de sampler seguinte -- os
  ladrilhos "se enxergam" o tempo todo, entao a imagem sai mais coerente.
  Custo: mais forwards por passo (rows x cols), nao menos passos totais.

Limitacao HONESTA (leia antes de usar): este guider so recorta o latente e
as chaves de conditioning ESPACIALMENTE COLADAS (mask/concat_latent_image
etc, a mesma lista conservadora do wan_tiled.py). Ele NAO reproduz o guia de
imagem por ladrilho da LTXVTiledSampler oficial (LTXVAddGuide/
LTXVAddLatentGuide, que recorta e re-injeta a imagem de condicionamento por
ladrilho) -- isso e um recurso avancado do LTX que precisaria de mais
integracao pra fazer com seguranca. Pra T2V/I2V simples (sem guia de imagem
por ladrilho), este guider funciona; pra fluxos com LTXVAddGuide pesado,
prefira a LTXVTiledSampler oficial da Lightricks por enquanto.

Credito: a arquitetura "step-fused" e a janela complementar seguem o
comfyui-deno-custom-nodes (Deno2026), feito originalmente pro LTX (arquivo
deno_ltx_tiling.py) -- reimplementado do zero aqui (nenhum codigo copiado),
igual ao wan_tiled.py ja faz pro Wan.
"""

import gc
import math
import time

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import comfy.samplers as _samplers
    import comfy.model_management as _mm
except Exception:
    _samplers = None
    _mm = None

CAT = "Bruxos do VFX/Tiles"


# ----------------------------------------------------------------------------
# Flash/Sage Attention: so diagnostico (mesma logica do bernini_tiled_optimized
# -- o backend e escolhido no LAUNCH do ComfyUI, nao por node).
# ----------------------------------------------------------------------------
def _bx_attention_backend_info():
    info = {"sage_installed": False, "flash_installed": False, "active_flag": None}
    try:
        import sageattention  # noqa: F401
        info["sage_installed"] = True
    except Exception:
        pass
    try:
        import flash_attn  # noqa: F401
        info["flash_installed"] = True
    except Exception:
        pass
    try:
        import comfy.cli_args as _cli
        a = _cli.args
        if getattr(a, "use_sage_attention", False):
            info["active_flag"] = "sage (--use-sage-attention)"
        elif getattr(a, "use_flash_attention", False):
            info["active_flag"] = "flash (--use-flash-attention)"
        elif getattr(a, "use_pytorch_cross_attention", False):
            info["active_flag"] = "pytorch/sdpa (--use-pytorch-cross-attention)"
    except Exception:
        pass
    return info


_BX_LTX_ATTN_LOGGED = {"done": False}


def _bx_log_attention_backend_once():
    if _BX_LTX_ATTN_LOGGED["done"]:
        return
    _BX_LTX_ATTN_LOGGED["done"] = True
    try:
        info = _bx_attention_backend_info()
        active = info["active_flag"] or "padrao do ComfyUI (provavelmente pytorch/sdpa)"
        print(f"[Bruxos LTX Tiled][attention] backend ativo: {active} | "
              f"sageattention instalado: {'sim' if info['sage_installed'] else 'nao'} | "
              f"flash-attn instalado: {'sim' if info['flash_installed'] else 'nao'}", flush=True)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# geometria (latente): plano de ladrilhos + janela de blend complementar.
# Reimplementacao propria (espirito identico ao deno_ltx_tiling.py: mesmo
# _axis_plan por arredondamento linear, mesma janela cosseno complementar).
# ----------------------------------------------------------------------------
def _axis_plan(total, count, overlap):
    total, count, overlap = int(total), max(1, int(count)), max(0, int(overlap))
    if count == 1 or total <= 1:
        return [0], total
    tile_size = min(total, math.ceil((total + (count - 1) * overlap) / count))
    if overlap >= tile_size:
        overlap = max(0, tile_size - 1)
    travel = total - tile_size
    starts = [int(round(i * travel / (count - 1))) for i in range(count)]
    starts[0], starts[-1] = 0, travel
    uniq = sorted(set(starts))
    return uniq, tile_size


def _tile_plan(H, W, rows, cols, overlap):
    ys, th = _axis_plan(H, rows, overlap)
    xs, tw = _axis_plan(W, cols, overlap)
    y_end = [min(y + th, H) for y in ys]
    x_end = [min(x + tw, W) for x in xs]
    specs = []
    for r, (y0, y1) in enumerate(zip(ys, y_end)):
        f_top = max(0, y_end[r - 1] - y0) if r > 0 else 0
        f_bot = max(0, y1 - ys[r + 1]) if r < len(ys) - 1 else 0
        for c, (x0, x1) in enumerate(zip(xs, x_end)):
            f_left = max(0, x_end[c - 1] - x0) if c > 0 else 0
            f_right = max(0, x1 - xs[c + 1]) if c < len(xs) - 1 else 0
            specs.append({"row": r, "col": c, "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                          "ft": f_top, "fb": f_bot, "fl": f_left, "fr": f_right})
    return specs


def _win1d(size, fade_a, fade_b, device, dtype):
    w = torch.ones(size, device=device, dtype=dtype)
    fa = min(max(int(fade_a), 0), size)
    fb = min(max(int(fade_b), 0), size)
    if fa:
        i = torch.arange(fa, device=device, dtype=dtype)
        w[:fa] = 0.5 * (1.0 - torch.cos(math.pi * i / fa))
    if fb:
        i = torch.arange(fb, device=device, dtype=dtype)
        w[-fb:] = 0.5 * (1.0 + torch.cos(math.pi * i / fb))
    return w


def _win2d(spec, device, dtype):
    h = spec["y1"] - spec["y0"]
    w = spec["x1"] - spec["x0"]
    vy = _win1d(h, spec["ft"], spec["fb"], device, dtype)
    vx = _win1d(w, spec["fl"], spec["fr"], device, dtype)
    return vy[:, None] * vx[None, :]


# ----------------------------------------------------------------------------
# recorte SEGURO de conditioning: so as chaves espacialmente coladas ao
# latente (mask/concat_latent_image etc). Qualquer outra coisa (embeddings de
# texto, guias de imagem ja processadas, streams de referencia) passa
# INTACTA -- cortar isso do jeito errado bagunca posicao/RoPE, igual o
# wan_tiled.py ja documenta pro Wan.
# ----------------------------------------------------------------------------
_BX_CROP_KEYS = {"mask", "noise_mask", "concat_latent_image", "concat_mask", "denoise_mask"}


def _crop_t(t, s, H, W):
    if torch.is_tensor(t) and t.dim() >= 3 and tuple(t.shape[-2:]) == (H, W):
        return t[..., s["y0"]:s["y1"], s["x0"]:s["x1"]].contiguous()
    return t


def _crop_val(v, s, H, W):
    if torch.is_tensor(v):
        return _crop_t(v, s, H, W)
    if isinstance(v, (list, tuple)) and v and all(torch.is_tensor(i) for i in v):
        return [_crop_t(i, s, H, W) for i in v]
    return v


def _crop_cond_list(cond, s, H, W):
    if not cond:
        return cond
    out = []
    for item in cond:
        if isinstance(item, dict):
            d = {k: (_crop_val(v, s, H, W) if k in _BX_CROP_KEYS else v) for k, v in item.items()}
            out.append(d)
        else:
            t = item[0]
            d0 = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
            d = {k: (_crop_val(v, s, H, W) if k in _BX_CROP_KEYS else v) for k, v in d0.items()}
            out.append([t, d])
    return out


if _samplers is not None and hasattr(_samplers, "CFGGuider"):

    class _BruxosLTXTiledGuider(_samplers.CFGGuider):
        """Prediz o ruido por ladrilho espacial (latente 5D do LTX) e FUNDE a
        cada passo de denoise. Timer + limpeza de VRAM opcionais por ladrilho."""

        def set_tiling(self, rows, cols, overlap, cleanup, debug):
            self._rows, self._cols = int(rows), int(cols)
            self._ovl = int(overlap)
            self._cleanup = bool(cleanup)
            self._debug = bool(debug)
            self._logged = False
            self._run_t0 = None
            self._step_i = 0

        def predict_noise(self, x, timestep, model_options={}, seed=None):
            rows = getattr(self, "_rows", 1)
            cols = getattr(self, "_cols", 1)
            if (rows <= 1 and cols <= 1) or x.dim() < 5:
                return super().predict_noise(x, timestep, model_options, seed)

            if self._run_t0 is None:
                self._run_t0 = time.time()

            # latente LTX e 5D: [batch, canais, frames, H, W] -- ladrilha nas
            # duas ultimas dims (espacial), igual ao Wan.
            H, W = int(x.shape[-2]), int(x.shape[-1])
            specs = _tile_plan(H, W, rows, cols, self._ovl)
            if len(specs) <= 1:
                return super().predict_noise(x, timestep, model_options, seed)

            if not self._logged:
                self._logged = True
                th = specs[0]["y1"] - specs[0]["y0"]
                tw = specs[0]["x1"] - specs[0]["x0"]
                print(f"[Bruxos LTX Tiled] latente {W}x{H} -> {len(specs)} ladrilho(s) "
                      f"de {tw}x{th} (overlap {self._ovl}) | fusao a cada passo", flush=True)

            step_t0 = time.time()
            conds_full = self.conds
            acc = torch.zeros_like(x, dtype=torch.float32)
            wsum = torch.zeros((1,) * (x.dim() - 2) + (H, W), device=x.device, dtype=torch.float32)

            try:
                for s in specs:
                    xt = x[..., s["y0"]:s["y1"], s["x0"]:s["x1"]].contiguous()
                    self.conds = {k: _crop_cond_list(v, s, H, W) for k, v in conds_full.items()}
                    pred = super().predict_noise(xt, timestep, model_options, seed)

                    win = _win2d(s, x.device, torch.float32)
                    acc[..., s["y0"]:s["y1"], s["x0"]:s["x1"]] += pred.float() * win
                    wsum[..., s["y0"]:s["y1"], s["x0"]:s["x1"]] += win

                    del xt, pred, win
                    if self._cleanup:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            finally:
                self.conds = conds_full

            wmin = float(wsum.min())
            if wmin <= 1e-7:
                raise RuntimeError(
                    f"[Bruxos LTX Tiled] os ladrilhos nao cobriram o latente inteiro "
                    f"(peso minimo {wmin}). Reduza o numero de ladrilhos ou o overlap."
                )
            out = (acc / wsum.clamp(min=1e-8)).to(x.dtype)

            # ---- TIMER: 1 linha por passo + total acumulado do run inteiro ----
            self._step_i += 1
            if self._debug:
                dt_step = time.time() - step_t0
                dt_total = time.time() - self._run_t0
                print(f"[Bruxos LTX Tiled][timer] passo {self._step_i}: {dt_step:.2f}s "
                      f"({len(specs)} ladrilho(s)) | acumulado {dt_total:.1f}s", flush=True)
            return out


class BruxosLTXTiledGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Modelo LTX (o mesmo que iria pro sampler)."}),
                "positive": ("CONDITIONING", {"tooltip": "Positivo."}),
                "negative": ("CONDITIONING", {"tooltip": "Negativo."}),
                "tile_count_width": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Colunas de ladrilho (no latente). 1 = nao corta na horizontal."}),
                "tile_count_height": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Linhas de ladrilho. 2x2 = 4 pedacos. 1x1 desliga o ladrilho."}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "CFG do LTX (nao e cfg-destilado por padrao, diferente do Wan+LightX2V)."}),
            },
            "optional": {
                "overlap": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Sobreposicao entre ladrilhos, em unidades de LATENTE. Maior = costura mais suave, mais VRAM."}),
                "limpar_vram": ("BOOLEAN", {"default": True,
                    "tooltip": "Esvazia o cache da VRAM depois de CADA ladrilho (por passo). Deixe ligado."}),
                "timer": ("BOOLEAN", {"default": True,
                    "tooltip": "Imprime no console o tempo de cada passo e o acumulado do run. Desligue se o log ficar poluido."}),
            },
        }

    RETURN_TYPES = ("GUIDER", "STRING")
    RETURN_NAMES = ("guider", "info")
    OUTPUT_TOOLTIPS = (
        "Ligue no SamplerCustomAdvanced (entrada 'guider'). O ladrilho acontece DENTRO do sampler, a cada passo.",
        "Resumo do plano de ladrilhos.",
    )
    FUNCTION = "build"
    CATEGORY = CAT
    DESCRIPTION = (
        "LTX Tiled Sampler (Bruxos) — step-fused: corta o latente do LTX em ladrilhos espaciais e "
        "funde as predicoes de ruido a cada passo (janela complementar, sem emenda visivel por "
        "construcao). Diferente da LTXVTiledSampler oficial da Lightricks (que roda cada ladrilho "
        "ATE O FIM antes de fundir): aqui os ladrilhos se enxergam o tempo todo, entao tende a sair "
        "mais coerente, ao custo de mais forwards por passo. NAO reproduz o guia de imagem por "
        "ladrilho (LTXVAddGuide) da versao oficial -- prefira ela pra fluxos com guia de imagem "
        "pesado por ladrilho. Com timer de progresso e limpeza de VRAM por ladrilho."
    )

    def build(self, model, positive, negative, tile_count_width, tile_count_height, cfg,
              overlap=8, limpar_vram=True, timer=True):
        if not _HAS_TORCH:
            raise RuntimeError("[Bruxos LTX Tiled] torch indisponivel.")
        if _samplers is None or not hasattr(_samplers, "CFGGuider"):
            raise RuntimeError("[Bruxos LTX Tiled] comfy.samplers.CFGGuider nao encontrado neste build.")

        _bx_log_attention_backend_once()

        g = _BruxosLTXTiledGuider(model)
        g.set_conds(positive, negative)
        try:
            g.set_cfg(float(cfg))
        except Exception:
            pass
        g.set_tiling(int(tile_count_height), int(tile_count_width), int(overlap),
                     bool(limpar_vram), bool(timer))
        # a LTXVTiledSampler oficial (Lightricks) espera guider.raw_conds pronto
        # se algum outro node tentar reusar este guider naquele node tambem.
        g.raw_conds = (positive, negative)

        n = int(tile_count_width) * int(tile_count_height)
        if n <= 1:
            info = "1x1 -> ladrilho DESLIGADO (roda o quadro inteiro, normal)"
        else:
            info = (f"{tile_count_width}x{tile_count_height} = {n} ladrilho(s) | overlap {overlap} "
                    f"latentes | cfg {cfg} | fusao a cada passo | timer={'on' if timer else 'off'}")
        print(f"[Bruxos LTX Tiled] {info}", flush=True)
        return (g, info)


NODE_CLASS_MAPPINGS = {"BruxosLTXTiledGuider": BruxosLTXTiledGuider}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosLTXTiledGuider": "LTX Tiled Sampler (Bruxos)"}
