# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Wan Tiled Sampler (step-fused)
==============================================
UM node. SEM For Loop. SEM emenda. SEM drift.

Ideia (a mesma do "step-fused tiled sampler" do Deno pro LTX, adaptada pro Wan):
ao inves de cortar a imagem e rodar o sampler INTEIRO em cada ladrilho (o que faz
cada ladrilho "inventar" coisas diferentes -> cor/conteudo divergem na emenda),
a gente corta no LATENTE e FUNDE os ladrilhos A CADA PASSO DE DENOISE:

    para cada step do sampler:
        para cada ladrilho:
            prediz o ruido so daquele pedaco (com o conditioning tambem recortado)
        funde todas as predicoes numa unica, com janela Hann (complementar)
    -> o sampler continua normal, achando que rodou o quadro inteiro

Resultado: os ladrilhos "se enxergam" a cada passo, entao a imagem sai coerente e
a costura sao invisiveis por construcao (as janelas somam 1 na sobreposicao).
O custo de VRAM cai porque o modelo so ve um ladrilho por vez.

Saida: GUIDER -> ligue no SamplerCustomAdvanced (no lugar do guider normal).

Credito: a arquitetura "step-fused" e a janela complementar seguem o
comfyui-deno-custom-nodes (Deno2026), feito originalmente pro LTX.
"""

import gc
import math
import logging

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


# ---------------------------------------------------------------------------
# plano dos ladrilhos (em coordenadas de LATENTE)
# ---------------------------------------------------------------------------
def _axis_plan(total, count, overlap):
    """Divide um eixo de tamanho `total` em `count` pedacos com `overlap`.
    Devolve (inicios, tamanho_do_pedaco)."""
    total, count, overlap = int(total), max(1, int(count)), max(0, int(overlap))
    if count == 1 or total <= 1:
        return [0], total
    size = math.ceil((total + (count - 1) * overlap) / count)
    size = min(size, total)
    if overlap >= size:
        overlap = max(0, size - 1)
    travel = total - size
    starts = [int(round(i * travel / (count - 1))) for i in range(count)]
    starts[0], starts[-1] = 0, travel
    # remove inicios duplicados (acontece se pedir tiles demais p/ um eixo curto)
    uniq = sorted(set(starts))
    return uniq, size


def _tile_plan(H, W, rows, cols, overlap):
    """Lista de ladrilhos cobrindo TODO o latente, com fades por borda interna."""
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


def _win1d(size, fade_a, fade_b, device, dtype, mode="hann"):
    """Janela 1D com subida/descida COMPLEMENTAR: duas janelas vizinhas somam 1
    exatamente na sobreposicao -> emenda invisivel por construcao."""
    w = torch.ones(size, device=device, dtype=dtype)
    fa = min(max(int(fade_a), 0), size)
    fb = min(max(int(fade_b), 0), size)
    if fa:
        i = torch.arange(fa, device=device, dtype=dtype)
        w[:fa] = 0.5 * (1.0 - torch.cos(math.pi * i / fa))       # sobe
    if fb:
        i = torch.arange(fb, device=device, dtype=dtype)
        w[-fb:] = 0.5 * (1.0 + torch.cos(math.pi * i / fb))      # desce
    return w


def _win2d(spec, device, dtype, mode="hann"):
    h = spec["y1"] - spec["y0"]
    w = spec["x1"] - spec["x0"]
    vy = _win1d(h, spec["ft"], spec["fb"], device, dtype, mode)
    vx = _win1d(w, spec["fl"], spec["fr"], device, dtype, mode)
    return (vy[:, None] * vx[None, :])


# ---------------------------------------------------------------------------
# recorte do conditioning
# IMPORTANTE (bug corrigido): o context_latents do Bernini/Wan NAO pode ser
# recortado. Ele e um stream de REFERENCIA com posicoes proprias (RoPE): se o
# modelo recebe so um pedaco, ele o trata como o quadro INTEIRO de referencia
# e reproduz o shot todo dentro de cada ladrilho (mosaico de copias). O certo
# e: cada ladrilho gera o seu pedaco VENDO a referencia inteira.
# So recortamos tensores que sao ESPACIALMENTE COLADOS ao latente de geracao
# (mask/noise_mask/concat_latent_image), nunca os streams de referencia.
# ---------------------------------------------------------------------------
_BX_NEVER_CROP = {"context_latents", "reference_latents", "pooled_output"}
_BX_CROP_KEYS = {"mask", "noise_mask", "concat_latent_image", "concat_mask", "denoise_mask"}


def _crop_t(t, s, H, W):
    """Recorta um tensor SE as duas ultimas dims baterem com o latente cheio."""
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
    """Recorta APENAS as chaves espaciais coladas (mask etc.); referencia
    (context_latents e afins) passa INTACTA — o modelo precisa ve-la inteira."""
    if not cond:
        return cond
    out = []
    for item in cond:
        if isinstance(item, dict):
            d = {}
            for k, v in item.items():
                d[k] = _crop_val(v, s, H, W) if k in _BX_CROP_KEYS else v
            out.append(d)
        else:
            t = item[0]
            d0 = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
            d = {}
            for k, v in d0.items():
                d[k] = _crop_val(v, s, H, W) if k in _BX_CROP_KEYS else v
            out.append([t, d])
    return out


if _samplers is not None and hasattr(_samplers, "CFGGuider"):

    class _WanTiledGuider(_samplers.CFGGuider):
        """Prediz o ruido por ladrilho e FUNDE a cada passo de denoise."""

        def set_tiling(self, rows, cols, overlap, blend, cleanup, debug):
            self._rows, self._cols = int(rows), int(cols)
            self._ovl = int(overlap)
            self._blend = blend
            self._cleanup = bool(cleanup)
            self._debug = bool(debug)
            self._logged = False

        def predict_noise(self, x, timestep, model_options={}, seed=None):
            rows = getattr(self, "_rows", 1)
            cols = getattr(self, "_cols", 1)
            # 1x1 ou latente 4D -> caminho normal, sem ladrilho
            if (rows <= 1 and cols <= 1) or x.dim() < 4:
                return super().predict_noise(x, timestep, model_options, seed)

            # LIMITACAO CONHECIDA (honesta): com context_latents (Bernini/Wan
            # V2V), o ladrilho espacial perde a POSICAO — o RoPE do pedaco
            # comeca em (0,0) e o modelo reproduz a referencia inteira dentro
            # de cada tile (mosaico de copias). Ate existir suporte a posicao
            # no motor, o tiled so vale p/ geracao SEM streams de referencia
            # (T2V puro). Detectamos e caimos pro caminho normal, avisando.
            pos = self.conds.get("positive") or []
            has_ctx = False
            for item in pos:
                d = item if isinstance(item, dict) else (item[1] if len(item) > 1 and isinstance(item[1], dict) else {})
                if d.get("context_latents"):
                    has_ctx = True
                    break
            if has_ctx:
                if not getattr(self, "_ctx_warned", False):
                    self._ctx_warned = True
                    print("[Bruxos Wan Tiled] context_latents detectado (video-fonte/refs): "
                          "o ladrilho espacial NAO preserva a posicao com streams de referencia "
                          "(sairia um mosaico de copias). Rodando SEM ladrilho neste passo. "
                          "P/ resolucao maior com V2V, use o modo bbox ou reducao de chunk.",
                          flush=True)
                return super().predict_noise(x, timestep, model_options, seed)

            H, W = int(x.shape[-2]), int(x.shape[-1])
            specs = _tile_plan(H, W, rows, cols, self._ovl)
            if len(specs) <= 1:
                return super().predict_noise(x, timestep, model_options, seed)

            if not self._logged:
                self._logged = True
                th = specs[0]["y1"] - specs[0]["y0"]
                tw = specs[0]["x1"] - specs[0]["x0"]
                print(f"[Bruxos Wan Tiled] latente {W}x{H} -> {len(specs)} ladrilho(s) "
                      f"de {tw}x{th} (overlap {self._ovl}) | fusao a cada passo",
                      flush=True)

            conds_full = self.conds                       # guarda o original
            acc = torch.zeros_like(x, dtype=torch.float32)
            wsum = torch.zeros((1,) * (x.dim() - 2) + (H, W), device=x.device, dtype=torch.float32)

            try:
                for s in specs:
                    xt = x[..., s["y0"]:s["y1"], s["x0"]:s["x1"]].contiguous()
                    # o conditioning TEM que ver o mesmo pedaco
                    self.conds = {k: _crop_cond_list(v, s, H, W) for k, v in conds_full.items()}
                    pred = super().predict_noise(xt, timestep, model_options, seed)

                    win = _win2d(s, x.device, torch.float32, self._blend)
                    acc[..., s["y0"]:s["y1"], s["x0"]:s["x1"]] += pred.float() * win
                    wsum[..., s["y0"]:s["y1"], s["x0"]:s["x1"]] += win

                    del xt, pred, win
                    if self._cleanup:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            finally:
                self.conds = conds_full                   # restaura SEMPRE

            wmin = float(wsum.min())
            if wmin <= 1e-7:
                raise RuntimeError(
                    f"[Bruxos Wan Tiled] os ladrilhos nao cobriram o latente inteiro "
                    f"(peso minimo {wmin}). Reduza o numero de ladrilhos ou o overlap."
                )
            return (acc / wsum.clamp(min=1e-8)).to(x.dtype)


class BruxosWanTiledGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "O modelo Wan/Bernini (o mesmo que iria pro sampler)."}),
                "positive": ("CONDITIONING", {"tooltip": "Positivo (pode vir do Bernini Conditioning, com context_latents -- eles sao recortados por ladrilho automaticamente)."}),
                "negative": ("CONDITIONING", {"tooltip": "Negativo."}),
                "tile_count_width": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Quantas COLUNAS de ladrilho. 1 = nao corta na horizontal."}),
                "tile_count_height": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Quantas LINHAS. 2x2 = 4 ladrilhos. 1x1 = desliga o ladrilho (roda normal)."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "CFG. Com LoRA LightX2V (cfg destilado) use 1.0 -- valores altos QUEIMAM."}),
            },
            "optional": {
                "overlap": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Sobreposicao entre ladrilhos, em unidades de LATENTE (1 = 8 pixels no Wan). 8 = 64px. Maior = costura mais suave e mais VRAM."}),
                "blend_mode": (["hann", "cosine"], {"default": "hann",
                    "tooltip": "Formato do degrade na sobreposicao. As duas janelas somam 1 exatamente -> emenda invisivel."}),
                "limpar_vram": ("BOOLEAN", {"default": True,
                    "tooltip": "Esvazia o cache da VRAM depois de CADA ladrilho. Deixe LIGADO -- e o que faz caber na memoria."}),
                "debug": ("BOOLEAN", {"default": False,
                    "tooltip": "Imprime o plano dos ladrilhos no console."}),
            },
        }

    RETURN_TYPES = ("GUIDER", "STRING")
    RETURN_NAMES = ("guider", "info")
    OUTPUT_TOOLTIPS = (
        "Ligue no SamplerCustomAdvanced (entrada 'guider'). O ladrilho acontece DENTRO do sampler.",
        "Resumo do plano de ladrilhos.",
    )
    FUNCTION = "build"
    CATEGORY = CAT
    DESCRIPTION = (
        "Wan Tiled Sampler (Bruxos) — roda o Wan em ladrilhos SEM For Loop e SEM emenda. "
        "Em vez de cortar a imagem e sampleiar cada pedaco separado (que faz cada pedaco divergir "
        "em cor/conteudo), ele corta no LATENTE e FUNDE os ladrilhos A CADA PASSO de denoise, com "
        "janela complementar. E UMA passada de sampler so: os ladrilhos se enxergam, a imagem sai "
        "coerente e a VRAM cai (o modelo so ve um pedaco por vez). "
        "Saida GUIDER -> SamplerCustomAdvanced. 1x1 desliga o ladrilho."
    )

    def build(self, model, positive, negative, tile_count_width, tile_count_height, cfg,
              overlap=8, blend_mode="hann", limpar_vram=True, debug=False):
        if _samplers is None or not hasattr(_samplers, "CFGGuider"):
            raise RuntimeError("[Bruxos Wan Tiled] comfy.samplers.CFGGuider nao encontrado neste build.")

        g = _WanTiledGuider(model)
        g.set_conds(positive, negative)
        g.set_cfg(float(cfg))
        g.set_tiling(int(tile_count_height), int(tile_count_width), int(overlap),
                     str(blend_mode), bool(limpar_vram), bool(debug))

        n = int(tile_count_width) * int(tile_count_height)
        if n <= 1:
            info = "1x1 -> ladrilho DESLIGADO (roda o quadro inteiro, normal)"
        else:
            info = (f"{tile_count_width}x{tile_count_height} = {n} ladrilho(s) | overlap {overlap} "
                    f"latentes (~{overlap * 8}px) | {blend_mode} | cfg {cfg} | fusao a cada passo")
        print(f"[Bruxos Wan Tiled] {info}", flush=True)
        return (g, info)


NODE_CLASS_MAPPINGS = {"BruxosWanTiledGuider": BruxosWanTiledGuider}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosWanTiledGuider": "Wan Tiled Sampler (Bruxos)"}
