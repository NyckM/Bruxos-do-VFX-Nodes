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
    import torch.nn.functional as F
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


def _attach_custom_fades(specs):
    """Calcula fades por borda para retangulos arbitrarios sobrepostos."""
    for a in specs:
        a["ft"] = a["fb"] = a["fl"] = a["fr"] = 0
        acx = (a["x0"] + a["x1"]) * 0.5
        acy = (a["y0"] + a["y1"]) * 0.5
        for b in specs:
            if a is b:
                continue
            ix0, iy0 = max(a["x0"], b["x0"]), max(a["y0"], b["y0"])
            ix1, iy1 = min(a["x1"], b["x1"]), min(a["y1"], b["y1"])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            bcx = (b["x0"] + b["x1"]) * 0.5
            bcy = (b["y0"] + b["y1"]) * 0.5
            if bcx < acx:
                a["fl"] = max(a["fl"], ix1 - a["x0"])
            elif bcx > acx:
                a["fr"] = max(a["fr"], a["x1"] - ix0)
            if bcy < acy:
                a["ft"] = max(a["ft"], iy1 - a["y0"])
            elif bcy > acy:
                a["fb"] = max(a["fb"], a["y1"] - iy0)
    return specs


def _custom_tile_plan(H, W, layout, overlap):
    """Converte retangulos normalizados do editor para o grid latente."""
    raw = layout.get("tiles", []) if isinstance(layout, dict) else []
    if not raw:
        raise ValueError("layout custom sem tiles")
    specs = []
    ov = max(0, int(overlap))
    for index, tile in enumerate(raw[:24]):
        x0 = int(math.floor(float(tile["x0"]) * W))
        y0 = int(math.floor(float(tile["y0"]) * H))
        x1 = int(math.ceil(float(tile["x1"]) * W))
        y1 = int(math.ceil(float(tile["y1"]) * H))
        # O overlap do node expande apenas bordas internas. Assim layouts que
        # se encostam ganham uma faixa real de fusao sem ultrapassar o canvas.
        if x0 > 0: x0 -= ov
        if y0 > 0: y0 -= ov
        if x1 < W: x1 += ov
        if y1 < H: y1 += ov
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        specs.append({
            "row": index, "col": 0, "id": tile.get("id", index + 1),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "weight": max(0.05, min(8.0, float(tile.get("weight", 1.0)))),
        })
    if not specs:
        raise ValueError("layout custom nao produziu tiles validos")
    cover = torch.zeros((H, W), dtype=torch.bool)
    for s in specs:
        cover[s["y0"]:s["y1"], s["x0"]:s["x1"]] = True
    if not bool(cover.all()):
        missing = int((~cover).sum())
        raise ValueError(f"layout custom deixou {missing} celula(s) latentes sem cobertura")
    return _attach_custom_fades(specs)


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
# O contexto 0 do Bernini e o video-fonte espacialmente alinhado ao latent de
# geracao. Ele acompanha o recorte de cada tile. Os contextos posteriores
# (tail/reference_video/imagens) seguem global, local ou hybrid.
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


def _reference_tile(t, s, H, W, mode):
    """Contexto de referencia local ou hibrido, sem criar um stream extra.

    O hibrido mistura o recorte detalhado com uma miniatura low-frequency da
    referencia inteira. Assim preserva composicao/identidade global sem mudar a
    quantidade/ordem dos context_latents (image0 continua sendo image0).
    """
    if not torch.is_tensor(t) or t.dim() < 4 or mode == "global":
        return t
    rh, rw = int(t.shape[-2]), int(t.shape[-1])
    x0 = max(0, min(rw - 1, int(math.floor(s["x0"] * rw / float(W)))))
    y0 = max(0, min(rh - 1, int(math.floor(s["y0"] * rh / float(H)))))
    x1 = max(x0 + 1, min(rw, int(math.ceil(s["x1"] * rw / float(W)))))
    y1 = max(y0 + 1, min(rh, int(math.ceil(s["y1"] * rh / float(H)))))
    local = t[..., y0:y1, x0:x1].contiguous()
    if mode != "hybrid":
        return local

    # Ancora global barata: reduz primeiro a no maximo 16x16 latentes para
    # remover detalhe fino, depois encaixa no tile e mistura 20%.
    lh, lw = int(local.shape[-2]), int(local.shape[-1])
    prefix = tuple(t.shape[:-2])
    flat = t.reshape(-1, 1, rh, rw).float()
    scale = min(1.0, 16.0 / max(1, rh), 16.0 / max(1, rw))
    ah, aw = max(1, round(rh * scale)), max(1, round(rw * scale))
    anchor = F.interpolate(flat, size=(ah, aw), mode="area")
    anchor = F.interpolate(anchor, size=(lh, lw), mode="bilinear", align_corners=False)
    anchor = anchor.reshape(*prefix, lh, lw).to(dtype=local.dtype)
    return (local * 0.80 + anchor * 0.20).contiguous()


def _crop_contexts(value, s, H, W, mode="global"):
    """Fonte primaria sempre local; referencias seguem global/local/hybrid."""
    if not isinstance(value, (list, tuple)) or not value:
        return value
    out = list(value)
    first = out[0]
    if torch.is_tensor(first) and first.dim() >= 4 and tuple(first.shape[-2:]) == (H, W):
        out[0] = first[..., s["y0"]:s["y1"], s["x0"]:s["x1"]].contiguous()
    if mode in ("local", "hybrid"):
        for i in range(1, len(out)):
            out[i] = _reference_tile(out[i], s, H, W, mode)
    return out


def _primary_context_status(conds, H, W):
    """Retorna (tem_contexto, contexto_0_alinhado_ao_canvas)."""
    for item in conds or []:
        d = item if isinstance(item, dict) else (
            item[1] if isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], dict) else {}
        )
        ctx = d.get("context_latents")
        if ctx is None or (isinstance(ctx, (list, tuple)) and not ctx):
            continue
        first = ctx[0] if isinstance(ctx, (list, tuple)) else ctx
        matched = bool(
            torch.is_tensor(first) and first.dim() >= 4
            and tuple(first.shape[-2:]) == (int(H), int(W))
        )
        return True, matched
    return False, False


def _crop_cond_list(cond, s, H, W, context_mode="global"):
    """Recorta APENAS as chaves espaciais coladas (mask etc.); referencia
    (context_latents e afins) passa INTACTA — o modelo precisa ve-la inteira."""
    if not cond:
        return cond
    out = []
    for item in cond:
        if isinstance(item, dict):
            d = {}
            for k, v in item.items():
                if k == "context_latents":
                    d[k] = _crop_contexts(v, s, H, W, context_mode)
                else:
                    d[k] = _crop_val(v, s, H, W) if k in _BX_CROP_KEYS else v
            out.append(d)
        else:
            t = item[0]
            d0 = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
            d = {}
            for k, v in d0.items():
                if k == "context_latents":
                    d[k] = _crop_contexts(v, s, H, W, context_mode)
                else:
                    d[k] = _crop_val(v, s, H, W) if k in _BX_CROP_KEYS else v
            out.append([t, d])
    return out


if _samplers is not None and hasattr(_samplers, "CFGGuider"):

    class _WanTiledGuider(_samplers.CFGGuider):
        """Prediz o ruido por ladrilho e FUNDE a cada passo de denoise."""

        def set_tiling(self, rows, cols, overlap, blend, cleanup, debug, layout=None,
                       context_mode="global"):
            self._rows, self._cols = int(rows), int(cols)
            self._ovl = int(overlap)
            self._blend = blend
            self._cleanup = bool(cleanup)
            self._debug = bool(debug)
            self._layout = layout if isinstance(layout, dict) and layout.get("tiles") else None
            self._context_mode = str(context_mode) if str(context_mode) in ("global", "local", "hybrid") else "global"
            self._logged = False
            self._ctx_warned = False

        def predict_noise(self, x, timestep, model_options={}, seed=None):
            rows = getattr(self, "_rows", 1)
            cols = getattr(self, "_cols", 1)
            # 1x1 ou latente 4D -> caminho normal, sem ladrilho
            if (rows <= 1 and cols <= 1) or x.dim() < 4:
                return super().predict_noise(x, timestep, model_options, seed)

            # O target e o contexto-fonte precisam usar o mesmo recorte local.
            # Se o contexto 0 nao casar com o canvas, desliga o tiled com
            # seguranca em vez de produzir um mosaico de copias.
            H, W = int(x.shape[-2]), int(x.shape[-1])
            pos = self.conds.get("positive") or []
            has_ctx, primary_matches = _primary_context_status(pos, H, W)
            if has_ctx and not primary_matches:
                if not getattr(self, "_ctx_warned", False):
                    self._ctx_warned = True
                    print("[Bruxos Wan Tiled] context_latents detectado (video-fonte/refs): "
                          "o contexto 0 nao possui o mesmo grid espacial do latent de geracao. "
                          "Rodando SEM ladrilho para evitar quadrantes repetidos.",
                          flush=True)
                return super().predict_noise(x, timestep, model_options, seed)

            custom = getattr(self, "_layout", None)
            specs = _custom_tile_plan(H, W, custom, self._ovl) if custom else _tile_plan(H, W, rows, cols, self._ovl)
            if len(specs) <= 1:
                return super().predict_noise(x, timestep, model_options, seed)

            if not self._logged:
                self._logged = True
                th = specs[0]["y1"] - specs[0]["y0"]
                tw = specs[0]["x1"] - specs[0]["x0"]
                context_info = (f" | fonte context[0] local; tail/refs {self._context_mode}"
                                if has_ctx else "")
                layout_name = "custom" if custom else f"{cols}x{rows}"
                print(f"[Bruxos Wan Tiled] latente {W}x{H} -> {len(specs)} ladrilho(s) [{layout_name}] "
                      f"de {tw}x{th} (overlap {self._ovl}) | fusao a cada passo{context_info}",
                      flush=True)
                if has_ctx:
                    print(f"[Bruxos Wan Tiled] tile_context_mode={self._context_mode} "
                          f"(fonte local; refs {'inteiras' if self._context_mode == 'global' else 'por tile'})",
                          flush=True)

            conds_full = self.conds                       # guarda o original
            acc = torch.zeros_like(x, dtype=torch.float32)
            wsum = torch.zeros((1,) * (x.dim() - 2) + (H, W), device=x.device, dtype=torch.float32)

            try:
                for s in specs:
                    xt = x[..., s["y0"]:s["y1"], s["x0"]:s["x1"]].contiguous()
                    # o conditioning TEM que ver o mesmo pedaco
                    self.conds = {k: _crop_cond_list(v, s, H, W, self._context_mode)
                                  for k, v in conds_full.items()}
                    pred = super().predict_noise(xt, timestep, model_options, seed)

                    win = _win2d(s, x.device, torch.float32, self._blend)
                    win = win * float(s.get("weight", 1.0))
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
                "tile_layout": ("BRUXOS_TILE_LAYOUT", {"tooltip":
                    "Layout livre vindo do Bernini Custom Tile Layout. Substitui tile_count_width/height."}),
                "tile_context_mode": (["hybrid", "local", "global"], {"default": "hybrid", "tooltip":
                    "Como tratar tail/referencias no tiled. global=inteiras em todo tile (mais lento); "
                    "local=recorte proporcional; hybrid=recorte local + 20% de ancora global reduzida."}),
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
              overlap=8, blend_mode="hann", limpar_vram=True, debug=False, tile_layout=None,
              tile_context_mode="hybrid"):
        if _samplers is None or not hasattr(_samplers, "CFGGuider"):
            raise RuntimeError("[Bruxos Wan Tiled] comfy.samplers.CFGGuider nao encontrado neste build.")

        g = _WanTiledGuider(model)
        g.set_conds(positive, negative)
        g.set_cfg(float(cfg))
        g.set_tiling(int(tile_count_height), int(tile_count_width), int(overlap),
                     str(blend_mode), bool(limpar_vram), bool(debug), tile_layout,
                     tile_context_mode)

        n = len(tile_layout.get("tiles", [])) if isinstance(tile_layout, dict) else int(tile_count_width) * int(tile_count_height)
        if n <= 1:
            info = "1x1 -> ladrilho DESLIGADO (roda o quadro inteiro, normal)"
        else:
            label = "custom" if isinstance(tile_layout, dict) else f"{tile_count_width}x{tile_count_height}"
            info = (f"{label} = {n} ladrilho(s) | overlap {overlap} "
                    f"latentes (~{overlap * 8}px) | {blend_mode} | cfg {cfg} | "
                    f"contexto {tile_context_mode} | fusao a cada passo")
        print(f"[Bruxos Wan Tiled] {info}", flush=True)
        return (g, info)


NODE_CLASS_MAPPINGS = {"BruxosWanTiledGuider": BruxosWanTiledGuider}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosWanTiledGuider": "Wan Tiled Sampler (Bruxos)"}
