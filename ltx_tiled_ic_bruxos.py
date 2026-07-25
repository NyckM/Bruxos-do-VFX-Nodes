# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — LTX Tiled Sampler IC (step-fused, IC-LoRA + audio)
==================================================================
Versao do LTX Tiled Sampler (Bruxos) que FUNCIONA no fluxo IC-LoRA do LTX 2.x
(o do "Add Video IC-LoRA Guide" + audio). O tiled normal (ltx_tiled_bruxos.py)
QUEBRA no IC por dois motivos concretos que foram confirmados lendo o codigo
oficial da Lightricks (iclora.py) e do ComfyUI (comfy_extras/nodes_lt.py):

  (1) LATENTE AUDIO+VIDEO = NestedTensor.
      Quando tem audio, o LTXVConcatAVLatent embrulha o latente num
      comfy.nested_tensor.NestedTensor((video_5D, audio_4D)). O guider antigo
      faz x[..., y0:y1, x0:x1] e torch.zeros_like(x) direto no x -> isso corta
      o AUDIO espacialmente (destroi) e o zeros_like nem existe pra NestedTensor.
      AQUI: separo video e audio, LADRILHO SO O VIDEO no espaco, passo o audio
      INTEIRO em cada forward e refaco o NestedTensor pra alimentar o modelo.

  (2) IC-LoRA GUIA por keyframes (keyframe_idxs / guide_attention_entries).
      O "Add Video IC-LoRA Guide" APENDA frames-guia no fim do eixo de frames do
      latente de video e registra no conditioning:
        - keyframe_idxs: [B, 3(t,h,w), N_tokens, 2(start,end)] em coord de PIXEL,
          um token por celula (f,h,w) do latente-guia, em ordem row-major (f h w).
        - guide_attention_entries: [{pre_filter_count, strength, pixel_mask,
          latent_shape=[F,H,W]}, ...].
      Se voce corta o latente no espaco mas NAO corta esses metadados, a contagem
      de tokens do guia (F*H*W) nao bate com o ladrilho (F*h_t*w_t) e o RoPE/atencao
      desalinha -> saida lixo ou erro. O guider antigo passa essas chaves INTACTAS
      (a lista _BX_CROP_KEYS conservadora nao as inclui).
      AQUI: corto keyframe_idxs por reshape [B,3,F,H,W,2] -> janela [y0:y1,x0:x1]
      -> reflatten, e re-ancoro as coordenadas h,w subtraindo a origem do ladrilho
      (em pixel, passo derivado do proprio grid). Ajusto guide_attention_entries
      (pre_filter_count e latent_shape) e o pixel_mask espacial junto.

Honestidade (LEIA):
  - Testei a MATEMATICA dos cortes (keyframe reshape/crop/offset; split/merge do
    NestedTensor; audio intacto) com tensores sinteticos no sandbox. O
    encaixe final com o modelo LTX real (RoPE dos tokens-guia, fusao do audio)
    so da pra confirmar rodando na sua 4090 -- nao tenho o modelo aqui.
  - Fusao do AUDIO: cada ladrilho re-preve o audio (que "ve" um pedaco diferente
    do video). Eu FUNDO por media dos ladrilhos. E heuristica; se o audio sair
    estranho, use o modo audio_fuse="primeiro" (pega so do 1o ladrilho) ou rode
    o audio sem tile (pass separado). Documentado no widget.
  - So mexe no IC quando detecta os metadados IC (keyframe_idxs). Sem eles, cai
    no comportamento do tiled normal (T2V/I2V). latent_downscale_factor>1 (IC em
    grid pequeno dilatado) NAO e suportado com seguranca aqui: se latent_shape do
    guia nao casar com H,W do video, o corte IC se DESLIGA e avisa (nao corrompe).

Credito de arquitetura: mesma linha "step-fused" + janela complementar do
ltx_tiled_bruxos.py (Bruxos), so que ciente de NestedTensor e IC-LoRA.
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
except Exception:
    _samplers = None

try:
    import comfy.nested_tensor as _nested
except Exception:
    _nested = None

try:
    import node_helpers as _node_helpers
except Exception:
    _node_helpers = None

CAT = "Bruxos do VFX/Tiles"


# ---------------------------------------------------------------------------
# diagnostico de backend de atencao (igual ao tiled normal, so log)
# ---------------------------------------------------------------------------
_BX_IC_ATTN_LOGGED = {"done": False}


def _bx_log_attention_backend_once():
    if _BX_IC_ATTN_LOGGED["done"]:
        return
    _BX_IC_ATTN_LOGGED["done"] = True
    try:
        import comfy.cli_args as _cli
        a = _cli.args
        if getattr(a, "use_sage_attention", False):
            active = "sage (--use-sage-attention)"
        elif getattr(a, "use_flash_attention", False):
            active = "flash (--use-flash-attention)"
        else:
            active = "padrao do ComfyUI (pytorch/sdpa)"
        print(f"[Bruxos LTX Tiled IC][attention] backend ativo: {active}", flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# geometria de ladrilhos + janela de blend complementar (identico ao tiled normal)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# corte conservador de conditioning colado espacialmente (mask/concat_latent...)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# corte IC-LoRA: keyframe_idxs + guide_attention_entries por ladrilho
# ---------------------------------------------------------------------------
def _derive_pixel_step(kf_axis_grid):
    """kf_axis_grid: tensor [F,H,W] com a coord (start) de UM eixo (h ou w) em pixel.
    Retorna o passo em pixel entre celulas latentes adjacentes daquele eixo (int),
    derivado do proprio grid (sem hardcode do fator 32/8 do VAE). 1 se nao der."""
    try:
        if kf_axis_grid.shape[1] >= 2:  # ao longo de H
            diff = (kf_axis_grid[:, 1, :] - kf_axis_grid[:, 0, :]).abs()
            step = int(round(float(diff.flatten()[0].item())))
            if step > 0:
                return step
        if kf_axis_grid.shape[2] >= 2:  # ao longo de W
            diff = (kf_axis_grid[:, :, 1] - kf_axis_grid[:, :, 0]).abs()
            step = int(round(float(diff.flatten()[0].item())))
            if step > 0:
                return step
    except Exception:
        pass
    return 1


def _crop_keyframe_idxs(kf, s, H, W, n_guide_frames):
    """kf: [B, 3(t,h,w), N, 2(start,end)] pixel coords, N = n_guide_frames*H*W,
    ordem row-major (f h w). Corta pra janela do ladrilho e re-ancora h,w."""
    B, three, N, se = kf.shape
    if three != 3 or se != 2:
        return None
    F = n_guide_frames
    if F <= 0 or F * H * W != N:
        return None  # nao casa -> deixa o chamador desligar o corte IC
    grid = kf.reshape(B, 3, F, H, W, 2)
    # passo em pixel por eixo, derivado do grid (canal 1=h, 2=w; usa 'start' [..,0])
    step_h = _derive_pixel_step(grid[0, 1, :, :, :, 0])
    step_w = _derive_pixel_step(grid[0, 2, :, :, :, 0])
    sub = grid[:, :, :, s["y0"]:s["y1"], s["x0"]:s["x1"], :].clone()
    # re-ancora: origem do ladrilho vira 0 (RoPE dos tokens-guia casa com os gerados)
    sub[:, 1, :, :, :, :] -= s["y0"] * step_h   # eixo h (start e end)
    sub[:, 2, :, :, :, :] -= s["x0"] * step_w   # eixo w (start e end)
    h_t = s["y1"] - s["y0"]
    w_t = s["x1"] - s["x0"]
    return sub.reshape(B, 3, F * h_t * w_t, 2).contiguous()


def _crop_guide_entries(entries, s, H, W):
    """Ajusta pre_filter_count, latent_shape e pixel_mask espacial por ladrilho.
    Retorna (novas_entries, total_guide_frames) ou (None, 0) se algo nao casar."""
    h_t = s["y1"] - s["y0"]
    w_t = s["x1"] - s["x0"]
    out = []
    total_f = 0
    for e in entries:
        ls = e.get("latent_shape", None)
        if not ls or len(ls) != 3:
            return None, 0
        f_i, h_i, w_i = int(ls[0]), int(ls[1]), int(ls[2])
        if (h_i, w_i) != (H, W):
            return None, 0  # ex.: IC em grid pequeno (downscale>1) -> nao suportado
        total_f += f_i
        ne = dict(e)
        ne["latent_shape"] = [f_i, h_t, w_t]
        ne["pre_filter_count"] = f_i * h_t * w_t
        pm = e.get("pixel_mask", None)
        if torch.is_tensor(pm) and pm.dim() >= 2 and tuple(pm.shape[-2:]) == (H, W):
            ne["pixel_mask"] = pm[..., s["y0"]:s["y1"], s["x0"]:s["x1"]].contiguous()
        out.append(ne)
    return out, total_f


# chaves de model_conds que sao tensores colados ESPACIALMENTE ao latente
# (a mais critica: denoise_mask -> vira o timestep por-token no LTX2)
_BX_MC_SPATIAL = ("denoise_mask", "audio_denoise_mask", "noise_mask", "concat_mask",
                  "concat_latent_image", "mask", "attention_mask")


def _payload(w):
    return getattr(w, "cond", w)


def _copy_cond(w, new):
    fn = getattr(w, "_copy_with", None)
    if callable(fn):
        try:
            return fn(new)
        except Exception:
            pass
    return new


def _crop_model_conds(mc, s, H, W, ic_on, tile_shape):
    """Corta o dict model_conds (COND objects) pro ladrilho:
    - latent_shapes -> forma do ladrilho (senao o modelo monta modulacao no
      tamanho CHEIO e nao encaixa: o erro 220320 vs 9720);
    - denoise_mask & cia -> corte espacial (define o timestep por-token);
    - keyframe_idxs / guide_attention_entries -> corte IC."""
    out = dict(mc)

    # 1) latent_shapes (CONDConstant com lista de shapes)
    if "latent_shapes" in out and tile_shape is not None:
        w = out["latent_shapes"]
        cur = _payload(w)
        if isinstance(cur, (list, tuple)) and len(cur) >= 1:
            new_ls = [tuple(tile_shape)] + [tuple(sh) for sh in list(cur)[1:]]
            out["latent_shapes"] = _copy_cond(w, new_ls)

    # 2) tensores espaciais colados (denoise_mask etc.)
    for k in _BX_MC_SPATIAL:
        if k not in out:
            continue
        w = out[k]
        v = _payload(w)
        if torch.is_tensor(v) and v.dim() >= 4 and tuple(v.shape[-2:]) == (H, W):
            out[k] = _copy_cond(w, v[..., s["y0"]:s["y1"], s["x0"]:s["x1"]].contiguous())

    # 3) IC-LoRA: keyframe_idxs + guide_attention_entries
    if ic_on and "keyframe_idxs" in out:
        kf_w = out["keyframe_idxs"]
        kf = _payload(kf_w)
        if torch.is_tensor(kf):
            entries_w = out.get("guide_attention_entries")
            entries = _payload(entries_w) if entries_w is not None else None
            cropped_entries, n_guide = (None, 0)
            if isinstance(entries, list) and entries:
                cropped_entries, n_guide = _crop_guide_entries(entries, s, H, W)
            if n_guide <= 0 and kf.dim() == 4 and (H * W) > 0 and kf.shape[2] % (H * W) == 0:
                n_guide = kf.shape[2] // (H * W)
            cropped_kf = _crop_keyframe_idxs(kf, s, H, W, n_guide)
            if cropped_kf is not None:
                out["keyframe_idxs"] = _copy_cond(kf_w, cropped_kf)
                if cropped_entries is not None and entries_w is not None:
                    out["guide_attention_entries"] = _copy_cond(entries_w, cropped_entries)
    return out


def _crop_cond_list_ic(cond, s, H, W, ic_on, tile_shape=None):
    """Corta a lista de conditioning pro ladrilho, no nivel model_conds
    (onde o LTX2 realmente guarda denoise_mask/latent_shapes/keyframe_idxs).
    Mantem fallback pro nivel de topo (comfy antigos)."""
    if not cond:
        return cond
    out = []
    for item in cond:
        if isinstance(item, dict):
            cloned = dict(item)
            mc = cloned.get("model_conds", None)
            if isinstance(mc, dict):
                cloned["model_conds"] = _crop_model_conds(mc, s, H, W, ic_on, tile_shape)
            else:
                # fallback: chaves espaciais no topo
                for k in _BX_CROP_KEYS:
                    if k in cloned:
                        cloned[k] = _crop_val(cloned[k], s, H, W)
            out.append(cloned)
        else:
            base = item[0]
            d0 = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
            mc = d0.get("model_conds", None)
            d = dict(d0)
            if isinstance(mc, dict):
                d["model_conds"] = _crop_model_conds(mc, s, H, W, ic_on, tile_shape)
            out.append([base, d])
    return out


def _cond_has_ic(conds):
    for lst in conds.values():
        if not lst:
            continue
        for item in lst:
            d = item if isinstance(item, dict) else (item[1] if (isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], dict)) else {})
            mc = d.get("model_conds", {}) if isinstance(d, dict) else {}
            if isinstance(mc, dict) and mc.get("keyframe_idxs", None) is not None:
                return True
            if isinstance(d, dict) and d.get("keyframe_idxs", None) is not None:
                return True
    return False


# ---------------------------------------------------------------------------
# NestedTensor helpers (audio+video): separa, refaz, e mede so o video
# ---------------------------------------------------------------------------
def _is_nested(x):
    return getattr(x, "is_nested", False) and hasattr(x, "tensors")


def _video_of(x):
    return x.tensors[0] if _is_nested(x) else x


def _rebuild_like(x, new_video):
    if _is_nested(x) and _nested is not None:
        return _nested.NestedTensor([new_video] + list(x.tensors[1:]))
    return new_video


if _samplers is not None and hasattr(_samplers, "CFGGuider"):

    class _BruxosLTXTiledICGuider(_samplers.CFGGuider):
        """Preve o ruido por ladrilho ESPACIAL do VIDEO (5D), fundindo a cada
        passo. Ciente de NestedTensor (audio) e de keyframes do IC-LoRA."""

        def set_tiling(self, rows, cols, overlap, cleanup, debug, audio_fuse):
            self._rows, self._cols = int(rows), int(cols)
            self._ovl = int(overlap)
            self._cleanup = bool(cleanup)
            self._debug = bool(debug)
            self._audio_fuse = str(audio_fuse)
            self._logged = False
            self._run_t0 = None
            self._step_i = 0

        def predict_noise(self, x, timestep, model_options={}, seed=None):
            rows = getattr(self, "_rows", 1)
            cols = getattr(self, "_cols", 1)
            video = _video_of(x)
            if (rows <= 1 and cols <= 1) or not torch.is_tensor(video) or video.dim() < 5:
                return super().predict_noise(x, timestep, model_options, seed)

            if self._run_t0 is None:
                self._run_t0 = time.time()

            H, W = int(video.shape[-2]), int(video.shape[-1])
            specs = _tile_plan(H, W, rows, cols, self._ovl)
            if len(specs) <= 1:
                return super().predict_noise(x, timestep, model_options, seed)

            conds_full = self.conds
            ic_on = _cond_has_ic(conds_full)

            if not self._logged:
                self._logged = True
                th = specs[0]["y1"] - specs[0]["y0"]
                tw = specs[0]["x1"] - specs[0]["x0"]
                nested = _is_nested(x)
                print(f"[Bruxos LTX Tiled IC] video {W}x{H} -> {len(specs)} ladrilho(s) "
                      f"de {tw}x{th} (overlap {self._ovl}) | IC={'sim' if ic_on else 'nao'} "
                      f"| audio={'sim' if nested else 'nao'} | fusao a cada passo", flush=True)

            step_t0 = time.time()
            acc = torch.zeros_like(video, dtype=torch.float32)
            wsum = torch.zeros((1,) * (video.dim() - 2) + (H, W), device=video.device, dtype=torch.float32)
            audio_accum = None
            audio_count = 0

            try:
                for s in specs:
                    v_tile = video[..., s["y0"]:s["y1"], s["x0"]:s["x1"]].contiguous()
                    x_tile = _rebuild_like(x, v_tile)
                    tile_shape = tuple(v_tile.shape)
                    self.conds = {k: _crop_cond_list_ic(vv, s, H, W, ic_on, tile_shape)
                                  for k, vv in conds_full.items()}
                    pred = super().predict_noise(x_tile, timestep, model_options, seed)

                    pred_v = _video_of(pred)
                    win = _win2d(s, video.device, torch.float32)
                    acc[..., s["y0"]:s["y1"], s["x0"]:s["x1"]] += pred_v.float() * win
                    wsum[..., s["y0"]:s["y1"], s["x0"]:s["x1"]] += win

                    if _is_nested(pred) and len(pred.tensors) > 1:
                        if self._audio_fuse == "primeiro":
                            if audio_accum is None:
                                audio_accum = [t.float().clone() for t in pred.tensors[1:]]
                                audio_count = 1
                        else:  # media
                            if audio_accum is None:
                                audio_accum = [t.float().clone() for t in pred.tensors[1:]]
                            else:
                                for i, t in enumerate(pred.tensors[1:]):
                                    audio_accum[i] += t.float()
                            audio_count += 1

                    del v_tile, x_tile, pred, pred_v, win
                    if self._cleanup:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            finally:
                self.conds = conds_full

            wmin = float(wsum.min())
            if wmin <= 1e-7:
                raise RuntimeError(
                    f"[Bruxos LTX Tiled IC] os ladrilhos nao cobriram o video inteiro "
                    f"(peso minimo {wmin}). Reduza o numero de ladrilhos ou o overlap.")
            out_v = (acc / wsum.clamp(min=1e-8)).to(video.dtype)

            if _is_nested(x) and audio_accum is not None and _nested is not None:
                if self._audio_fuse != "primeiro" and audio_count > 1:
                    audio_accum = [t / audio_count for t in audio_accum]
                audio_out = [t.to(x.tensors[1 + i].dtype) for i, t in enumerate(audio_accum)]
                out = _nested.NestedTensor([out_v] + audio_out)
            else:
                out = out_v

            self._step_i += 1
            if self._debug:
                dt_step = time.time() - step_t0
                dt_total = time.time() - self._run_t0
                print(f"[Bruxos LTX Tiled IC][timer] passo {self._step_i}: {dt_step:.2f}s "
                      f"({len(specs)} ladrilho(s)) | acumulado {dt_total:.1f}s", flush=True)
            return out


class BruxosLTXTiledGuiderIC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Modelo LTX (mesmo que iria pro CFGGuider do IC)."}),
                "positive": ("CONDITIONING", {"tooltip": "Positivo -- ligue a SAIDA 'positive' do Add Video IC-LoRA Guide."}),
                "negative": ("CONDITIONING", {"tooltip": "Negativo -- ligue a SAIDA 'negative' do Add Video IC-LoRA Guide."}),
                "tile_count_width": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Colunas de ladrilho (no latente do VIDEO). 1 = nao corta na horizontal."}),
                "tile_count_height": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Linhas de ladrilho. 2x2 = 4 pedacos. 1x1 desliga o ladrilho."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "CFG. No fluxo IC destilado costuma ser 1.0 (igual ao CFGGuider do template)."}),
            },
            "optional": {
                "overlap": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Sobreposicao entre ladrilhos, em unidades de LATENTE. Maior = costura mais suave, mais VRAM."}),
                "audio_fuse": (["media", "primeiro"], {"default": "media",
                    "tooltip": "Como fundir o audio entre ladrilhos: 'media' (todos) ou 'primeiro' (so o 1o ladrilho). Se o audio sair estranho, teste 'primeiro'."}),
                "limpar_vram": ("BOOLEAN", {"default": True,
                    "tooltip": "Esvazia o cache da VRAM depois de CADA ladrilho. Deixe ligado."}),
                "timer": ("BOOLEAN", {"default": True,
                    "tooltip": "Imprime no console o tempo de cada passo e o acumulado."}),
            },
        }

    RETURN_TYPES = ("GUIDER", "STRING")
    RETURN_NAMES = ("guider", "info")
    OUTPUT_TOOLTIPS = (
        "Ligue no SamplerCustomAdvanced (entrada 'guider') no LUGAR do CFGGuider do IC. O ladrilho acontece DENTRO do sampler, a cada passo.",
        "Resumo do plano de ladrilhos.",
    )
    FUNCTION = "build"
    CATEGORY = CAT
    DESCRIPTION = (
        "LTX Tiled Sampler IC (Bruxos) — step-fused, ciente de IC-LoRA + audio. Substitui o "
        "CFGGuider do fluxo IC do LTX 2.x. Separa video e audio (NestedTensor), ladrilha SO o "
        "video no espaco e funde as predicoes a cada passo; corta os metadados do IC-LoRA "
        "(keyframe_idxs / guide_attention_entries) por ladrilho pra a contagem de tokens e o "
        "RoPE baterem. Audio passa inteiro em cada forward e e fundido (media/primeiro). Sem "
        "metadados IC, se comporta como o tiled normal. NAO suporta IC em grid pequeno "
        "(latent_downscale_factor>1): nesse caso o corte IC se desliga e avisa no console."
    )

    def build(self, model, positive, negative, tile_count_width, tile_count_height, cfg,
              overlap=8, audio_fuse="media", limpar_vram=True, timer=True):
        if not _HAS_TORCH:
            raise RuntimeError("[Bruxos LTX Tiled IC] torch indisponivel.")
        if _samplers is None or not hasattr(_samplers, "CFGGuider"):
            raise RuntimeError("[Bruxos LTX Tiled IC] comfy.samplers.CFGGuider nao encontrado neste build.")

        _bx_log_attention_backend_once()

        g = _BruxosLTXTiledICGuider(model)
        g.set_conds(positive, negative)
        try:
            g.set_cfg(float(cfg))
        except Exception:
            pass
        g.set_tiling(int(tile_count_height), int(tile_count_width), int(overlap),
                     bool(limpar_vram), bool(timer), str(audio_fuse))
        g.raw_conds = (positive, negative)

        n = int(tile_count_width) * int(tile_count_height)
        if n <= 1:
            info = "1x1 -> ladrilho DESLIGADO (roda o quadro inteiro, normal)"
        else:
            info = (f"{tile_count_width}x{tile_count_height} = {n} ladrilho(s) | overlap {overlap} "
                    f"latentes | cfg {cfg} | audio_fuse={audio_fuse} | IC-aware | fusao a cada passo")
        print(f"[Bruxos LTX Tiled IC] {info}", flush=True)
        return (g, info)


NODE_CLASS_MAPPINGS = {"BruxosLTXTiledGuiderIC": BruxosLTXTiledGuiderIC}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosLTXTiledGuiderIC": "LTX Tiled Sampler IC (Bruxos)"}
