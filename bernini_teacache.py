# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — TeaCache pro Bernini/Wan NATIVO do ComfyUI
==========================================================
Porta o TeaCache block-level (pular uma janela de blocos do transformer quando
dois passos de denoise consecutivos produzem estados quase iguais) pro caminho
NATIVO do ComfyUI que os nodes Bruxos usam — nao pro WanModel do wrapper.

Por que funciona sem reescrever o modelo: a assinatura do bloco de atencao do
Wan nativo do ComfyUI e IDENTICA a do wrapper:
    WanAttentionBlock.forward(self, x, e, freqs, context, context_img_len=257,
                              transformer_options={})
Entao o mesmo hook de skip/cache encaixa direto nos blocos do modelo nativo.

Logica portada de ComfyUI-BerniniRWrapper (utils/teacache.py, MIT), que por sua
vez veio do WanVideoWrapper (Kijai). Reimplementado/adaptado pro ModelPatcher
nativo + driver de step proprio (o wrapper dirige o step pelo sampler dele;
aqui a gente dirige por um model_function_wrapper que avanca o contador quando
o timestep muda).

EXPERIMENTAL: mexe no forward dos blocos por monkeypatch (com detach garantido
em finally). Desligado por padrao (so age se voce ligar o node de config e
conectar). Ganho tipico: 1.5-2x, com pequena perda de qualidade conforme o
threshold. TESTE na sua GPU e compare com/sem antes de confiar.
"""

from __future__ import annotations

import logging

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

log = logging.getLogger(__name__)

CAT = "Bruxos do VFX/Bernini"

DEFAULT_START_BLOCK = 3
DEFAULT_MAX_SKIP_BLOCKS = 15
DEFAULT_REL_L1_THRESH = 0.08
DEFAULT_WARMUP_STEPS = 1
DEFAULT_COOLDOWN_STEPS = 2


def _l1(x, y):
    return (x - y).abs().float().mean().item()


def _get_wan_blocks_owner(model):
    """Acha o objeto com `.blocks` (o WanModel) a partir de um ModelPatcher
    NATIVO do ComfyUI. Tenta a API do patcher primeiro, depois anda na cadeia."""
    # 1) API do ModelPatcher nativo
    try:
        dm = model.get_model_object("diffusion_model")
        if dm is not None and hasattr(dm, "blocks"):
            return dm
    except Exception:
        pass
    # 2) anda .model / .diffusion_model
    cur = model
    for _ in range(4):
        if hasattr(cur, "blocks"):
            return cur
        nxt = getattr(cur, "diffusion_model", None) or getattr(cur, "model", None)
        if nxt is None:
            break
        cur = nxt
    return cur if hasattr(cur, "blocks") else None


class TeaCache:
    """Patcha os blocos do WanModel nativo e pula uma janela deles quando o L1
    do input ao start_block cai abaixo do threshold. detach() restaura tudo."""

    def __init__(self, model, *, start_block=DEFAULT_START_BLOCK,
                 max_skip_blocks=DEFAULT_MAX_SKIP_BLOCKS,
                 rel_l1_thresh=DEFAULT_REL_L1_THRESH,
                 warmup_steps=DEFAULT_WARMUP_STEPS,
                 cooldown_steps=DEFAULT_COOLDOWN_STEPS,
                 batch_size=1):
        dm = _get_wan_blocks_owner(model)
        if dm is None:
            raise RuntimeError("[Bruxos TeaCache] nao achei os blocos do WanModel no ModelPatcher.")
        self._wan = dm
        n_blocks = len(dm.blocks)
        self._start = max(0, min(int(start_block), n_blocks - 1))
        self._end = min(self._start + int(max_skip_blocks), n_blocks)
        self._thresh = float(rel_l1_thresh)
        self._warmup = int(warmup_steps)
        self._cooldown = int(cooldown_steps)
        self._batch_gt_1 = int(batch_size) > 1

        self._step = 0
        self._total_steps = 0
        self._skipping = False
        self._cache_output_pending = False
        self._orig_forwards = {}
        self._patched = False

        self._window_cache = {}
        self._window_last_consumed = {}
        self._active_window_key = None
        self._active_cached_input = None
        self._active_cached_output = None

        self._patch()

    def reset(self, total_steps):
        self._step = 0
        self._total_steps = int(total_steps)
        self._skipping = False
        self._cache_output_pending = False
        self._window_cache.clear()
        self._window_last_consumed.clear()
        self._active_window_key = None
        self._active_cached_input = None
        self._active_cached_output = None

    def step(self):
        self._step += 1
        self._skipping = False

    def detach(self):
        if not self._patched:
            return
        for i, orig in self._orig_forwards.items():
            try:
                if i < len(self._wan.blocks):
                    self._wan.blocks[i].forward = orig
            except Exception:
                pass
        self._orig_forwards.clear()
        self._window_cache.clear()
        self._window_last_consumed.clear()
        self._active_cached_input = None
        self._active_cached_output = None
        self._wan = None
        self._patched = False

    def _active(self):
        return self._warmup < self._step <= self._total_steps - self._cooldown

    def _patch(self):
        if self._patched:
            return
        for i in range(self._start, self._end):
            blk = self._wan.blocks[i]
            self._orig_forwards[i] = blk.forward
            blk.forward = self._hook(blk, i)
        self._patched = True
        log.info("[Bruxos TeaCache] blocos [%d,%d) de %d, thresh=%.3f warmup=%d cooldown=%d",
                 self._start, self._end, len(self._wan.blocks), self._thresh,
                 self._warmup, self._cooldown)

    def _hook(self, block, idx):
        orig = self._orig_forwards[idx]

        def forward(x, e, freqs, context, context_img_len=257, transformer_options=None):
            if transformer_options is None:
                transformer_options = {}
            if idx == self._start:
                win_key = transformer_options.get("_bx_context_window") or 0
                fresh = (win_key not in self._window_last_consumed
                         or self._step != self._window_last_consumed[win_key])
                if fresh:
                    self._window_last_consumed[win_key] = self._step
                    self._active_window_key = win_key
                    if win_key in self._window_cache:
                        self._active_cached_input, self._active_cached_output = self._window_cache[win_key]
                    else:
                        self._active_cached_input = None
                        self._active_cached_output = None
                    x_cmp = x[:1] if self._batch_gt_1 else x
                    if (self._active() and self._active_cached_input is not None
                            and _l1(x_cmp, self._active_cached_input) < self._thresh):
                        self._skipping = True
                    else:
                        self._skipping = False
                        self._active_cached_input = x_cmp.detach()
                        self._cache_output_pending = True
                else:
                    self._active_window_key = win_key
                    self._skipping = False

            if self._skipping:
                result = self._active_cached_output if idx == self._start else x
                if idx == self._start and self._batch_gt_1 and result is not None:
                    result = result.expand(x.shape[0], -1, -1)
                if idx == self._end - 1:
                    self._skipping = False
                return result

            result = orig(x, e=e, freqs=freqs, context=context,
                          context_img_len=context_img_len,
                          transformer_options=transformer_options)

            if idx == self._end - 1 and self._cache_output_pending:
                result_cache = result[:1].detach() if self._batch_gt_1 else result.detach()
                self._window_cache[self._active_window_key] = (self._active_cached_input, result_cache)
                self._cache_output_pending = False
            return result

        try:
            forward = torch._dynamo.disable(forward)
        except Exception:
            pass
        return forward


def make_step_driver(tc):
    """model_function_wrapper que avanca o contador do TeaCache 1x por passo de
    denoise (quando o timestep muda) e passa a chamada adiante. Compativel com
    cfg=1 (1 forward/passo) e cfg>1 (cond+uncond no MESMO timestep = 1 step())."""
    state = {"last_t": None}

    def wrapper(model_function, params):
        try:
            t = params["timestep"]
            key = float(t.flatten()[0].item()) if hasattr(t, "flatten") else float(t)
            if state["last_t"] is None or key != state["last_t"]:
                state["last_t"] = key
                tc.step()
        except Exception:
            pass
        x = params["input"]
        t = params["timestep"]
        c = params["c"]
        return model_function(x, t, **c)

    return wrapper


class BruxosBerniniTeaCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rel_l1_thresh": ("FLOAT", {"default": DEFAULT_REL_L1_THRESH, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Pula a janela de blocos quando o L1 entre passos consecutivos < threshold. 0.04 = seguro (perda minima), 0.08 = padrao, 0.12+ = rapido (pode degradar). Comece em 0.08 e ajuste."}),
                "max_skip_blocks": ("INT", {"default": DEFAULT_MAX_SKIP_BLOCKS, "min": 1, "max": 40, "step": 1,
                    "tooltip": "Quantos blocos a janela de cache cobre. Maior = mais aceleracao, mais risco de qualidade."}),
                "start_block": ("INT", {"default": DEFAULT_START_BLOCK, "min": 0, "max": 39, "step": 1,
                    "tooltip": "Primeiro bloco cacheavel. O L1 e medido na entrada dele pra decidir pular ou computar."}),
                "warmup_steps": ("INT", {"default": DEFAULT_WARMUP_STEPS, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Primeiros N passos que NUNCA cacheiam (formacao de estrutura). Com 6 steps, 1 e bom."}),
                "cooldown_steps": ("INT", {"default": DEFAULT_COOLDOWN_STEPS, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Ultimos N passos que NUNCA cacheiam (refino de detalhe). Com 6 steps, 1-2 e bom."}),
            },
        }

    RETURN_TYPES = ("BERNINI_TEACACHE", "STRING")
    RETURN_NAMES = ("teacache", "info")
    OUTPUT_TOOLTIPS = (
        "Ligue no input 'teacache' do Bernini I2V (ou dos nodes Bernini que aceitarem). Sem ligar = desligado.",
        "Resumo da config.",
    )
    FUNCTION = "build"
    CATEGORY = CAT
    DESCRIPTION = (
        "TeaCache (Bruxos): acelera a amostragem pulando uma janela de blocos do transformer quando dois "
        "passos consecutivos sao quase iguais (1.5-2x, pequena perda de qualidade). Portado pro Wan NATIVO "
        "do ComfyUI. EXPERIMENTAL — teste com/sem e ajuste o threshold. Ligue no input 'teacache' do node."
    )

    def build(self, rel_l1_thresh, max_skip_blocks, start_block, warmup_steps, cooldown_steps):
        cfg = {
            "rel_l1_thresh": float(rel_l1_thresh),
            "max_skip_blocks": int(max_skip_blocks),
            "start_block": int(start_block),
            "warmup_steps": int(warmup_steps),
            "cooldown_steps": int(cooldown_steps),
        }
        info = (f"TeaCache thresh={cfg['rel_l1_thresh']:.3f} blocos[{cfg['start_block']}.."
                f"{cfg['start_block']+cfg['max_skip_blocks']}) warmup={cfg['warmup_steps']} "
                f"cooldown={cfg['cooldown_steps']}")
        print(f"[Bruxos TeaCache] {info}", flush=True)
        return (cfg, info)


NODE_CLASS_MAPPINGS = {"BruxosBerniniTeaCache": BruxosBerniniTeaCache}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosBerniniTeaCache": "Bernini TeaCache (Bruxos)"}
