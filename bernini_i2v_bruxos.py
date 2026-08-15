# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Bernini I2V / Reference-to-Video (com KSampler embutido)
=======================================================================
Gera um VIDEO a partir de UMA (ou duas) imagem de REFERENCIA, sem source_video.
E o mesmo caminho do node oficial "BerniniConditioning" no modo i2v:

  - a imagem entra como REFERENCIA (nao como fonte de movimento);
  - o node cria um LATENTE VAZIO de `length` frames;
  - dois passes (high noise -> low noise) geram do ruido, guiados pela
    referencia + o texto do positive.

Diferente do "Bernini Infinity" (que e V2V e EXIGE source_video pra tirar o
movimento/estrutura), aqui NAO ha fonte: o modelo inventa o movimento guiado
pela referencia. E "reference-to-video" (familia S2V/RV2V do paper), nao edicao.

Tudo num node so: conditioning de referencia + KSampler (split high/low) +
decode. Reaproveita os helpers do proprio pacote (mesma amostragem do Bernini
Infinity), entao o comportamento casa com o resto.

Categoria: Bruxos do VFX/Bernini
"""

import time
import logging

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Helpers do pacote (mesma amostragem do Bernini Infinity).
try:
    from .nodes import (
        BerniniInfinity as _BERNINI,
        _collect_reference_latents as _bx_collect_ref,
        _clone_conditioning_set_values as _bx_clone_cond,
        _make_empty_latent as _bx_make_latent,
        _decode_video as _bx_decode_video,
        _align_up_4n1 as _bx_align_up_4n1,
        _lat_len as _bx_lat_len,
        _mem_cleanup as _bx_mem_cleanup,
        _make_context_wrapper as _bx_make_ctx_wrapper,
        _encode_video as _bx_encode_video,
        _merge_linear_overlap as _bx_merge_overlap,
        BasicScheduler as _bx_BasicScheduler,
        KSamplerSelect as _bx_KSamplerSelect,
        SplitSigmas as _bx_SplitSigmas,
    )
    _HAS_HELPERS = True
except Exception:
    try:
        from nodes import (
            BerniniInfinity as _BERNINI,
            _collect_reference_latents as _bx_collect_ref,
            _clone_conditioning_set_values as _bx_clone_cond,
            _make_empty_latent as _bx_make_latent,
            _decode_video as _bx_decode_video,
            _align_up_4n1 as _bx_align_up_4n1,
            _lat_len as _bx_lat_len,
            _mem_cleanup as _bx_mem_cleanup,
            _make_context_wrapper as _bx_make_ctx_wrapper,
            _encode_video as _bx_encode_video,
            _merge_linear_overlap as _bx_merge_overlap,
            BasicScheduler as _bx_BasicScheduler,
            KSamplerSelect as _bx_KSamplerSelect,
            SplitSigmas as _bx_SplitSigmas,
        )
        _HAS_HELPERS = True
    except Exception as e:  # pragma: no cover
        logging.warning(f"[Bernini I2V] helpers do nodes.py indisponiveis: {e}")
        _BERNINI = None
        _bx_collect_ref = _bx_clone_cond = _bx_make_latent = _bx_decode_video = None
        _bx_align_up_4n1 = _bx_lat_len = _bx_mem_cleanup = _bx_make_ctx_wrapper = None
        _bx_encode_video = _bx_merge_overlap = None
        _bx_BasicScheduler = _bx_KSamplerSelect = _bx_SplitSigmas = None
        _HAS_HELPERS = False

try:
    import comfy.samplers as _cs
    _SAMPLERS = list(getattr(_cs, "SAMPLER_NAMES", ["euler"]))
    _SCHEDULERS = list(getattr(_cs, "SCHEDULER_NAMES", ["simple"]))
except Exception:
    _SAMPLERS = ["euler", "res_multistep"]
    _SCHEDULERS = ["simple"]

# node de tile em PIXELS (o que FUNCIONA com um video de fonte) — usado no
# hd_upscale pra subir a resolucao por dentro do proprio node I2V.
try:
    from .bernini_tiled_optimized import BruxosBerniniInfinityTiledOptimized as _BX_TILED
except Exception:
    try:
        from bernini_tiled_optimized import BruxosBerniniInfinityTiledOptimized as _BX_TILED
    except Exception:
        _BX_TILED = None

# TeaCache (acelera pulando blocos do transformer) — opcional.
try:
    from .bernini_teacache import TeaCache as _BX_TEACACHE, make_step_driver as _bx_tc_driver
except Exception:
    try:
        from bernini_teacache import TeaCache as _BX_TEACACHE, make_step_driver as _bx_tc_driver
    except Exception:
        _BX_TEACACHE = None
        _bx_tc_driver = None

CAT = "Bruxos do VFX/Bernini"


def _fmt_t(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{s - 60 * int(s // 60):04.1f}s"


def _tile_anchor_image(image, width, height, mode="crop"):
    """Primeiro frame da referencia no canvas exato do latent de geracao."""
    x = image[:1, ..., :3].movedim(-1, 1).float()
    sh, sw = int(x.shape[-2]), int(x.shape[-1])
    W, H = int(width), int(height)
    if mode == "stretch" or sh <= 0 or sw <= 0:
        out = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    else:
        scale = max(W / float(sw), H / float(sh))
        rw, rh = max(W, int(round(sw * scale))), max(H, int(round(sh * scale)))
        resized = torch.nn.functional.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)
        x0, y0 = max(0, (rw - W) // 2), max(0, (rh - H) // 2)
        out = resized[..., y0:y0 + H, x0:x0 + W]
    return out.movedim(1, -1).contiguous().clamp(0, 1)


class BruxosBerniniI2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "Positivo (o que voce quer gerar). Use o Bernini Prompt Enhancer no modo i2v ou um CLIP Text Encode."}),
                "negative": ("CONDITIONING", {"tooltip": "Negativo (ex.: 'bad video')."}),
                "high_model": ("MODEL", {"tooltip": "Modelo HIGH noise (Bernini/Wan). Roda os primeiros steps (ate split_step)."}),
                "low_model": ("MODEL", {"tooltip": "Modelo LOW noise. Roda os steps finais."}),
                "vae": ("VAE", {"tooltip": "VAE de VIDEO do Wan (o mesmo do Bernini)."}),
                "reference_image": ("IMAGE", {"tooltip": "A imagem de REFERENCIA (identidade/aparencia). O video sera gerado a partir dela — nao e fonte de movimento, o modelo inventa o movimento guiado por ela + o prompt."}),
                "width": ("INT", {"default": 848, "min": 16, "max": 8192, "step": 16, "tooltip": "Largura de saida (multiplo de 16). O Bernini foi treinado em 480x832/832x480 — fique perto disso."}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16, "tooltip": "Altura de saida (multiplo de 16)."}),
                "length": ("INT", {"default": 81, "min": 5, "max": 1024, "step": 4, "tooltip": "Quantos frames gerar. Use 4n+1 (5,9,...,81,121). Mais frames = mais VRAM/tempo (atencao O(n^2))."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Semente do ruido."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 1000, "tooltip": "Steps totais. Com LoRA LightX2V: 6."}),
                "split_step": ("INT", {"default": 4, "min": 0, "max": 999, "tooltip": "Em qual step troca do high pro low. Com LightX2V: 4."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "CFG. Com LightX2V use 1.0."}),
                "sampler_name": (_SAMPLERS, {"default": _SAMPLERS[0] if _SAMPLERS else "euler", "tooltip": "Algoritmo de amostragem (ex.: euler)."}),
                "scheduler": (_SCHEDULERS, {"default": _SCHEDULERS[0] if _SCHEDULERS else "simple", "tooltip": "Scheduler (ex.: simple)."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "1.0 = gera do zero (recomendado p/ i2v)."}),
            },
            "optional": {
                "reference_image_2": ("IMAGE", {"tooltip": "2a referencia (opcional) — ex.: um close do rosto pra reforcar a identidade."}),
                "reference_image_3": ("IMAGE", {"tooltip": "3a referencia (opcional)."}),
                "reference_image_4": ("IMAGE", {"tooltip": "4a referencia (opcional)."}),
                "reference_image_5": ("IMAGE", {"tooltip": "5a referencia (opcional)."}),
                "reference_image_6": ("IMAGE", {"tooltip": "6a referencia (opcional)."}),
                "reference_image_7": ("IMAGE", {"tooltip": "7a referencia (opcional)."}),
                "reference_image_8": ("IMAGE", {"tooltip": "8a referencia (opcional). O Bernini aceita ate ~8 refs; mais que isso costuma nao ajudar e gasta mais VRAM. Cada IMAGE pode ter varios frames, e todos viram referencia."}),
                "reference_video": ("IMAGE", {"tooltip": "Video de referencia (opcional). Vira contexto extra."}),
                "teacache": ("BERNINI_TEACACHE", {"tooltip": "[experimental] Ligue a saida do node 'Bernini TeaCache (Bruxos)' aqui pra acelerar (1.5-2x) pulando blocos do transformer. So age com guidance_mode=off. Sem ligar = desligado. Teste com/sem e compare qualidade."}),
                "guidance_mode": (["off", "multi", "tiled"], {"default": "off", "tooltip": "off = CFG normal (rapido). multi = guidance por stream (~4x mais lento). tiled = divide o latent espacialmente; a primeira reference_image vira ancora alinhada e cada tile recebe uma parte diferente. Refs extras seguem tile_context_mode."}),
                "reference_strength": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 30.0, "step": 0.05, "tooltip": "[guidance_mode=multi] Quanto FORCA a imagem de referencia. Maior = mais preso a ela (identidade/roupa/estilo). Paper: 1.25 (base) ate 3.0 (RV2V, referencia forte). Comece em 1.5-2.5. So tem efeito com guidance_mode=multi. Aplica aos dois streams de referencia (1a ref + refs extras)."}),
                "ref_influence_vid_off": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05, "tooltip": "[guidance_mode=off/tiled] Controla a influencia do VIDEO de referencia (reference_video) SEM o modo multi (4x mais lento) -- escala a magnitude do latente antes de virar contexto. 1.0 = neutro. Independente de ref_influence_img_off: suba um e desca o outro pra pender mais pro video ou mais pras imagens. EXPERIMENTAL -- comece em 1.5-2.5; 5+ tende a quebrar a imagem. Sem efeito no modo multi (la manda reference_strength)."}),
                "ref_influence_img_off": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05, "tooltip": "[guidance_mode=off/tiled] Controla a influencia das IMAGENS de referencia SEM o modo multi -- escala a magnitude do latente de cada imagem antes de virar contexto. 1.0 = neutro. Pra 'parecer mais com as imagens': suba este. Pra 'parecer mais com o video de referencia': suba o ref_influence_vid_off e desca este. EXPERIMENTAL -- comece em 1.5-2.5; 5+ tende a quebrar a imagem. Sem efeito no modo multi (la manda reference_strength)."}),
                "tile_w": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "[guidance_mode=tiled] Colunas. Cada tile recebe a regiao correspondente da primeira reference_image."}),
                "tile_h": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "[guidance_mode=tiled] Linhas. 2x2 gera quatro partes diferentes que completam o quadro."}),
                "tile_overlap": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1, "tooltip": "[guidance_mode=tiled] Sobreposicao entre ladrilhos, em latentes (1 ~ 8px)."}),
                "mode": (["context_window", "sequential"], {"default": "context_window", "tooltip": "Como cortar o video no tempo (so importa com chunk_size>0). context_window = a cada passo de denoise roda TODAS as janelas e funde (mais coerente, mas o video inteiro fica na VRAM -> forca janelas pequenas em alta-res). sequential = processa UM BLOCO por vez, denoise completo, carregando os ultimos frames como memoria (tail) pro proximo -> MUITO menos VRAM, entao voce pode usar chunk GRANDE (49/81) mesmo em 1080p = poucas janelas grandes = mais rapido no nativo pesado. Comece com context_window; troque pra sequential se em alta-res o chunk pequeno gerar janelas demais."}),
                "chunk_size": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 4, "tooltip": "[JANELA TEMPORAL — anti-cliff em alta resolucao] Tamanho da janela/bloco em FRAMES. 0 = desligado (video inteiro = atencao global). CONTRAINTUITIVO: chunk MAIOR = MENOS janelas = mais rapido (ate o limite de VRAM). Nao use valores pequenos tipo 12 (gera janelas demais). Use 4n+1 (25,33,49,81). 720p context_window: ~33. 1080p: use mode=sequential + chunk 49/81."}),
                "overlap": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1, "tooltip": "[janela/bloco] Sobreposicao entre janelas/blocos, em frames, pro crossfade nao deixar emenda. 8-16 e bom. So tem efeito com chunk_size > 0 e menor que length."}),
                "ref_max_size": ("INT", {"default": 1280, "min": 16, "max": 8192, "step": 16, "tooltip": "Lado maior (px) pra redimensionar as referencias antes de virar latente. Maior = referencia mais detalhada, mais VRAM."}),
                "limpar_vram": (["off", "leve", "agressivo"], {"default": "leve", "tooltip": "Limpeza de VRAM entre o high e o low (igual ao Bernini Infinity)."}),
                "force_unload_between_passes": ("BOOLEAN", {"default": False, "tooltip": "[anti-OOM] Descarrega o high antes do low, furando o guard de LoRA. Ligue se der OOM na transicao high->low."}),
                "monitor_memoria": ("BOOLEAN", {"default": False, "tooltip": "Cronometro no console."}),
                "tile_anchor_fit": (["crop", "stretch"], {"default": "crop", "tooltip": "[tiled] Como ajustar a primeira reference_image ao canvas antes de dividi-la. crop preserva proporcao e corta bordas; stretch mostra tudo, mas pode distorcer."}),
                "tile_layout": ("BRUXOS_TILE_LAYOUT", {"tooltip": "[tiled] Layout desenhado no Bernini Custom Tile Layout. Substitui tile_w x tile_h."}),
                "tile_context_mode": (["hybrid", "local", "global"], {"default": "hybrid", "tooltip":
                    "[tiled] global mantem refs extras inteiras; local recorta por tile; hybrid mistura "
                    "o recorte detalhado com 20% de uma ancora global reduzida (recomendado)."}),
                # NOTA: o HD upscale saiu daqui. Refino ladrilhado com o renderer
                # Bernini + LoRA destilado (LightX2V) so funciona em denoise cheio;
                # em denoise parcial ele alucina (rede/fantasma). Pra subir resolucao,
                # gere em 480/720p e use o Bernini Infinity Tiled (denoise ~1.0) ou
                # um upscaler de video (SeedVR2). Ver README.
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "INT")
    RETURN_NAMES = ("images", "latent", "total_frames")
    OUTPUT_TOOLTIPS = (
        "Video gerado a partir da referencia (length frames).",
        "Latente do resultado.",
        "Numero de frames gerados.",
    )
    FUNCTION = "generate"
    CATEGORY = CAT
    DESCRIPTION = (
        "Bernini I2V / Reference-to-Video (Bruxos): gera um video a partir de UMA imagem de referencia, "
        "SEM source_video. A imagem entra como referencia, o node cria um latente vazio de N frames e "
        "gera do ruido (high->low) guiado pela referencia + prompt. Node unico: conditioning + KSampler "
        "(split high/low) + decode. Mesmo caminho do 'BerniniConditioning' oficial no modo i2v."
    )

    def generate(self, positive, negative, high_model, low_model, vae, reference_image,
                 width, height, length, seed, steps, split_step, cfg, sampler_name, scheduler, denoise,
                 reference_image_2=None, reference_image_3=None, reference_image_4=None,
                 reference_image_5=None, reference_image_6=None, reference_image_7=None,
                 reference_image_8=None, reference_video=None, teacache=None,
                 guidance_mode="off", reference_strength=1.5,
                 ref_influence_vid_off=1.0, ref_influence_img_off=1.0,
                 tile_w=2, tile_h=2, tile_overlap=8, mode="context_window",
                 chunk_size=0, overlap=8, ref_max_size=1280,
                 limpar_vram="leve", force_unload_between_passes=False, monitor_memoria=False,
                 tile_anchor_fit="crop",
                 tile_layout=None,
                 tile_context_mode="hybrid",
                 # compat: workflows antigos podem mandar hd_* -> aceitos e ignorados
                 hd_upscale=False, hd_width=1664, hd_height=960,
                 hd_tiles_w=2, hd_tiles_h=2, hd_denoise=0.35, **_ignored):
        if not _HAS_TORCH:
            raise RuntimeError("[Bernini I2V] torch indisponivel.")
        if not _HAS_HELPERS or _BERNINI is None:
            raise RuntimeError("[Bernini I2V] helpers do nodes.py nao importaram — instale o pacote completo.")

        t0 = time.time()
        # VAE do Wan: espacial /8, mas o canvas precisa de multiplo de 16 pra bater
        # com o resto do pipeline. Arredonda pra baixo pro multiplo de 16 mais proximo.
        W = max(16, (int(width) // 16) * 16)
        H = max(16, (int(height) // 16) * 16)
        aligned = _bx_align_up_4n1(int(length))

        # ---- REFERENCIAS -> latentes de contexto (SEM source_video) ----
        # a ordem importa: _collect_reference_latents percorre sorted(chaves).
        # Chave "reference_image_0..8" (1 digito) ordena certo.
        _all_refs = [reference_image, reference_image_2, reference_image_3, reference_image_4,
                     reference_image_5, reference_image_6, reference_image_7, reference_image_8]
        refs = {}
        idx = 0
        for img in _all_refs:
            if img is not None:
                refs[f"reference_image_{idx}"] = img
                idx += 1

        # ref_influence_vid_off/img_off so valem fora do multi (la quem manda e
        # reference_strength via CFG multi-stream) -- ficam neutros (1.0) em multi
        # pra nao dobrar o efeito.
        _eff_mode = guidance_mode if guidance_mode in ("multi", "tiled") else "off"
        _ref_scale_vid = float(ref_influence_vid_off) if _eff_mode != "multi" else 1.0
        _ref_scale_img = float(ref_influence_img_off) if _eff_mode != "multi" else 1.0

        tiled_anchor = False
        if _eff_mode == "tiled" and reference_image is not None:
            # context[0] e espacial: seu HxW casa com o latent vazio. O guider
            # tiled recorta este item nas mesmas coordenadas do target. Todo o
            # resto permanece global para identidade/memoria.
            anchor_px = _tile_anchor_image(reference_image, W, H, tile_anchor_fit)
            anchor_latent = _bx_encode_video(vae, anchor_px)
            if _ref_scale_img != 1.0:
                anchor_latent = anchor_latent * _ref_scale_img
            extra_refs = {k: v for k, v in refs.items() if k != "reference_image_0"}
            context_latents = [anchor_latent]
            context_latents.extend(_bx_collect_ref(
                vae, int(aligned), int(ref_max_size),
                reference_video=reference_video, reference_images=extra_refs,
                scale_vid=_ref_scale_vid, scale_img=_ref_scale_img,
            ))
            tiled_anchor = True
        else:
            context_latents = list(_bx_collect_ref(
                vae, int(aligned), int(ref_max_size),
                reference_video=reference_video, reference_images=refs,
                scale_vid=_ref_scale_vid, scale_img=_ref_scale_img,
            ))
        # avisos (NAO travam nada — so orientam; roda do jeito que voce pediu)
        n_ref_imgs = len(refs)
        if not context_latents:
            print("[Bernini I2V] AVISO: sem referencia nem reference_video — vira T2V puro "
                  "(so o texto guia). Ligue ao menos a reference_image pra i2v.", flush=True)
        elif n_ref_imgs == 1 and reference_video is None:
            print("[Bernini I2V] DICA: o Bernini i2v costuma ir MAL com 1 referencia so — "
                  "prefere 2-3+ imagens (a comunidade confirma). E no prompt mencione cada uma "
                  "como 'from image0', 'from image1' (nao 'in image0').", flush=True)
        if int(length) > 121:
            print(f"[Bernini I2V] AVISO: length={int(length)} > 121 — acima de 121 frames o Bernini "
                  f"costuma dar cor desbotada / movimento tremido (50/50 de sucesso). Valores "
                  f"seguros: 81 ou 121. (Nao estou limitando — rodando com {int(length)} como voce pediu.)",
                  flush=True)

        print(f"[Bernini I2V] {W}x{H} x{int(length)}f (alinhado {aligned}) | "
              f"refs={len(context_latents)} | steps={int(steps)} split={int(split_step)} "
              f"cfg={float(cfg)}", flush=True)

        values = {"context_latents": context_latents}
        pos = _bx_clone_cond(positive, values)
        neg = _bx_clone_cond(negative, values)

        sampler = _bx_KSamplerSelect.execute(sampler_name).args[0]
        sigmas = _bx_BasicScheduler.execute(low_model, scheduler, int(steps), float(denoise)).args[0]
        high_sigmas, low_sigmas = _bx_SplitSigmas.execute(sigmas, int(split_step)).args

        bern = _BERNINI()
        # Espelha o Bernini Infinity: multi = forca a referencia (w_vid/w_img);
        # tiled = target e ancora espacial recortados juntos a cada passo.
        bern._g_mode = guidance_mode if guidance_mode in ("multi", "tiled") else "off"
        bern._g_wvid = float(reference_strength)
        bern._g_wimg = float(reference_strength)
        bern._g_cfg_warned = False
        # atributos do modo tiled (mesmos nomes que o BerniniInfinity.render usa)
        bern._g_cols = int(tile_w)
        bern._g_rows = int(tile_h)
        bern._g_overlap = int(tile_overlap)
        bern._g_blend = "hann"
        bern._g_tile_cleanup = True
        bern._g_tile_layout = tile_layout if isinstance(tile_layout, dict) else None
        bern._g_tile_context_mode = str(tile_context_mode)
        if bern._g_mode == "multi":
            print(f"[Bernini I2V] guidance=multi | reference_strength={float(reference_strength):.2f} "
                  f"(~4x mais lento, forca a referencia).", flush=True)
        elif bern._g_mode == "tiled":
            if tiled_anchor:
                _layout_name = f"custom {len(tile_layout.get('tiles', []))} tiles" if isinstance(tile_layout, dict) else f"{tile_w}x{tile_h}"
                print(f"[Bernini I2V] guidance=tiled ({_layout_name}, overlap {tile_overlap}) | "
                      f"context={bern._g_tile_context_mode} | "
                      f"reference_image -> ancora espacial {W}x{H} ({tile_anchor_fit}); "
                      f"cada tile recebe uma regiao diferente.", flush=True)
            else:
                print(f"[Bernini I2V] guidance=tiled ({tile_w}x{tile_h}, overlap {tile_overlap}) — "
                      f"T2V puro sem referencia.", flush=True)

        latent = {"samples": _bx_make_latent(int(aligned), int(W), int(H), 1)}

        # ---- JANELA TEMPORAL (anti-cliff em alta resolucao) ----
        # Se chunk_size>0 e menor que o video, corta a atencao em janelas
        # deslizantes no tempo -> a sequencia para de crescer com a duracao.
        L = int(length)
        win_lat = _bx_lat_len(int(chunk_size)) if int(chunk_size) > 0 else 0
        ovl_lat = _bx_lat_len(int(overlap)) if int(overlap) > 0 else 0
        # sequential = "por partes": bloco a bloco (menos VRAM, chunk grande possivel)
        use_sequential = bool(mode == "sequential" and int(chunk_size) > 0
                              and int(chunk_size) < L and _bx_encode_video is not None
                              and _bx_merge_overlap is not None)
        # context_window = janelas por passo (so quando NAO for sequential)
        use_window = bool(not use_sequential and win_lat and _bx_make_ctx_wrapper is not None
                          and win_lat < _bx_lat_len(L))
        if int(chunk_size) > 0 and not use_window and not use_sequential:
            print(f"[Bernini I2V] chunk_size={chunk_size} >= length; sem janela/bloco "
                  f"(video inteiro cabe de uma vez).", flush=True)

        # TeaCache so age no caminho de amostragem padrao (guidance off). E ele
        # usa o MESMO slot de model_function_wrapper que a janela temporal, entao
        # os dois nao coexistem: se a janela estiver ativa, ela tem prioridade
        # (e a correcao de alta resolucao) e o TeaCache e pulado neste run.
        tc_cfg = teacache if (teacache and _BX_TEACACHE is not None
                              and bern._g_mode == "off" and not use_window) else None
        if teacache and tc_cfg is None:
            motivo = ("janela temporal ativa (usa o mesmo wrapper)" if use_window
                      else "so funciona com guidance_mode=off")
            print(f"[Bernini I2V] TeaCache ignorado ({motivo}).", flush=True)

        if use_window:
            print(f"[Bernini I2V] janela temporal ON (context_window): chunk={chunk_size}f "
                  f"(win_lat={win_lat}), overlap={overlap}f -> atencao capada. DICA: chunk maior = "
                  f"menos janelas = mais rapido (ate o limite de VRAM).", flush=True)

        def _pass(model_src, add_noise, seed_, sig, latent_in, pos_b, neg_b, allow_window=True):
            m = model_src.clone()
            tc = None
            if allow_window and use_window:
                try:
                    wrapper = _bx_make_ctx_wrapper(int(win_lat), int(ovl_lat), int(ovl_lat), jitter=True)
                    m.set_model_unet_function_wrapper(wrapper)
                except Exception as e:
                    print(f"[Bernini I2V] janela temporal falhou ({e}); atencao global.", flush=True)
            elif tc_cfg is not None:
                try:
                    tc = _BX_TEACACHE(m, **tc_cfg)
                    tc.reset(total_steps=max(1, int(sig.shape[0]) - 1))
                    m.set_model_unet_function_wrapper(_bx_tc_driver(tc))
                except Exception as e:
                    print(f"[Bernini I2V] TeaCache desligado neste passe ({e}).", flush=True)
                    tc = None
            try:
                return bern._sample_pass(m, add_noise, seed_, float(cfg), pos_b, neg_b, sampler, sig, latent_in)
            finally:
                if tc is not None:
                    try:
                        tc.detach()
                    except Exception:
                        pass

        result_latent = None
        if use_sequential:
            # ---- MODO SEQUENTIAL: bloco a bloco, com tail-memory + crossfade ----
            fstep = max(1, int(chunk_size) - int(overlap))
            tail_n = max(0, min(int(overlap), 8))
            print(f"[Bernini I2V] SEQUENTIAL ON: bloco={chunk_size}f, overlap={overlap}f "
                  f"(step={fstep}, tail={tail_n}) -> um bloco por vez, VRAM baixa.", flush=True)
            stitched = None
            prev_imgs = None
            bi = 0
            for start in range(0, L, fstep):
                end = min(start + int(chunk_size), L)
                blk_len = end - start
                if blk_len <= 0:
                    break
                a_blk = _bx_align_up_4n1(blk_len)
                # contexto do bloco = refs (identidade) + tail-memory do bloco anterior
                ctx_b = list(context_latents)
                if prev_imgs is not None and tail_n > 0:
                    try:
                        tail = prev_imgs[-tail_n:]
                        ctx_b = ctx_b + [_bx_encode_video(vae, tail[..., :3])]
                    except Exception as _e:
                        print(f"[Bernini I2V][seq] tail-memory falhou ({_e}); bloco sem memoria.", flush=True)
                pos_b = _bx_clone_cond(positive, {"context_latents": ctx_b})
                neg_b = _bx_clone_cond(negative, {"context_latents": ctx_b})
                latent_b = {"samples": _bx_make_latent(int(a_blk), int(W), int(H), 1)}
                high_b = _pass(high_model, True, int(seed), high_sigmas, latent_b, pos_b, neg_b, allow_window=False)
                _bx_mem_cleanup(limpar_vram, model=high_model, between_passes=True,
                                force_unload=bool(force_unload_between_passes))
                low_b = _pass(low_model, False, 0, low_sigmas, high_b, pos_b, neg_b, allow_window=False)
                result_latent = low_b["samples"]
                imgs_b = _bx_decode_video(vae, result_latent, False).float().clamp(0, 1)
                if int(imgs_b.shape[0]) > blk_len:
                    imgs_b = imgs_b[:blk_len]
                elif int(imgs_b.shape[0]) < blk_len:
                    imgs_b = torch.cat([imgs_b, imgs_b[-1:].repeat(blk_len - int(imgs_b.shape[0]), 1, 1, 1)], dim=0)
                imgs_b = imgs_b.cpu()
                prev_imgs = imgs_b
                stitched = imgs_b if stitched is None else _bx_merge_overlap(stitched, imgs_b, int(overlap))
                bi += 1
                print(f"[Bernini I2V][seq] bloco {bi}: frames {start}..{end - 1} ({blk_len}f) ok", flush=True)
                _bx_mem_cleanup(limpar_vram)
                if end >= L:
                    break
            imgs = stitched if stitched is not None else torch.zeros((1, int(H), int(W), 3))
        else:
            # ---- caminho padrao: latente unico (com janela/teacache opcionais) ----
            high = _pass(high_model, True, int(seed), high_sigmas, latent, pos, neg)
            _bx_mem_cleanup(limpar_vram, model=high_model, between_passes=True,
                            force_unload=bool(force_unload_between_passes))
            low = _pass(low_model, False, 0, low_sigmas, high, pos, neg)
            result_latent = low["samples"]
            imgs = _bx_decode_video(vae, result_latent, False).float().clamp(0, 1)

        # corta o padding de alinhamento 4n+1 de volta ao length pedido
        if int(imgs.shape[0]) > L:
            imgs = imgs[:L]
        elif int(imgs.shape[0]) < L:
            imgs = torch.cat([imgs, imgs[-1:].repeat(L - int(imgs.shape[0]), 1, 1, 1)], dim=0)
        if result_latent is not None:
            try:
                result_latent = result_latent[:, :, :_bx_lat_len(L)]
            except Exception:
                pass

        _bx_mem_cleanup(limpar_vram)

        # ---- HD upscale REMOVIDO do I2V ----
        # O refino ladrilhado com o renderer Bernini + LoRA destilado so presta em
        # denoise cheio; aqui era denoise parcial e alucinava. Pra subir resolucao:
        # gere aqui em 480/720p e leve a saida pro Bernini Infinity Tiled (denoise
        # ~1.0) ou pro SeedVR2. Mantemos o parametro so por compatibilidade.
        hd_msg = ""
        if bool(hd_upscale):
            print("[Bernini I2V] hd_upscale foi REMOVIDO deste node (alucinava com LoRA "
                  "destilado em denoise parcial). Use o Bernini Infinity Tiled ou o SeedVR2 "
                  "no video gerado. Devolvendo o nativo.", flush=True)

        info = (f"{W}x{H} x{L}f | refs={len(context_latents)} | "
                f"steps {int(steps)} (split {int(split_step)}){hd_msg} | {_fmt_t(time.time() - t0)}")
        print(f"[Bernini I2V] DONE: {info}", flush=True)
        lat_out = result_latent.cpu() if result_latent is not None else _bx_make_latent(
            int(_bx_align_up_4n1(L)), int(W), int(H), 1).cpu()
        return (imgs.cpu(), {"samples": lat_out}, int(imgs.shape[0]))


NODE_CLASS_MAPPINGS = {"BruxosBerniniI2V": BruxosBerniniI2V}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosBerniniI2V": "Bernini I2V / Ref-to-Video (Bruxos)"}
