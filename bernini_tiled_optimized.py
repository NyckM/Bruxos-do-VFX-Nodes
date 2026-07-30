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
    from .nodes import (
        BerniniInfinity as _BERNINI,
        _mask_bbox as _bx_mask_bbox,
        _align_up_4n1 as _bx_align_up_4n1,
        _encode_video as _bx_encode_video,
        _collect_reference_latents as _bx_collect_ref,
        _clone_conditioning_set_values as _bx_clone_cond,
        _make_empty_latent as _bx_make_latent,
        _decode_video as _bx_decode_video,
        _resize_source_video as _bx_resize_source,
        _normalize_mask as _bx_norm_mask_fn,
        _grow_blur_mask as _bx_grow_blur,
        _rect_feather_mask as _bx_rect_feather,
        _mem_cleanup as _bx_mem_cleanup,
        _mirror_pad_frames as _bx_mirror_pad,
        _mask_to_latent as _bx_mask_to_latent,
        _lat_len as _bx_lat_len,
        BasicScheduler as _bx_BasicScheduler,
        KSamplerSelect as _bx_KSamplerSelect,
        SplitSigmas as _bx_SplitSigmas,
    )
except Exception:
    try:
        from nodes import (
            BerniniInfinity as _BERNINI,
            _mask_bbox as _bx_mask_bbox,
            _align_up_4n1 as _bx_align_up_4n1,
            _encode_video as _bx_encode_video,
            _collect_reference_latents as _bx_collect_ref,
            _clone_conditioning_set_values as _bx_clone_cond,
            _make_empty_latent as _bx_make_latent,
            _decode_video as _bx_decode_video,
            _resize_source_video as _bx_resize_source,
            _normalize_mask as _bx_norm_mask_fn,
            _grow_blur_mask as _bx_grow_blur,
            _rect_feather_mask as _bx_rect_feather,
            _mem_cleanup as _bx_mem_cleanup,
            _mirror_pad_frames as _bx_mirror_pad,
            _mask_to_latent as _bx_mask_to_latent,
            _lat_len as _bx_lat_len,
            BasicScheduler as _bx_BasicScheduler,
            KSamplerSelect as _bx_KSamplerSelect,
            SplitSigmas as _bx_SplitSigmas,
        )
    except Exception:
        _BERNINI = None
        _bx_mask_bbox = _bx_align_up_4n1 = _bx_encode_video = _bx_collect_ref = None
        _bx_clone_cond = _bx_make_latent = _bx_decode_video = _bx_resize_source = None
        _bx_norm_mask_fn = _bx_grow_blur = _bx_rect_feather = _bx_mem_cleanup = None
        _bx_mirror_pad = _bx_BasicScheduler = _bx_KSamplerSelect = _bx_SplitSigmas = None
        _bx_mask_to_latent = _bx_lat_len = None

try:
    import comfy.samplers as _cs
    _SAMPLERS = list(getattr(_cs, "SAMPLER_NAMES", ["res_multistep", "euler"]))
    _SCHEDULERS = list(getattr(_cs, "SCHEDULER_NAMES", ["simple"]))
except Exception:
    _SAMPLERS = ["res_multistep", "euler"]
    _SCHEDULERS = ["simple"]

CAT = "Bruxos do VFX/Tiles"


# ----------------------------------------------------------------------------
# Flash/Sage Attention: NAO da pra ligar por dentro deste node -- o backend de
# atencao e escolhido pelo ComfyUI no LAUNCH (flag de linha de comando) e vale
# pro processo inteiro, nao por node/grafo. O que da pra fazer aqui e checar
# o que esta ativo agora e avisar como ligar, sem forcar nada (evita efeito
# colateral global vindo de um unico node).
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
        elif getattr(a, "use_split_cross_attention", False):
            info["active_flag"] = "split (legado, lento)"
        elif getattr(a, "use_quad_cross_attention", False):
            info["active_flag"] = "quad (legado, lento)"
    except Exception:
        pass
    return info


_BX_ATTN_LOGGED = {"done": False}


def _bx_log_attention_backend_once():
    """Imprime 1x por sessao o backend de atencao ativo e como trocar. So
    diagnostico -- nunca muda nada, pra nao ter efeito colateral global vindo
    de um node de tiling."""
    if _BX_ATTN_LOGGED["done"]:
        return
    _BX_ATTN_LOGGED["done"] = True
    try:
        info = _bx_attention_backend_info()
        active = info["active_flag"] or "padrao do ComfyUI (provavelmente pytorch/sdpa)"
        print(f"[Bernini Tiled][attention] backend ativo: {active} | "
              f"sageattention instalado: {'sim' if info['sage_installed'] else 'nao'} | "
              f"flash-attn instalado: {'sim' if info['flash_installed'] else 'nao'}", flush=True)
        if info["active_flag"] is None:
            tips = []
            if info["sage_installed"]:
                tips.append("pacote sageattention ja esta instalado -> reinicie o ComfyUI com "
                             "--use-sage-attention pra ativar (mais rapido, sem mexer no node)")
            else:
                tips.append("instale 'pip install sageattention' e reinicie o ComfyUI com "
                             "--use-sage-attention")
            print(f"[Bernini Tiled][attention] {'; '.join(tips)}. Acelera a ATENCAO INTEIRA do "
                  f"Wan (sem token pruning), efeito automatico em todo passe/ladrilho -- "
                  f"nao precisa mudar nada neste node.", flush=True)
    except Exception:
        pass


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


class BruxosBerniniInfinityTiledOptimized:
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
                "mask_mode": (["off", "inpaint", "bbox"], {"default": "off", "tooltip": "off = modifica o shot todo. inpaint = so a area da mascara muda (renderiza o tile completo). bbox = DUPLO RECORTE: dentro do tile, recorta ainda na bounding box da mascara e roda so essa area (mais rapido se o objeto e pequeno relativo ao tile). Com pular_tiles_vazios, ladrilhos sem mascara nem renderizam."}), 
                "costura_viva": ("BOOLEAN", {"default": True, "tooltip": "LIGADO: cada ladrilho recebe o resultado ja gerado dos vizinhos como CONTEXTO na sobreposicao, mas a mascara original continua ativa. Assim nao sobra parte do objeto na costura. DESLIGADO: ladrilhos independentes (so o fade disfarca)."}),
                "pular_tiles_vazios": ("BOOLEAN", {"default": True, "tooltip": "[inpaint/bbox] Ladrilhos onde a mascara nao toca saem direto da fonte, sem renderizar. Remocao em shot grande fica MAIS RAPIDA."}),
                "vary_seed_per_tile": ("BOOLEAN", {"default": False, "tooltip": "DESLIGADO (recomendado para remocao/modificacao): todos os ladrilhos usam a MESMA seed -> estilo consistente entre tiles. LIGADO: cada tile usa seed+N (evita padrao repetido em geracao pura T2V, mas pode criar inconsistencia visual entre tiles em V2V)."}),
                "bbox_compose": (["rectangle", "silhouette"], {"default": "rectangle", "tooltip": "[bbox] Como colar de volta o resultado do bbox. rectangle = retangulo inteiro com feather nas bordas (sem linha de contorno, recomendado). silhouette = usa o contorno da mascara como alpha."}),
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
                "latent_encode_once": ("BOOLEAN", {"default": True, "tooltip": "[experimental] No modo off, sem costura viva e sem janelas temporais, codifica o video inteiro no VAE uma vez e recorta o latent por tile. Evita encode repetido; o decode ainda e por tile."}),
                "decode_once": ("BOOLEAN", {"default": False, "tooltip": "[experimental] Com latent_encode_once, monta os latents (todo o canvas no modo off; so as regioes de mascara no modo bbox) e decodifica uma unica vez no final, em vez de decodificar por tile/bbox. Requer VRAM adicional no decode. Sem efeito em inpaint/costura_viva (usam decode por tile)."}),
                "agrupar_high_low": ("BOOLEAN", {"default": False, "tooltip": "[experimental] Com latent_encode_once (off/bbox, sem costura_viva): roda o passo HIGH de TODOS os ladrilhos primeiro (high_model fica residente na VRAM), depois o passo LOW de todos (troca de modelo 1x em vez de 1x por ladrilho -- menos re-stage no log). Custo: guarda os latentes intermediarios de TODOS os ladrilhos ao mesmo tempo entre as duas fases (mais RAM/VRAM). So ligue se sobrar memoria; senao mantenha desligado (intercalado por ladrilho, como hoje)."}),
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

    def _prepare_full_latent(self, src, vae, T):
        """Codifica uma vez a fonte espacial completa, alinhada em 4n+1.

        Este caminho e seguro quando os tiles sao independentes. No BBox ele
        tambem e valido: a mascara so decide onde o resultado sera composto,
        enquanto o condicionamento do modelo continua sendo um recorte da mesma
        fonte imutavel.
        """
        if any(x is None for x in (_bx_encode_video, _bx_mirror_pad)):
            raise RuntimeError("helpers de latent do Bernini indisponiveis")
        aligned = _bx_align_up_4n1(int(T))
        raw = src if aligned == int(T) else _bx_mirror_pad(src, aligned)
        latent = _bx_encode_video(vae, raw)
        print(f"[Bernini Tiled][latent-once] fonte codificada uma vez: {tuple(latent.shape)}", flush=True)
        return latent, aligned

    def _prep_high_pass(self, source_latent, tile, tw, th, aligned, vae,
                        positive, negative, high_model, low_model,
                        seed, steps, split_step, cfg, sampler_name, scheduler, denoise,
                        reference_video, kwargs):
        """Fase 1 (HIGH) de um tile a partir de um crop do source_latent ja
        codificado. Devolve tudo que a fase 2 (LOW) precisa pra terminar --
        NAO toca no low_model, entao varios tiles podem passar por aqui em
        sequencia com o high_model residente na VRAM o tempo todo (em vez de
        trocar high<->low a cada tile)."""
        if any(x is None for x in (_bx_clone_cond, _bx_make_latent,
                                   _bx_collect_ref, _bx_BasicScheduler,
                                   _bx_KSamplerSelect, _bx_SplitSigmas)):
            raise RuntimeError("helpers de sampler do Bernini indisponiveis")
        # Wan VAE usa razao espacial 8x; _plan garante coordenadas mult. de 16.
        lx0, ly0 = int(tile["x0"]) // 8, int(tile["y0"]) // 8
        lw, lh = int(tw) // 8, int(th) // 8
        encoded_tile = source_latent[:, :, :, ly0:ly0 + lh, lx0:lx0 + lw].contiguous()
        if int(encoded_tile.shape[-2]) != lh or int(encoded_tile.shape[-1]) != lw:
            raise RuntimeError("crop latente fora do canvas; usando caminho legado")

        refs = {k: v for k, v in kwargs.items()
                if k.startswith("reference_images.reference_image_") and v is not None}
        context_latents = [encoded_tile]
        context_latents.extend(_bx_collect_ref(
            vae, int(aligned), 848, reference_video=reference_video,
            reference_images=refs,
        ))
        values = {"context_latents": context_latents}
        pos = _bx_clone_cond(positive, values)
        neg = _bx_clone_cond(negative, values)
        sampler = _bx_KSamplerSelect.execute(sampler_name).args[0]
        # BasicScheduler so LE metadados de sampling do modelo (shift/schedule),
        # nao roda forward nem exige residencia em VRAM -- mesmo model de
        # referencia do codigo original (low_model), sem custo de troca real.
        sigmas = _bx_BasicScheduler.execute(low_model, scheduler, int(steps), float(denoise)).args[0]
        high_sigmas, low_sigmas = _bx_SplitSigmas.execute(sigmas, int(split_step)).args
        bern = _BERNINI()
        bern._g_mode = "off"
        latent = {"samples": _bx_make_latent(int(aligned), int(tw), int(th), 1)}
        # IMPORTANTE: high_model precisa chegar aqui JA CLONADO por quem chamou.
        # Cada .clone() de um ModelPatcher com muitos patches de LoRA forca o
        # ComfyUI a re-aplicar os patches (re-stage caro); clonar 1x por FASE
        # (nao por tile) e o que faz o agrupar_high_low economizar de verdade.
        high = bern._sample_pass(high_model, True, int(seed), float(cfg), pos, neg,
                                 sampler, high_sigmas, latent)
        return {"pos": pos, "neg": neg, "sampler": sampler, "low_sigmas": low_sigmas, "high": high}

    def _finish_low_pass(self, item, low_model, vae, cfg, T, return_latent=False):
        """Fase 2 (LOW) a partir do que _prep_high_pass devolveu. So aqui o
        low_model e tocado -- pode rodar pra varios tiles em sequencia com o
        low_model residente o tempo todo."""
        bern = _BERNINI()
        bern._g_mode = "off"
        # mesmo motivo do high: low_model precisa chegar ja clonado por quem chamou.
        low = bern._sample_pass(low_model, False, 0, float(cfg), item["pos"], item["neg"],
                                item["sampler"], item["low_sigmas"], item["high"])
        if return_latent:
            return low["samples"].cpu()
        imgs = _bx_decode_video(vae, low["samples"], False).float().clamp(0, 1)
        if int(imgs.shape[0]) > int(T):
            imgs = imgs[:int(T)]
        elif int(imgs.shape[0]) < int(T):
            imgs = torch.cat([imgs, imgs[-1:].repeat(int(T) - int(imgs.shape[0]), 1, 1, 1)], dim=0)
        return imgs.cpu()

    def _render_from_latent_tile(self, source_latent, tile, tw, th, T, aligned,
                                 positive, negative, high_model, low_model, vae,
                                 seed, steps, split_step, cfg, sampler_name, scheduler,
                                 denoise, reference_video, kwargs, limpar_vram,
                                 monitor_memoria, return_latent=False):
        """Amostra um tile usando um crop do source_latent ja codificado
        (HIGH e LOW sequenciais, um tile de cada vez). Caminho padrao -- usa
        os mesmos _prep_high_pass/_finish_low_pass do modo agrupado, so que
        sem separar as duas fases entre tiles."""
        item = self._prep_high_pass(source_latent, tile, tw, th, aligned, vae,
                                    positive, negative, high_model.clone(), low_model,
                                    seed, steps, split_step, cfg, sampler_name, scheduler, denoise,
                                    reference_video, kwargs)
        _bx_mem_cleanup(limpar_vram, model=high_model, between_passes=True)
        return self._finish_low_pass(item, low_model.clone(), vae, cfg, T, return_latent=return_latent)

    def _render_bbox_tile(self, src_tile, m_tile, tw, th, T,
                          positive, negative, high_model, low_model, vae,
                          seed, steps, split_step, cfg, sampler_name, scheduler, denoise,
                          chunk_size, overlap_frames, mask_grow, mask_blur,
                          limpar_vram, monitor_memoria, bbox_compose,
                          reference_video, source_latent=None, aligned_latent=None,
                          tile_origin=None, return_latent_contrib=False,
                          defer_low_pass=False, **kwargs):
        """Roda o Bernini no bbox da mascara DENTRO do tile — duplo recorte.

        Para evitar borda de cor/textura, o modelo recebe o bbox EXPANDIDO com
        uma janela de contexto do fundo ao redor (ctx_pad). O Bernini gera nessa
        area maior (ve o fundo externo e casa a textura), mas na composicao final
        so colamos de volta a regiao da MASCARA — o fundo externo e descartado.
        """
        if _bx_mask_bbox is None:
            raise RuntimeError("[Bernini Tiled][bbox] helpers do nodes.py nao importaram.")

        # bbox da mascara dentro do tile (em pixels do tile)
        x0, y0, x1, y1 = _bx_mask_bbox(m_tile, int(mask_grow), 16, tw, th)
        cw, ch = x1 - x0, y1 - y0
        area_pct = 100.0 * (cw * ch) / (tw * th)

        # janela de CONTEXTO ao redor do bbox: o modelo ve o fundo externo e
        # casa a textura/cor — elimina a borda. ctx_pad = metade do tile_overlap
        # (ja existe sobreposicao com vizinhos, reusar a mesma logica faz sentido).
        ctx_pad = max(64, int(mask_blur) * 4)   # pelo menos 64px de contexto
        cx0 = max(0,  x0 - ctx_pad)
        cy0 = max(0,  y0 - ctx_pad)
        cx1 = min(tw, x1 + ctx_pad)
        cy1 = min(th, y1 + ctx_pad)
        # alinha ao multiplo de 16
        cx0 = (cx0 // 16) * 16
        cy0 = (cy0 // 16) * 16
        cx1 = min(tw, -(-cx1 // 16) * 16)
        cy1 = min(th, -(-cy1 // 16) * 16)
        ccw, cch = cx1 - cx0, cy1 - cy0

        print(f"[Bernini Tiled][bbox] mascara: ({x0},{y0})-({x1},{y1}) {cw}x{ch} "
              f"(~{area_pct:.0f}% do tile) | contexto: ({cx0},{cy0})-({cx1},{cy1}) "
              f"{ccw}x{cch} (ctx_pad={ctx_pad}px)", flush=True)

        # fonte e mascara na janela de CONTEXTO (maior que o bbox puro)
        src_ctx  = src_tile[:, cy0:cy1, cx0:cx1, :].contiguous()
        # IMPORTANTE: o grow precisa atingir a MASCARA entregue ao inpaint,
        # nao somente o retangulo. A dilatacao retangular e separavel: dois
        # pools 1D sao matematicamente equivalentes a um pool (2g+1)^2, mas
        # evitam centenas de segundos de CPU num video inteiro.
        g = int(mask_grow)
        xmask = m_tile.float().unsqueeze(1)
        if g > 0:
            k = 2 * g + 1
            xmask = torch.nn.functional.max_pool2d(xmask, (1, k), stride=1, padding=(0, g))
            xmask = torch.nn.functional.max_pool2d(xmask, (k, 1), stride=1, padding=(g, 0))
        elif g < 0:
            k = 2 * (-g) + 1
            xmask = -torch.nn.functional.max_pool2d(-xmask, (1, k), stride=1, padding=(0, -g))
            xmask = -torch.nn.functional.max_pool2d(xmask, (k, 1), stride=1, padding=(-g, 0))
        # O blur fica no Bernini para preservar uma borda suave controlada.
        m_effective = xmask.squeeze(1).clamp(0, 1)
        # mascara efetiva na janela (sem ativar o contexto externo)
        m_ctx = torch.zeros((T, cch, ccw), dtype=torch.float32)
        # posicao da mascara dentro da janela de contexto
        my0, my1 = y0 - cy0, y1 - cy0
        mx0, mx1 = x0 - cx0, x1 - cx0
        m_ctx[:, my0:my1, mx0:mx1] = m_effective[:, y0:y1, x0:x1]

        # O BBox so usa a mascara na COMPOSICAO, nao como token de entrada no
        # Wan. Assim podemos recortar o latent global ja codificado e eliminar
        # os N encodes de VAE sem mudar a amostragem ou a borda composta.
        use_latent_ctx = (source_latent is not None and aligned_latent is not None
                          and tile_origin is not None)

        if defer_low_pass:
            # ---- AGRUPAR HIGH/LOW: so a fase HIGH deste tile. Devolve um
            # "embrulho" com tudo que _finish_bbox_tile precisa pra terminar
            # (LOW + composicao) depois que TODOS os tiles ja passaram pelo
            # HIGH -- assim o high_model fica residente pra grade inteira, e
            # so troca pro low_model uma vez (nao 1x por tile).
            if not use_latent_ctx:
                raise RuntimeError("[Bernini Tiled][bbox][agrupar] requer latent_encode_once ativo.")
            ctx_tile = {"x0": int(tile_origin["x0"]) + cx0,
                        "y0": int(tile_origin["y0"]) + cy0}
            high_item = self._prep_high_pass(
                source_latent, ctx_tile, ccw, cch, aligned_latent, vae,
                positive, negative, high_model, low_model,
                seed, steps, split_step, cfg, sampler_name, scheduler, denoise,
                reference_video, kwargs,
            )
            return {
                "high_item": high_item,
                "geometry": dict(x0=x0, y0=y0, x1=x1, y1=y1, cw=cw, ch=ch,
                                 my0=my0, my1=my1, mx0=mx0, mx1=mx1,
                                 tile_origin=dict(tile_origin)),
                "m_tile": m_tile, "src_tile": src_tile, "bbox_compose": bbox_compose,
                "mask_blur": int(mask_blur), "aligned_latent": int(aligned_latent),
                "return_latent_contrib": bool(return_latent_contrib),
            }

        if return_latent_contrib:
            # ---- DECODE-ONCE: devolve a CONTRIBUICAO EM LATENTE (sem decodificar
            # este tile). O chamador acumula num canvas latente global e decodifica
            # tudo de uma vez so no final. So a area do BBOX entra (o contexto ao
            # redor e descartado aqui tambem, exatamente como no caminho em pixel).
            if not use_latent_ctx:
                raise RuntimeError("[Bernini Tiled][bbox][decode-once] requer latent_encode_once ativo.")
            ctx_tile = {"x0": int(tile_origin["x0"]) + cx0,
                        "y0": int(tile_origin["y0"]) + cy0}
            ctx_latent = self._render_from_latent_tile(
                source_latent, ctx_tile, ccw, cch, T, aligned_latent,
                positive, negative, high_model, low_model, vae,
                seed, steps, split_step, cfg, sampler_name, scheduler, denoise,
                reference_video, kwargs, limpar_vram, monitor_memoria,
                return_latent=True,
            )  # [1,16,Tlat,cch/8,ccw/8] cpu -- NENHUM decode aqui.
            print("[Bernini Tiled][bbox][decode-once] contexto amostrado em latente "
                  "(decode adiado pro canvas global).", flush=True)

            # bbox e janela de contexto sao multiplos de 16 (stride do _mask_bbox e
            # do alinhamento acima) -> dividir por 8 (razao espacial do VAE Wan) e
            # sempre exato, sem arredondamento.
            lat_ch, lat_cw = ch // 8, cw // 8
            lat_my0, lat_mx0 = my0 // 8, mx0 // 8
            contrib = ctx_latent[:, :, :, lat_my0:lat_my0 + lat_ch, lat_mx0:lat_mx0 + lat_cw].contiguous()
            Tlat = int(contrib.shape[2])

            if bbox_compose == "rectangle":
                # feather retangular: espacial-so (igual em todo frame), reduz
                # direto pra resolucao latente -- sem precisar reamostrar no tempo.
                feather_lat = max(1, int(mask_blur) // 8)
                blend2d = _bx_rect_feather(1, lat_ch, lat_cw, feather_lat)
                blend = blend2d.view(1, 1, 1, lat_ch, lat_cw).expand(1, 1, Tlat, lat_ch, lat_cw).contiguous()
            else:
                # silhueta: a mascara real varia por frame -> reduz certo no tempo
                # (max por bloco de 4, igual ao proprio VAE) e no espaco (/8).
                m_orig_px = m_tile[:, y0:y1, x0:x1]
                if int(m_orig_px.shape[0]) != int(aligned_latent):
                    m_orig_px = _bx_mirror_pad(m_orig_px, int(aligned_latent))
                blend = _bx_mask_to_latent(m_orig_px, Tlat, lat_ch, lat_cw, contrib.device, contrib.dtype)
            blend = blend.to(contrib.dtype)

            lx0 = (int(tile_origin["x0"]) + x0) // 8
            ly0 = (int(tile_origin["y0"]) + y0) // 8
            return {"contrib": contrib * blend, "weight": blend, "lx0": lx0, "ly0": ly0}

        if use_latent_ctx:
            ctx_tile = {"x0": int(tile_origin["x0"]) + cx0,
                        "y0": int(tile_origin["y0"]) + cy0}
            ctx_imgs = self._render_from_latent_tile(
                source_latent, ctx_tile, ccw, cch, T, aligned_latent,
                positive, negative, high_model, low_model, vae,
                seed, steps, split_step, cfg, sampler_name, scheduler, denoise,
                reference_video, kwargs, limpar_vram, monitor_memoria,
            ).float().clamp(0, 1)
            print("[Bernini Tiled][bbox][latent-once] contexto recortado do latent global.", flush=True)
        else:
            # Fallback legado: codifica somente a janela de contexto deste tile.
            bern = _BERNINI()
            imgs, _, _ = bern.render(
                positive=positive, negative=negative,
                high_model=high_model, low_model=low_model, vae=vae,
                source_video=src_ctx, width=ccw, height=cch,
                seed=int(seed), steps=int(steps), split_step=int(split_step),
                cfg=float(cfg), sampler_name=sampler_name, scheduler=scheduler,
                denoise=float(denoise),
                chunk_size=int(chunk_size), overlap=int(overlap_frames),
                max_frames=0, tail_memory=True, tail_frames=5,
                decode_tiled=False, decode_chunk=0, vary_seed_per_chunk=False,
                ref_max_size=848, mode="context_window", context_jitter=True,
                mask_mode="inpaint", mask_grow=0, mask_blur=int(mask_blur),
                mask_pad=0, bbox_compose="rectangle", resize_mode="stretch",
                limpar_vram=limpar_vram, monitor_memoria=bool(monitor_memoria),
                guidance_mode="off",
                region_mask=m_ctx, reference_video=reference_video,
                **kwargs,
            )
            ctx_imgs = imgs.float().clamp(0, 1)
        n = min(T, int(ctx_imgs.shape[0]))

        # cola de volta NO TILE: so a area da MASCARA (nao o contexto externo)
        # o fundo externo gerado e descartado — so serve pra guiar a cor/textura
        out = src_tile.clone()
        m_orig = m_tile[:n, y0:y1, x0:x1]   # mascara original (sem grow)
        if bbox_compose == "rectangle":
            blend = _bx_rect_feather(n, ch, cw, int(mask_blur)).unsqueeze(-1)
        else:
            blend = m_orig.unsqueeze(-1)
        # extrai so a area da mascara do resultado do contexto
        result_crop = ctx_imgs[:n, my0:my1, mx0:mx1, :]
        region = out[:n, y0:y1, x0:x1, :]
        out[:n, y0:y1, x0:x1, :] = region * (1.0 - blend) + result_crop * blend
        return out.cpu()

    def _finish_bbox_tile(self, bundle, low_model, vae, cfg, T):
        """Fase 2 (LOW + composicao) de um tile BBOX cujo HIGH ja rodou via
        _render_bbox_tile(..., defer_low_pass=True). So aqui o low_model e
        tocado."""
        g = bundle["geometry"]
        x0, y0, x1, y1 = g["x0"], g["y0"], g["x1"], g["y1"]
        cw, ch = g["cw"], g["ch"]
        my0, my1, mx0, mx1 = g["my0"], g["my1"], g["mx0"], g["mx1"]
        tile_origin = g["tile_origin"]
        bbox_compose = bundle["bbox_compose"]
        mask_blur = bundle["mask_blur"]
        m_tile = bundle["m_tile"]

        if bundle["return_latent_contrib"]:
            ctx_latent = self._finish_low_pass(bundle["high_item"], low_model, vae, cfg, T,
                                               return_latent=True)
            lat_ch, lat_cw = ch // 8, cw // 8
            lat_my0, lat_mx0 = my0 // 8, mx0 // 8
            contrib = ctx_latent[:, :, :, lat_my0:lat_my0 + lat_ch, lat_mx0:lat_mx0 + lat_cw].contiguous()
            Tlat = int(contrib.shape[2])

            if bbox_compose == "rectangle":
                feather_lat = max(1, int(mask_blur) // 8)
                blend2d = _bx_rect_feather(1, lat_ch, lat_cw, feather_lat)
                blend = blend2d.view(1, 1, 1, lat_ch, lat_cw).expand(1, 1, Tlat, lat_ch, lat_cw).contiguous()
            else:
                m_orig_px = m_tile[:, y0:y1, x0:x1]
                if int(m_orig_px.shape[0]) != int(bundle["aligned_latent"]):
                    m_orig_px = _bx_mirror_pad(m_orig_px, int(bundle["aligned_latent"]))
                blend = _bx_mask_to_latent(m_orig_px, Tlat, lat_ch, lat_cw, contrib.device, contrib.dtype)
            blend = blend.to(contrib.dtype)

            lx0 = (int(tile_origin["x0"]) + x0) // 8
            ly0 = (int(tile_origin["y0"]) + y0) // 8
            return {"contrib": contrib * blend, "weight": blend, "lx0": lx0, "ly0": ly0}

        ctx_imgs = self._finish_low_pass(bundle["high_item"], low_model, vae, cfg, T,
                                         return_latent=False)
        n = min(T, int(ctx_imgs.shape[0]))
        out = bundle["src_tile"].clone()
        m_orig = m_tile[:n, y0:y1, x0:x1]
        if bbox_compose == "rectangle":
            blend = _bx_rect_feather(n, ch, cw, int(mask_blur)).unsqueeze(-1)
        else:
            blend = m_orig.unsqueeze(-1)
        result_crop = ctx_imgs[:n, my0:my1, mx0:mx1, :]
        region = out[:n, y0:y1, x0:x1, :]
        out[:n, y0:y1, x0:x1, :] = region * (1.0 - blend) + result_crop * blend
        return out.cpu()

    def _render_tiles_batched(self, tiles, mask_mode, umask, src, T,
                              positive, negative, high_model, low_model, vae,
                              seed, steps, split_step, cfg, sampler_name, scheduler, denoise,
                              vary_seed_per_tile, source_latent, aligned_latent,
                              pular_tiles_vazios, bbox_compose, mask_grow, mask_blur,
                              chunk_size, overlap_frames, limpar_vram, monitor_memoria,
                              reference_video, use_decode_once, outs, latent_outs,
                              lacc, lwsum, kwargs):
        """Agrupa TODOS os passos HIGH primeiro (high_model residente pra grade
        inteira), depois TODOS os LOW (troca de modelo uma unica vez em vez de
        uma vez por ladrilho). Preenche outs[]/latent_outs[]/lacc/lwsum com
        exatamente o mesmo formato que o loop intercalado -- a montagem final
        (fade em pixel ou decode-once) e reaproveitada sem mudanca."""
        n_tiles = len(tiles)
        pending = [None] * n_tiles
        rendered, skipped = 0, 0
        log = []

        # ---- fase 0: ladrilhos vazios (bbox) saem direto da fonte, sem HIGH nem LOW ----
        for i, t in enumerate(tiles):
            x0, y0, x1, y1 = t["x0"], t["y0"], t["x1"], t["y1"]
            if (mask_mode == "bbox" and pular_tiles_vazios
                    and float(umask[:, y0:y1, x0:x1].max()) < 0.02):
                outs[i] = src[:, y0:y1, x0:x1, :]
                skipped += 1
                log.append(f"tile {i + 1}/{n_tiles} (L{t['r']}C{t['c']}): PULADO (mascara vazia)")
                print(f"[Bernini Tiled] {log[-1]}", flush=True)

        # ---- fase 1: HIGH de todos os ladrilhos pendentes (high_model residente) ----
        # Clona UMA VEZ pra fase inteira: reusar o MESMO ModelPatcher clonado
        # em todos os tiles evita que o ComfyUI reaplique os patches de LoRA
        # (caro, o log mostrava "892 patches attached" a cada tile) -- clonar
        # de novo por tile e o motivo do agrupamento nao render tempo antes.
        high_clone = high_model.clone()
        t_high0 = time.time()
        for i, t in enumerate(tiles):
            if outs[i] is not None:
                continue
            x0, y0, x1, y1 = t["x0"], t["y0"], t["x1"], t["y1"]
            tw, th = x1 - x0, y1 - y0
            src_tile = src[:, y0:y1, x0:x1, :]
            seed_i = int(seed) + (i if vary_seed_per_tile else 0)
            if mask_mode == "bbox":
                m_tile = umask[:, y0:y1, x0:x1]
                bundle = self._render_bbox_tile(
                    src_tile, m_tile, tw, th, T,
                    positive, negative, high_clone, low_model, vae,
                    seed_i, steps, split_step, cfg, sampler_name, scheduler, denoise,
                    chunk_size, overlap_frames, mask_grow, mask_blur,
                    limpar_vram, monitor_memoria, bbox_compose,
                    reference_video, source_latent=source_latent, aligned_latent=aligned_latent,
                    tile_origin=t, return_latent_contrib=use_decode_once,
                    defer_low_pass=True, **kwargs,
                )
                pending[i] = {"kind": "bbox", "bundle": bundle}
            else:
                high_item = self._prep_high_pass(
                    source_latent, t, tw, th, aligned_latent, vae,
                    positive, negative, high_clone, low_model,
                    seed_i, steps, split_step, cfg, sampler_name, scheduler, denoise,
                    reference_video, kwargs,
                )
                pending[i] = {"kind": "off", "item": high_item}
            print(f"[Bernini Tiled][agrupar] tile {i + 1}/{n_tiles} (L{t['r']}C{t['c']}): HIGH ok", flush=True)

        del high_clone
        _bx_mem_cleanup(limpar_vram, model=high_model, between_passes=True)
        if monitor_memoria:
            print(f"[Bernini Tiled][agrupar] fase HIGH completa em {time.time() - t_high0:.0f}s "
                  f"({sum(1 for p in pending if p is not None)} ladrilho(s)); "
                  f"trocando pro low_model uma unica vez.", flush=True)

        # ---- fase 2: LOW de todos os ladrilhos pendentes (low_model residente) ----
        # mesmo raciocinio do high: 1 clone pra fase inteira, nao 1 por tile.
        low_clone = low_model.clone()
        t_low0 = time.time()
        for i, t in enumerate(tiles):
            p = pending[i]
            if p is None:
                continue
            tt0 = time.time()
            if p["kind"] == "bbox":
                result = self._finish_bbox_tile(p["bundle"], low_clone, vae, cfg, T)
                if use_decode_once:
                    lh, lw = result["weight"].shape[-2], result["weight"].shape[-1]
                    lx0, ly0 = result["lx0"], result["ly0"]
                    lacc[:, :, :, ly0:ly0 + lh, lx0:lx0 + lw] += result["contrib"]
                    lwsum[:, :, :, ly0:ly0 + lh, lx0:lx0 + lw] += result["weight"]
                else:
                    outs[i] = result
            else:
                out = self._finish_low_pass(p["item"], low_clone, vae, cfg, T,
                                            return_latent=use_decode_once)
                if use_decode_once:
                    latent_outs[i] = out
                else:
                    outs[i] = out
            rendered += 1
            dt = time.time() - tt0
            log.append(f"tile {i + 1}/{n_tiles} (L{t['r']}C{t['c']}): {p['kind']} ok em {dt:.0f}s (agrupado)")
            print(f"[Bernini Tiled] {log[-1]}", flush=True)

        del low_clone
        _bx_mem_cleanup(limpar_vram, model=low_model, between_passes=True)
        if monitor_memoria:
            print(f"[Bernini Tiled][agrupar] fase LOW completa em {time.time() - t_low0:.0f}s.", flush=True)

        return rendered, skipped, log

    # ------------------------------------------------------------------ main
    def render_tiled(self, positive, negative, high_model, low_model, vae, source_video,
                     width, height, tile_count_width, tile_count_height, tile_overlap,
                     seed, steps, split_step, cfg, sampler_name, scheduler, denoise,
                     mask_mode, costura_viva, pular_tiles_vazios, vary_seed_per_tile=False,
                     region_mask=None, reference_video=None,
                     mode="context_window", chunk_size=121, overlap_frames=8,
                     mask_grow=20, mask_blur=6, bbox_compose="rectangle",
                     limpar_vram="leve", monitor_memoria=False,
                     latent_encode_once=True, decode_once=False,
                     agrupar_high_low=False, **kwargs):
        if not _HAS_TORCH:
            raise RuntimeError("[Bernini Tiled] torch indisponivel.")
        if _BERNINI is None:
            raise RuntimeError("[Bernini Tiled] nao achei o Bernini Infinity no pacote (nodes.py). "
                               "Este node roda o Bernini por ladrilho; instale o pacote completo.")
        _bx_log_attention_backend_once()

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

        # BBox tambem pode reutilizar o latent: a mascara e aplicada somente na
        # composicao final e a fonte que condiciona os tiles permanece imutavel.
        use_latent_once = bool(latent_encode_once and mask_mode in ("off", "bbox") and not costura_viva
                               and mode == "context_window" and int(chunk_size) >= T)
        source_latent = aligned_latent = None
        if use_latent_once:
            try:
                source_latent, aligned_latent = self._prepare_full_latent(src, vae, T)
            except Exception as e:
                use_latent_once = False
                print(f"[Bernini Tiled][latent-once] fallback legado: {e}", flush=True)
        use_decode_once = bool(decode_once and use_latent_once and mask_mode in ("off", "bbox"))
        latent_outs = [None] * n_tiles if (use_decode_once and mask_mode == "off") else None
        lacc = lwsum = None
        if use_decode_once:
            base_dtype = source_latent.dtype
            base_shape = tuple(source_latent.shape)
            lacc = torch.zeros(base_shape, dtype=base_dtype)
            lwsum = torch.zeros((base_shape[0], 1, base_shape[2], base_shape[3], base_shape[4]), dtype=base_dtype)
        if bool(decode_once) and not use_decode_once:
            print("[Bernini Tiled][decode-once] indisponivel neste modo (requer latent_encode_once); "
                  "usando decode por tile.", flush=True)

        # AGRUPAR HIGH/LOW: passo HIGH de TODOS os ladrilhos primeiro (high_model
        # residente pra grade inteira), depois passo LOW de todos (troca de
        # modelo 1x em vez de 1x por ladrilho). So disponivel no mesmo terreno
        # do latent_encode_once (off/bbox, sem costura_viva); guarda os latentes
        # intermediarios de TODOS os ladrilhos entre as duas fases -> so ligue
        # com RAM/VRAM de sobra.
        use_batched = bool(agrupar_high_low and use_latent_once)
        if bool(agrupar_high_low) and not use_batched:
            print("[Bernini Tiled][agrupar-high-low] indisponivel neste modo (requer "
                  "latent_encode_once); rodando high/low intercalado por ladrilho.", flush=True)
        if use_batched:
            rendered, skipped, batched_log = self._render_tiles_batched(
                tiles, mask_mode, umask, src, T,
                positive, negative, high_model, low_model, vae,
                seed, steps, split_step, cfg, sampler_name, scheduler, denoise,
                vary_seed_per_tile, source_latent, aligned_latent,
                pular_tiles_vazios, bbox_compose, mask_grow, mask_blur,
                chunk_size, overlap_frames, limpar_vram, monitor_memoria,
                reference_video, use_decode_once, outs, latent_outs, lacc, lwsum, kwargs,
            )
            log.extend(batched_log)

        for i, t in enumerate(tiles if not use_batched else []):
            x0, y0, x1, y1 = t["x0"], t["y0"], t["x1"], t["y1"]
            # Comeca com views: evita copiar o tile inteiro e uma mascara cheia
            # quando o Bernini so vai ler os tensores (off/bbox sem costura).
            # A copia e feita logo abaixo apenas se a costura viva for escrever.
            src_tile = src[:, y0:y1, x0:x1, :]
            m_tile = umask[:, y0:y1, x0:x1] if mask_mode in ("inpaint", "bbox") else None

            # ---- COSTURA VIVA: vizinhos viram CONTEXTO, mascara continua ativa ----
            if costura_viva:
                # O comportamento antigo zerava a mascara na faixa copiada.
                # Se o tracking/margem do tile anterior nao cobrisse uma parte
                # do objeto, o tile atual era proibido de corrigi-la e ela
                # "vazava" na costura. Mantemos a mascara para ele renderizar a
                # area solicitada e usamos o vizinho apenas como guia visual.
                src_tile = src_tile.clone()
                if m_tile is None:
                    m_tile = torch.ones((T, th, tw), dtype=torch.float32)
                else:
                    m_tile = m_tile.clone()
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

            # ---- pular ladrilho vazio (inpaint/bbox): mascara do usuario nao toca ----
            if (mask_mode in ("inpaint", "bbox") and pular_tiles_vazios
                    and float(umask[:, y0:y1, x0:x1].max()) < 0.02):
                outs[i] = src_tile
                skipped += 1
                log.append(f"tile {i + 1}/{n_tiles} (L{t['r']}C{t['c']}): PULADO (mascara vazia)")
                print(f"[Bernini Tiled] {log[-1]}", flush=True)
                continue

            tt0 = time.time()

            # ---- BBOX: duplo recorte (tile + bbox da mascara dentro do tile) ----
            if mask_mode == "bbox" and umask is not None:
                print(f"[Bernini Tiled] tile {i + 1}/{n_tiles} (L{t['r']}C{t['c']}) em ({x0},{y0}) "
                      f"{tw}x{th} | mask=bbox ...", flush=True)
                result = self._render_bbox_tile(
                    src_tile, m_tile, tw, th, T,
                    positive, negative, high_model, low_model, vae,
                    int(seed) + (i if vary_seed_per_tile else 0), steps, split_step, cfg,
                    sampler_name, scheduler, denoise,
                    chunk_size, overlap_frames, mask_grow, mask_blur,
                    limpar_vram, monitor_memoria, bbox_compose,
                    reference_video, source_latent=source_latent if use_latent_once else None,
                    aligned_latent=aligned_latent, tile_origin=t,
                    return_latent_contrib=use_decode_once, **kwargs,
                )
                if use_decode_once:
                    lh, lw = result["weight"].shape[-2], result["weight"].shape[-1]
                    lx0, ly0 = result["lx0"], result["ly0"]
                    lacc[:, :, :, ly0:ly0 + lh, lx0:lx0 + lw] += result["contrib"]
                    lwsum[:, :, :, ly0:ly0 + lh, lx0:lx0 + lw] += result["weight"]
                else:
                    outs[i] = result
                rendered += 1
                dt = time.time() - tt0
                tag = " [latent]" if use_decode_once else ""
                log.append(f"tile {i + 1}/{n_tiles} (L{t['r']}C{t['c']}): bbox ok em {dt:.0f}s{tag}")
                print(f"[Bernini Tiled] {log[-1]}", flush=True)
                continue

            # ---- renderiza o ladrilho com o Bernini COMPLETO (off / inpaint) ----
            eff_mode = "inpaint" if (mask_mode == "inpaint" or costura_viva) else "off"
            tt0 = time.time()
            print(f"[Bernini Tiled] tile {i + 1}/{n_tiles} (L{t['r']}C{t['c']}) em ({x0},{y0}) "
                  f"{tw}x{th} | mask={eff_mode} ...", flush=True)
            if use_latent_once:
                out = self._render_from_latent_tile(
                    source_latent, t, tw, th, T, aligned_latent,
                    positive, negative, high_model, low_model, vae,
                    int(seed) + (i if vary_seed_per_tile else 0),
                    steps, split_step, cfg, sampler_name, scheduler, denoise,
                    reference_video, kwargs, limpar_vram, monitor_memoria,
                    return_latent=use_decode_once,
                )
                if use_decode_once:
                    latent_outs[i] = out
                else:
                    outs[i] = out
                rendered += 1
                dt = time.time() - tt0
                log.append(f"tile {i + 1}/{n_tiles} (L{t['r']}C{t['c']}): latent-once ok em {dt:.0f}s")
                print(f"[Bernini Tiled] {log[-1]}", flush=True)
                continue
            imgs, _lat, _tf = bern.render(
                positive=positive, negative=negative,
                high_model=high_model, low_model=low_model, vae=vae,
                source_video=src_tile,
                width=tw, height=th,
                seed=int(seed) + (i if vary_seed_per_tile else 0), steps=int(steps), split_step=int(split_step),
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
        if use_decode_once:
            # Os tiles e o canvas sao multiplos de 16; no Wan isso equivale a
            # coordenadas exatas no grid espacial do latent (8 px por celula).
            # lacc/lwsum ja foram alocados antes do loop de ladrilhos (bbox
            # acumula sua contribuicao ali mesmo, tile a tile).
            base = source_latent.cpu()
            if mask_mode == "off":
                # Particao COMPLETA do canvas por ladrilhos com fade nas bordas:
                # os pesos somam ~1 em todo lugar -> normaliza por divisao.
                for i, t in enumerate(tiles):
                    fl = fr = ft = fb = 0
                    for j, nb in enumerate(tiles):
                        if j == i:
                            continue
                        it = _inter(t, nb)
                        if it is None:
                            continue
                        ix0, iy0, ix1, iy1 = it
                        if nb["c"] < t["c"] and iy1 > iy0: fl = max(fl, ix1 - t["x0"])
                        if nb["c"] > t["c"] and iy1 > iy0: fr = max(fr, t["x1"] - ix0)
                        if nb["r"] < t["r"] and ix1 > ix0: ft = max(ft, iy1 - t["y0"])
                        if nb["r"] > t["r"] and ix1 > ix0: fb = max(fb, t["y1"] - iy0)
                    lh, lw = int(latent_outs[i].shape[-2]), int(latent_outs[i].shape[-1])
                    wx = _ramp(lw, fl // 8, fr // 8, "cpu")
                    wy = _ramp(lh, ft // 8, fb // 8, "cpu")
                    wm = (wy.view(lh, 1) * wx.view(1, lw)).view(1, 1, 1, lh, lw).to(base.dtype)
                    lx0, ly0 = int(t["x0"]) // 8, int(t["y0"]) // 8
                    lacc[:, :, :, ly0:ly0 + lh, lx0:lx0 + lw] += latent_outs[i] * wm
                    lwsum[:, :, :, ly0:ly0 + lh, lx0:lx0 + lw] += wm
                merged = lacc / lwsum.clamp(min=1e-6)
            else:
                # BBox: regioes ESPARSAS (so onde teve mascara). Fora delas o
                # peso e 0 -> mantem a fonte original. Onde teve, blend linear
                # igual ao caminho em pixel: fonte*(1-w) + gerado*w. Se dois
                # tiles vizinhos contribuirem pro MESMO ponto (bbox cai na
                # faixa de overlap de ambos), lwsum pode passar de 1 -- nesse
                # caso normaliza pela media antes de aplicar o peso, pra nao
                # superexpor o latente.
                w = lwsum.clamp(0.0, 1.0)
                avg = lacc / lwsum.clamp(min=1e-6)
                merged = base * (1.0 - w) + avg * w
            print(f"[Bernini Tiled][decode-once] decodificando latent global {tuple(merged.shape)}", flush=True)
            final = _bx_decode_video(vae, merged, False).float().clamp(0, 1)[:T, :H, :W, :]
        else:
            # Pixel path legado (inclusive BBox/inpaint): cada tile ja foi decodificado.
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
                f"agrupar_high_low={'on' if use_batched else 'off'} | "
                f"total {total_dt / 60:.1f}min")
        print(f"[Bernini Tiled] DONE: {info}", flush=True)
        return (final, int(T), info)


NODE_CLASS_MAPPINGS = {"BruxosBerniniInfinityTiledOptimized": BruxosBerniniInfinityTiledOptimized}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosBerniniInfinityTiledOptimized": "Bernini Infinity Tiled Optimized (Bruxos)"}
