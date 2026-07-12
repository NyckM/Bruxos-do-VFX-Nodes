# -*- coding: utf-8 -*-
"""
Bruxos do VFX — MoCha (Wan / WanVideoWrapper)
=============================================
Nodes pro modelo MoCha (https://github.com/Orange-3DV-Team/MoCha), que roda pelo
WanVideoWrapper do Kijai (tipos WANVAE / WANVIDIMAGE_EMBEDS).

NAO reimplementa o MoCha: substitui o node `MochaEmbeds` por uma versao com o
padrao Bruxos, e roda o sampler ORIGINAL do wrapper (puxado do registro) quando
voce quer processar em blocos.

O que ganhamos em cima do MochaEmbeds original:

1. FIX DE FRAMES (4n+1).  O original faz `F = (F-1)//4*4 + 1`, que arredonda pra
   BAIXO e DESCARTA frames (111 entra -> 109 sai). Aqui a gente sobe pro proximo
   4n+1 com padding ESPELHADO (ping-pong, sem frame congelado) e devolve o
   `frames_originais` pra voce cortar de volta depois do decode
   (use o "Trim 4n+1 back to N (Bruxos)" que ja existe no pacote).
   Modo `truncar` disponivel pra reproduzir o comportamento antigo.

2. MASCARA DE VERDADE.  Aceita MASK *ou* IMAGE colorida (SAM3 / SCAIL-2 /
   FaceFusion) e ainda dilata (`mask_grow`) e suaviza (`mask_blur`) — o original
   so aceita MASK crua.

3. MEMORIA.  `limpar_vram` (off/leve/agressivo, com o guard de re-stage/re-patch)
   e `monitor_memoria` (RAM+VRAM no console), iguais aos do Bernini Infinity.

4. CRONOMETRO.  Reporta o tempo do encode no console (e nas saidas `info`).

Categoria: Bruxos do VFX/Mocha
"""

import gc
import time
import logging

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from comfy import model_management as mm
except Exception:  # pragma: no cover
    mm = None

try:
    import nodes as _comfy_nodes
except Exception:  # pragma: no cover
    _comfy_nodes = None

# Reaproveita os helpers do pacote (mesma conta do Bernini Infinity).
try:
    from .nodes import (
        _normalize_mask,
        _grow_blur_mask,
        _mirror_pad_frames,
        _align_up_4n1,
        _mem_cleanup,
        _mem_report,
    )
    _HAS_HELPERS = True
except Exception as e:  # pragma: no cover
    logging.info(f"[Bruxos Mocha] helpers do nodes.py indisponiveis ({e}); usando fallback")
    _HAS_HELPERS = False

    def _align_up_4n1(n):
        n = int(n)
        if n < 1:
            return 1
        r = (n - 1) % 4
        return n if r == 0 else n + (4 - r)

    def _mirror_pad_frames(video, target_len):
        cur = int(video.shape[0])
        if cur >= int(target_len):
            return video
        need = int(target_len) - cur
        if cur == 1:
            return torch.cat([video, video[-1:].repeat(need, *([1] * (video.dim() - 1)))], dim=0)
        idx, i, d = [], cur - 2, -1
        while len(idx) < need:
            idx.append(i)
            i += d
            if i < 0:
                i, d = 1, 1
            elif i > cur - 1:
                i, d = cur - 2, -1
        return torch.cat([video, video[idx]], dim=0)

    def _normalize_mask(mask):
        m = mask
        if m.dim() == 4:
            m = m[..., :3].amax(dim=-1)
        elif m.dim() == 2:
            m = m.unsqueeze(0)
        return m.float().clamp(0.0, 1.0)

    def _grow_blur_mask(m, grow=0, blur=0):
        return m

    def _mem_cleanup(level="leve", model=None, between_passes=False):
        if level == "off":
            return
        try:
            gc.collect()
            if mm is not None:
                mm.soft_empty_cache()
        except Exception:
            pass

    def _mem_report(tag):
        pass


CAT = "Bruxos do VFX/Mocha"


def _fmt_t(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{s - 60 * int(s // 60):04.1f}s"


def _registry():
    return getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", {}) if _comfy_nodes else {}


# =============================================================================
# Mocha Embeds (Bruxos) — substitui o MochaEmbeds do wrapper
# =============================================================================
class BruxosMochaEmbeds:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("WANVAE", {"tooltip": "VAE do Wan (saida do WanVideo VAE Loader). MESMO VAE que vai no sampler."}),
                "input_video": ("IMAGE", {"tooltip": "Video de entrada (o footage que sera editado pelo MoCha)."}),
                "mask": ("MASK,IMAGE", {"tooltip": "Mascara da regiao a editar (branco = edita). Aceita MASK ou IMAGE colorida -- pode ligar direto o SAM3 / SCAIL-2 / a face_mask do FaceFusion, sem converter."}),
                "ref1": ("IMAGE", {"tooltip": "Imagem de referencia 1 (a aparencia/identidade que o MoCha vai usar)."}),
            },
            "optional": {
                "ref2": ("IMAGE", {"tooltip": "Imagem de referencia 2 (opcional)."}),
                "mask_frame_mode": (["uniao (todos os frames)", "primeiro frame"], {"default": "uniao (todos os frames)", "tooltip": "O MoCha usa UMA mascara so pro video inteiro (nao e por frame). Se voce ligar uma mascara por frame (SAM3/SCAIL), escolha como reduzir: 'uniao' = soma todos os frames, cobrindo o sujeito onde quer que ele passe (RECOMENDADO p/ sujeito em movimento). 'primeiro frame' = usa so o frame 0 (bom se a regiao e fixa)."}),
                "frame_fix": (["espelhado (4n+1)", "truncar (original)"], {"default": "espelhado (4n+1)", "tooltip": "O Wan VAE so aceita comprimentos 4n+1. 'espelhado' sobe pro proximo 4n+1 com padding em espelho e devolve 'frames_originais' pra voce cortar de volta depois (NAO perde frame). 'truncar' reproduz o node original, que joga fora os frames que sobram (111 -> 109)."}),
                "mask_grow": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1, "tooltip": "Dilata (+) ou contrai (-) a mascara, em pixels. Util pra pegar a borda do sujeito."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, "tooltip": "Suaviza a borda da mascara. OBS: o MoCha binariza a mascara no latente, entao o blur ajuda pouco aqui -- use o grow."}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Encode do VAE em ladrilhos: usa MENOS VRAM, um pouco mais lento. Ligue se estourar a memoria em resolucao alta."}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Descarrega o VAE da VRAM depois do encode (libera memoria pro sampler). Deixe ligado."}),
                "limpar_vram": (["off", "leve", "agressivo"], {"default": "leve", "tooltip": "Limpeza de memoria antes/depois do encode. leve = gc + esvazia cache (recomendado). agressivo = tambem descarrega modelos (cuidado: se o modelo tiver muita LoRA, o guard evita o re-stage caro)."}),
                "monitor_memoria": ("BOOLEAN", {"default": False, "tooltip": "Imprime RAM/VRAM no console (antes e depois do encode). Use pra achar onde a memoria enche."}),
            },
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "INT", "STRING")
    RETURN_NAMES = ("image_embeds", "frames_originais", "info")
    OUTPUT_TOOLTIPS = (
        "Embeds do MoCha. Ligue no WanVideo Sampler (mesmo lugar do MochaEmbeds original).",
        "Quantos frames o video TINHA antes do padding 4n+1. Ligue no 'Trim 4n+1 back to N (Bruxos)' depois do decode pra voltar ao tamanho exato.",
        "Relatorio: frames, padding aplicado, resolucao do latente, seq_len e tempo do encode.",
    )
    FUNCTION = "process"
    CATEGORY = CAT
    DESCRIPTION = (
        "Embeds do MoCha (Bruxos) — versao do MochaEmbeds com o padrao do pacote: fix de frames 4n+1 "
        "por ESPELHO (o original TRUNCA e perde frames), mascara que aceita MASK ou IMAGE colorida "
        "(SAM3/SCAIL/FaceFusion) com grow/blur, tiled VAE pra pouca VRAM, limpeza de memoria e cronometro. "
        "Saida pluga direto no WanVideo Sampler."
    )

    def process(self, vae, input_video, mask, ref1, ref2=None,
                mask_frame_mode="uniao (todos os frames)",
                frame_fix="espelhado (4n+1)", mask_grow=0, mask_blur=0,
                tiled_vae=False, force_offload=True,
                limpar_vram="leve", monitor_memoria=False):
        if torch is None or mm is None:
            raise RuntimeError("[Bruxos Mocha] torch/comfy indisponiveis.")

        t0 = time.time()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if monitor_memoria:
            _mem_report("mocha: antes do encode")

        H, W = int(input_video.shape[1]), int(input_video.shape[2])
        F_in = int(input_video.shape[0])

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        # ---- FIX DE FRAMES -------------------------------------------------
        # Original: F = (F-1)//4*4 + 1  -> arredonda pra BAIXO e PERDE frames.
        # Bruxos:   sobe pro proximo 4n+1 com espelho, e devolve o tamanho real.
        if frame_fix.startswith("truncar"):
            F = (F_in - 1) // 4 * 4 + 1
            video = input_video.clone()[:F]
            pad_msg = f"truncado {F_in}->{F} (perde {F_in - F} frame(s))"
            if F_in != F:
                logging.info(f"[Bruxos Mocha] ATENCAO: modo truncar descartou {F_in - F} frame(s).")
        else:
            F = _align_up_4n1(F_in)
            video = input_video.clone()
            if F > F_in:
                video = _mirror_pad_frames(video, F)
            video = video[:F]
            pad_msg = (f"padding espelhado {F_in}->{F} (4n+1)" if F > F_in
                       else f"{F} frames (ja era 4n+1)")

        # ---- MASCARA -------------------------------------------------------
        # IMPORTANTE: o MoCha usa UM UNICO plano de mascara pro video inteiro
        # (nao e por frame). Ver o seq_len = (T*2 + 1 + num_refs): esse "+1" e a
        # mascara. Se vier uma mascara por frame (SAM3 etc.), reduzimos a 1 plano.
        m = _normalize_mask(mask)                       # [T,H,W] 0..1 (aceita IMAGE colorida)
        if int(m.shape[0]) > 1:
            if str(mask_frame_mode).startswith("primeiro"):
                m = m[:1]
            else:  # uniao: cobre o sujeito em QUALQUER frame (bom p/ sujeito que se move)
                m = m.amax(dim=0, keepdim=True)
        if int(mask_grow) != 0 or int(mask_blur) != 0:
            m = _grow_blur_mask(m, int(mask_grow), int(mask_blur))
        # casa o tamanho espacial com o video
        if int(m.shape[1]) != H or int(m.shape[2]) != W:
            m = torch.nn.functional.interpolate(
                m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1).clamp(0, 1)
        m = m[:1]                                        # garante 1 plano

        # ---- ENCODE --------------------------------------------------------
        _mem_cleanup(limpar_vram)
        vae.to(device)

        video = video[..., :3].to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)
        r1 = ref1.clone()[..., :3].to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)

        latents = vae.encode(video * 2.0 - 1.0, device, tiled=tiled_vae)

        ref_latents = vae.encode(r1 * 2.0 - 1.0, device, tiled=tiled_vae)
        num_refs = 1
        if ref2 is not None:
            r2 = ref2.clone()[..., :3].to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)
            ref_latents = torch.cat([ref_latents, vae.encode(r2 * 2.0 - 1.0, device, tiled=tiled_vae)], dim=2)
            num_refs = 2

        # mascara -> latente, binarizada em -1/+1 (igual ao MoCha original)
        input_latent_mask = torch.nn.functional.interpolate(
            m.unsqueeze(1).to(vae.dtype), size=(lat_h, lat_w), mode="nearest"
        ).unsqueeze(1)
        input_latent_mask = input_latent_mask.repeat(1, 16, 1, 1, 1).to(device, vae.dtype)
        input_latent_mask[input_latent_mask <= 0.5] = 0
        input_latent_mask[input_latent_mask > 0.5] = 1
        input_latent_mask[input_latent_mask == 0] = -1

        mocha_embeds = torch.cat([latents, input_latent_mask, ref_latents], dim=2)[0]

        target_shape = (16, (F - 1) // 4 + 1, lat_h, lat_w)
        seq_len = (target_shape[1] * 2 + 1 + num_refs) * (target_shape[2] * target_shape[3] // 4)

        if force_offload:
            vae.model.to(offload_device)
        _mem_cleanup(limpar_vram)
        if monitor_memoria:
            _mem_report("mocha: depois do encode")

        dt = time.time() - t0
        info = (f"{W}x{H} | {pad_msg} | latente {lat_h}x{lat_w} T={target_shape[1]} | "
                f"refs={num_refs} | seq_len={seq_len} | encode {_fmt_t(dt)}")
        print(f"[Bruxos Mocha] {info}", flush=True)

        image_embeds = {
            "seq_len": seq_len,
            "mocha_embeds": mocha_embeds,
            "num_frames": F,
            "target_shape": target_shape,
            "mocha_num_refs": num_refs,
        }
        return (image_embeds, int(F_in), info)


# =============================================================================
# Mocha Info (Bruxos) — planeja antes de rodar (sem gastar VRAM)
# =============================================================================
class BruxosMochaInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_video": ("IMAGE", {"tooltip": "O video que voce vai mandar pro MoCha."}),
                "upsampling_factor": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1, "tooltip": "Fator do VAE (Wan = 8). So pra calcular o tamanho do latente."}),
            },
            "optional": {
                "num_refs": ("INT", {"default": 1, "min": 1, "max": 2, "step": 1, "tooltip": "Quantas imagens de referencia voce vai usar (1 ou 2)."}),
                "chunk_size": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 4, "tooltip": "Se > 0, calcula em quantos blocos o video seria dividido (pra pouca VRAM). 0 = uma passada so."}),
                "overlap": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1, "tooltip": "Frames de sobreposicao entre blocos (pro blend costurar)."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("info", "frames_alinhados", "seq_len")
    OUTPUT_TOOLTIPS = (
        "Relatorio: frames, alinhamento 4n+1, tamanho do latente, seq_len e plano de blocos.",
        "Quantos frames o MoCha vai processar de fato (o proximo 4n+1).",
        "seq_len estimado (quanto maior, mais VRAM e mais lento).",
    )
    FUNCTION = "info"
    CATEGORY = CAT
    DESCRIPTION = ("Info do MoCha (Bruxos): calcula ANTES de rodar quantos frames vao entrar (4n+1), o "
                   "tamanho do latente, o seq_len (custo) e, se voce der um chunk_size, em quantos blocos "
                   "o video seria dividido. Nao gasta VRAM -- serve pra planejar.")

    def info(self, input_video, upsampling_factor=8, num_refs=1, chunk_size=0, overlap=8):
        F_in = int(input_video.shape[0])
        H, W = int(input_video.shape[1]), int(input_video.shape[2])
        F = _align_up_4n1(F_in)
        lat_h, lat_w = H // int(upsampling_factor), W // int(upsampling_factor)
        T = (F - 1) // 4 + 1
        seq_len = (T * 2 + 1 + int(num_refs)) * (lat_h * lat_w // 4)

        lines = [f"video: {W}x{H} x {F_in} frames",
                 f"4n+1: {F_in} -> {F}" + (" (espelhado)" if F > F_in else " (ja alinhado)"),
                 f"latente: {lat_h}x{lat_w} x T={T} | refs={int(num_refs)}",
                 f"seq_len: {seq_len}"]

        if int(chunk_size) > 0:
            cs = _align_up_4n1(int(chunk_size))
            step = max(1, cs - int(overlap))
            n = max(1, -(-max(0, F - int(overlap)) // step)) if F > cs else 1
            Tc = (cs - 1) // 4 + 1
            seq_c = (Tc * 2 + 1 + int(num_refs)) * (lat_h * lat_w // 4)
            lines.append(f"blocos: chunk={cs} overlap={int(overlap)} -> ~{n} bloco(s), "
                         f"seq_len por bloco={seq_c} ({100.0 * seq_c / max(1, seq_len):.0f}% do total)")

        out = " | ".join(lines)
        print(f"[Bruxos Mocha Info] {out}", flush=True)
        return (out, int(F), int(seq_len))


NODE_CLASS_MAPPINGS = {
    "BruxosMochaEmbeds": BruxosMochaEmbeds,
    "BruxosMochaInfo": BruxosMochaInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosMochaEmbeds": "Mocha Embeds (Bruxos)",
    "BruxosMochaInfo": "Mocha Info (Bruxos)",
}
