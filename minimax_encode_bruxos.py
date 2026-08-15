# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — MinimaxDaBruxos Encode/Decode (COMPATIBILIDADE)
===============================================================
Este arquivo tinha uma implementacao ERRADA que gerava ruido. Ele foi mantido
APENAS pra nao quebrar os grafos que ja usam os IDs `MinimaxDaBruxosEncode` /
`MinimaxDaBruxosDecode` -- agora ele DELEGA pra implementacao correta em
`minimax_h3_bruxos.py`.

O QUE ESTAVA ERRADO (e por que dava aquele ruido tipo camuflagem):

  1. Permutava os eixos na mao:
         frames.permute(3,0,1,2).unsqueeze(0)   ->  [B,C,T,H,W]
     Mas o `vae.encode()` do ComfyUI espera [T,H,W,C] e faz o movedim SOZINHO.
     Com a entrada ja permutada, o movedim interno embaralhava de novo e o
     tensor virava 6-D. Por isso o encode temporal do H3 devolvia lista vazia:
         comfy/ldm/minimax/vae.py -> torch.cat(z_list, dim=2)
         ValueError: expected a non-empty list of Tensors
     Era exatamente o erro do node #37.

  2. Normalizava 2x:
         video_input = frames * 2.0 - 1.0
     O `vae.encode()` ja faz o (x*2-1) internamente -> range errado.

  3. Usava `encode_tiled` sempre. No VAE do H3 o encode_tiled so chama o encode
     normal, e o caminho tiled do ComfyUI ainda mexe nos eixos.

  4. Inventava o latente de audio:
         torch.zeros((B, 1, T))
     Shape chutado. O H3 e AUDIO+VIDEO e tem VAE de audio proprio -- o certo e
     codificar SILENCIO nesse VAE, ai o shape e a distribuicao saem corretos.

  5. No decode, des-normalizava 2x ((x+1)/2) e permutava de novo -- o
     `vae.decode()` ja devolve [T,H,W,C] em 0..1.

RECOMENDADO: troque pelos nodes novos
    'MiniMax H3 · Encode Video -> Latente (Bruxos)'
    'Inspetor de LATENT (Bruxos)'
Estes aqui continuam funcionando, so que agora pelo caminho certo.
"""

import logging

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/MiniMax H3"

try:
    from .minimax_h3_bruxos import BruxosMinimaxH3EncodeVideo as _Impl
except Exception:  # pragma: no cover
    try:
        from minimax_h3_bruxos import BruxosMinimaxH3EncodeVideo as _Impl
    except Exception as e:
        _Impl = None
        log.warning("[MinimaxDaBruxos] implementacao nova indisponivel: %s", e)


class MinimaxDaBruxosEncode:
    """[LEGADO] Mesmo ID de node, implementacao corrigida por dentro."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Frames do video. Passe CRU -- nao normalize nem permute antes."}),
                "vae": ("VAE", {"tooltip": "O VAE DE VIDEO do H3 (minimax_h3_video_vae_*)."}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip":
                    "O VAE DE AUDIO do H3 (minimax_h3_audio_vae_*). LIGUE MESMO SEM QUERER AUDIO: o H3 e audio+video "
                    "e o sampler espera os dois latentes. Sem audio, codificamos SILENCIO aqui (shape correto, em vez "
                    "do tensor de zeros chutado que a versao antiga criava)."}),
                "audio": ("AUDIO", {"tooltip": "[opcional] Audio real do video. Se vazio, usamos silencio."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "FPS do video -- define a duracao do silencio (n_frames/fps)."}),
                "audio_sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 100,
                    "tooltip": "Taxa esperada pelo VAE de audio."}),
                "reference_latent": ("LATENT", {"tooltip":
                    "[opcional, caminho mais seguro] Saida LATENT do node nativo 'MiniMax H3 Image to Video'. "
                    "Se ligado, herdamos o container exato dele."}),
                "strict": ("BOOLEAN", {"default": True,
                    "tooltip": "[com reference_latent] Para com erro claro se os shapes nao baterem."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode_h3_av"
    CATEGORY = CAT
    DESCRIPTION = (
        "[LEGADO - prefira 'MiniMax H3 · Encode Video -> Latente (Bruxos)'] Mesmo ID de node de antes, mas a "
        "implementacao interna foi corrigida: sem normalizar 2x, sem permutar eixos e sem inventar o latente "
        "de audio. Ligue o 'audio_vae' pra funcionar."
    )

    def encode_h3_av(self, frames, vae, audio_vae=None, audio=None, fps=24.0,
                     audio_sample_rate=44100, reference_latent=None, strict=True,
                     force_offload=True, monitor_memoria=False, **_ignorados):
        # force_offload/monitor_memoria continuam aceitos so pra nao quebrar
        # grafos antigos que tenham esses widgets salvos.
        if _Impl is None:
            raise RuntimeError(
                "[MinimaxDaBruxos] nao consegui carregar a implementacao nova (minimax_h3_bruxos.py). "
                "Confira se o arquivo esta na pasta ComfyUI-Bruxos-do-VFX."
            )
        print("[MinimaxDaBruxos] node LEGADO -> rodando pela implementacao CORRIGIDA "
              "(recomendado trocar pelo 'MiniMax H3 · Encode Video -> Latente (Bruxos)').", flush=True)
        latent, _info = _Impl().encode(
            frames=frames, video_vae=vae, audio_vae=audio_vae, audio=audio,
            fps=fps, audio_sample_rate=audio_sample_rate,
            reference_latent=reference_latent, strict=strict,
        )
        return (latent,)


class MinimaxDaBruxosDecode:
    """[LEGADO] Decode do latente do H3 -> frames. Sem des-normalizar 2x."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latente vindo do sampler."}),
                "vae": ("VAE", {"tooltip": "O VAE DE VIDEO do H3."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode_h3_av"
    CATEGORY = CAT
    DESCRIPTION = (
        "[LEGADO - prefira o 'VAE Decode' normal do ComfyUI] Extrai o componente de VIDEO do container do H3 e "
        "decodifica. Corrigido: NAO faz (x+1)/2 nem permuta (o vae.decode() ja devolve [T,H,W,C] em 0..1)."
    )

    def decode_h3_av(self, latent, vae, force_offload=True, **_ignorados):
        import torch
        s = latent.get("samples", latent) if isinstance(latent, dict) else latent
        video = None
        if torch.is_tensor(s):
            video = s
        else:
            t = getattr(s, "tensors", None)
            if isinstance(t, (list, tuple)) and t:
                video = t[0]
            elif isinstance(s, (list, tuple)) and s:
                video = s[0]
        if video is None:
            raise ValueError(
                "[MinimaxDaBruxosDecode] nao achei o componente de video no latente. "
                "Ponha o 'Inspetor de LATENT (Bruxos)' no fio pra ver a estrutura."
            )
        print(f"[MinimaxDaBruxosDecode] decodificando video {tuple(video.shape)}...", flush=True)
        out = vae.decode(video)   # ja volta [T,H,W,C] em 0..1 -- NAO mexer
        if isinstance(out, dict) and "samples" in out:
            out = out["samples"]
        if out.ndim == 5 and out.shape[0] == 1:
            out = out.squeeze(0)
        print(f"[MinimaxDaBruxosDecode] saida {tuple(out.shape)}", flush=True)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MinimaxDaBruxosEncode": MinimaxDaBruxosEncode,
    "MinimaxDaBruxosDecode": MinimaxDaBruxosDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxDaBruxosEncode": "MinimaxDaBruxos Encode H3 (AV) [legado]",
    "MinimaxDaBruxosDecode": "MinimaxDaBruxos Decode H3 (AV) [legado]",
}
