# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — MiniMax H3: loader tudo-em-um
==============================================
O H3 precisa de QUATRO arquivos, e errar qual vai onde e o erro mais comum:

    diffusion_models/  minimax_h3_ref2va_*.safetensors   -> o transformer
    text_encoders/     qwen3vl_32b_minimax_h3_*.safetensors -> o encoder
    vae/               minimax_h3_video_vae_*.safetensors   -> VAE de VIDEO
    vae/               minimax_h3_audio_vae_*.safetensors   -> VAE de AUDIO

Trocar o VAE de video pelo de audio nao da erro de carregamento -- da ruido, e
voce perde uma hora procurando no lugar errado. Este node junta os quatro,
separa as saidas com nome claro e AVISA quando um arquivo escolhido nao parece
ser o que aquele slot pede.

Substitui: UNETLoader + CLIPLoader + VAELoader + VAELoader.

DUAS VARIANTES DO MODELO (nao sao intercambiaveis):
    ref2va  -> aceita imagens/videos/audios de REFERENCIA (o R2V)
    fl2va   -> texto e primeiro/ultimo frame (o T2V/I2V)
"""

import logging

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/MiniMax H3"


def _lista(pasta):
    try:
        import folder_paths
        v = folder_paths.get_filename_list(pasta)
        return list(v) if v else ["(nenhum encontrado)"]
    except Exception:
        return ["(nenhum encontrado)"]


def _parece(nome, *chaves):
    n = (nome or "").lower().replace("\\", "/")
    return any(c in n for c in chaves)


class BruxosH3Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "modelo": (_lista("diffusion_models"), {"tooltip":
                    "O transformer do H3 (33B), em models/diffusion_models.\n"
                    "ref2va = modo referencia (imagens/videos/audios de referencia).\n"
                    "fl2va  = texto + primeiro/ultimo frame.\n"
                    "Os dois NAO sao intercambiaveis: o node do grafo tem que casar com a variante."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {"default": "default", "tooltip":
                    "Deixe 'default' nos checkpoints ja quantizados (int8/convrot/nvfp4): o arquivo ja "
                    "traz os metadados de quantizacao e forcar fp8 por cima costuma quebrar."}),
                "text_encoder": (_lista("text_encoders"), {"tooltip":
                    "O encoder do H3 -- um Qwen3-VL-32B, em models/text_encoders. "
                    "NAO e um CLIP comum; o H3 usa os hidden states da camada 50 dele."}),
                "vae_video": (_lista("vae"), {"tooltip":
                    "minimax_h3_VIDEO_vae (f16t4d24: 16x espaco, 4x tempo, 24 canais).\n"
                    "Se voce puser o de audio aqui NAO da erro -- da ruido."}),
                "vae_audio": (_lista("vae"), {"tooltip":
                    "minimax_h3_AUDIO_vae (32 kHz, taxa latente 40 Hz, estereo).\n"
                    "Obrigatorio mesmo sem querer audio: o H3 e audio+video e o sampler espera os dois."}),
            },
            "optional": {
                "device_encoder": (["default", "cpu"], {"default": "default", "tooltip":
                    "'cpu' mantem o text encoder (32B!) fora da VRAM. Deixa a codificacao do prompt mais "
                    "lenta, mas libera espaco pro sampler -- util na 4090, onde modelo + encoder + VAEs "
                    "nao cabem juntos."}),
                "avisar": ("BOOLEAN", {"default": True, "tooltip":
                    "Confere se cada arquivo escolhido parece ser o do slot (procura 'video'/'audio' no "
                    "nome do VAE, etc.) e avisa no console. Nao impede nada, so alerta."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae_video", "vae_audio", "info")
    OUTPUT_TOOLTIPS = (
        "-> BasicGuider / BasicScheduler (ou o Patch Sol-Attn antes).",
        "-> 'clip' do MiniMax H3 Reference to Video.",
        "-> 'vae' do node do H3 e do VAE Decode do VIDEO.",
        "-> 'audio_vae' do node do H3 e do VAE Decode Audio.",
        "O que foi carregado e os avisos.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "MiniMax H3 Loader (Bruxos): carrega os quatro arquivos do H3 num node so -- transformer, text encoder "
        "(Qwen3-VL-32B) e os DOIS VAEs (video e audio) -- com as saidas nomeadas pra nao trocar um pelo outro. "
        "Substitui UNETLoader + CLIPLoader + VAELoader + VAELoader. Avisa quando o arquivo escolhido nao parece "
        "ser o daquele slot: por o VAE de audio no de video nao da erro, da ruido."
    )

    def run(self, modelo, weight_dtype, text_encoder, vae_video, vae_audio,
            device_encoder="default", avisar=True):
        import comfy.sd
        import comfy.utils
        import folder_paths

        avisos = []
        if avisar:
            if not _parece(vae_video, "video", "vid"):
                avisos.append(f"'vae_video' = {vae_video!r} nao tem 'video' no nome. Se voce pos o VAE de "
                              f"AUDIO aqui, o resultado sai como RUIDO, sem mensagem de erro.")
            if not _parece(vae_audio, "audio", "aud"):
                avisos.append(f"'vae_audio' = {vae_audio!r} nao tem 'audio' no nome. Confira -- os dois VAEs "
                              f"ficam na mesma pasta e sao faceis de trocar.")
            if vae_video == vae_audio:
                avisos.append("os DOIS VAEs apontam pro MESMO arquivo. Um deles esta errado.")
            if not _parece(text_encoder, "qwen", "minimax", "h3"):
                avisos.append(f"'text_encoder' = {text_encoder!r} nao parece o encoder do H3 "
                              f"(esperado um qwen3vl_*_minimax_h3_*).")
            if not _parece(modelo, "minimax", "h3"):
                avisos.append(f"'modelo' = {modelo!r} nao parece um checkpoint do H3.")

        # ---- transformer ---------------------------------------------------
        opcoes = {}
        if weight_dtype == "fp8_e4m3fn":
            opcoes["dtype"] = __import__("torch").float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            import torch
            opcoes["dtype"] = torch.float8_e4m3fn
            opcoes["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            opcoes["dtype"] = __import__("torch").float8_e5m2

        caminho = folder_paths.get_full_path_or_raise("diffusion_models", modelo)
        model = comfy.sd.load_diffusion_model(caminho, model_options=opcoes)

        # ---- text encoder --------------------------------------------------
        te_opcoes = {}
        if device_encoder == "cpu":
            import comfy.model_management
            te_opcoes["load_device"] = te_opcoes["offload_device"] = \
                comfy.model_management.text_encoder_offload_device()
        te_caminho = folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        # O membro do enum NAO se chama MINIMAX_H3. O CLIPLoader nativo lista os
        # membros em minusculo e o valor que aparece la e 'minimax' -- ou seja,
        # CLIPType.MINIMAX. Eu chutei o nome errado, o getattr falhava, e o
        # fallback carregava o Qwen3-VL como STABLE_DIFFUSION: tokenizer errado,
        # condicionamento errado, prompt quase sem efeito. Ficou assim em toda
        # geracao ate ser pego. Por isso agora tenta uma LISTA de nomes e, se
        # nenhum servir, IMPRIME os que existem em vez de adivinhar de novo.
        tipo = None
        for _nome in ("MINIMAX_H3", "MINIMAX", "MINIMAXH3", "QWEN3VL", "QWEN_VL"):
            tipo = getattr(comfy.sd.CLIPType, _nome, None)
            if tipo is not None:
                break
        if tipo is None:
            try:
                disponiveis = ", ".join(sorted(m.name for m in comfy.sd.CLIPType))
            except Exception:
                disponiveis = "(nao consegui listar)"
            avisos.append(
                "NAO ACHEI o CLIPType do MiniMax H3 neste ComfyUI. Vou carregar o text "
                "encoder como STABLE_DIFFUSION, e isso e tokenizer ERRADO: o prompt perde "
                "quase todo o efeito e o condicionamento das referencias sai torto -- sem "
                "erro nenhum, so resultado pior.\n"
                "     Tipos que existem aqui: %s\n"
                "     Me diga qual e o do MiniMax que eu acrescento na lista." % disponiveis)
            tipo = comfy.sd.CLIPType.STABLE_DIFFUSION
        else:
            print("[Bruxos H3 Loader] CLIPType.%s" % _nome, flush=True)
        clip = comfy.sd.load_clip(ckpt_paths=[te_caminho],
                                  embedding_directory=folder_paths.get_folder_paths("embeddings"),
                                  clip_type=tipo, model_options=te_opcoes)

        # ---- os dois VAEs --------------------------------------------------
        def _vae(nome):
            sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", nome))
            return comfy.sd.VAE(sd=sd)

        vv, va = _vae(vae_video), _vae(vae_audio)

        info = (f"modelo={modelo} ({weight_dtype}) | encoder={text_encoder}"
                f"{' [cpu]' if device_encoder == 'cpu' else ''} | "
                f"vae_video={vae_video} | vae_audio={vae_audio}")
        print(f"[Bruxos H3 Loader] {info}", flush=True)
        for a in avisos:
            print(f"[Bruxos H3 Loader]   AVISO: {a}", flush=True)
        if avisos:
            info += "\n" + "\n".join(f"- {a}" for a in avisos)
        return (model, clip, vv, va, info)


NODE_CLASS_MAPPINGS = {"BruxosH3Loader": BruxosH3Loader}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosH3Loader": "MiniMax H3 · Loader tudo-em-um (Bruxos)"}
