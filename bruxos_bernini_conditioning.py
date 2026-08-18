# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Bernini Conditioning (standalone)
====================================================
PRA QUE SERVE
    O 'BerniniInfinity' monta a referencia/identidade (context_latents) por
    DENTRO dele mesmo, a cada chamada -- e essa versao "enriquecida" do
    conditioning nunca sai pra fora (ele nao tem uma saida CONDITIONING).
    Isso significa que, se voce ligar o positive/negative do seu Prompt Guide
    direto num KSampler comum (ex.: pra um 2o passe leve, so de refino de
    resolucao), esse KSampler NAO VE a referencia -- so o texto. Resultado
    classico: a identidade deriva, e baixar o denoise so desacelera a deriva,
    nao resolve.

    Este node faz SO essa parte (monta o context_latents a partir do
    source_video/reference_video/reference_images e devolve positive/negative
    prontos), SEM rodar nenhum sampler -- pra voce plugar num KSampler comum,
    SamplerCustomAdvanced, ou o que quiser. E o mesmo codigo que o
    BerniniInfinity usa por dentro (_collect_reference_latents,
    _clone_conditioning_set_values), so exposto como um passo separado.

LIGACAO TIPICA (2o passe leve, sem o BerniniInfinity inteiro)
    BerniniInfinity #1 (baixa res, denoise=1.0)
        latent -> Bernini Latent Upscale (Bruxos)
                     latent -> KSampler.latent_image (denoise 0.4-0.6)
    [este node] (MESMO source_video/reference_images do passe 1, width/height
                 do passe 2) -> positive/negative -> KSampler

CUIDADOS
  * Use o MESMO source_video (e reference_video/reference_images, se usar)
    do passe 1 -- e o que mantem a identidade entre os passes.
  * width/height aqui devem bater com a resolucao do passe 2 (a mesma que
    voce vai usar no KSampler) -- e o que alinha o contexto 0 ao latente
    que vai ser denoisado.
  * Isto NAO faz o split high/low model do Bernini-R -- e so o
    conditioning. Escolha 1 modelo (normalmente o low_model, que faz a
    maior parte do refino de detalhe) pro KSampler do passe 2.
"""

import logging

try:
    import torch
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

try:
    from .nodes import (
        _align_up_4n1,
        _mirror_pad_frames,
        _resize_source_video,
        _encode_video,
        _collect_reference_latents,
        _clone_conditioning_set_values,
    )
except Exception:  # pragma: no cover
    from nodes import (
        _align_up_4n1,
        _mirror_pad_frames,
        _resize_source_video,
        _encode_video,
        _collect_reference_latents,
        _clone_conditioning_set_values,
    )

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Bernini"


class BruxosBerniniConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip":
                    "Texto ja codificado (ex.: saida do Prompt Guide/CLIP Text Encode), SEM a referencia ainda."}),
                "negative": ("CONDITIONING", {"tooltip": "Negativo, mesma logica do positive."}),
                "vae": ("VAE", {"tooltip": "O MESMO VAE (wan_2.1_vae) usado no passe 1."}),
                "source_video": ("IMAGE", {"tooltip":
                    "O MESMO video-fonte do passe 1. E ele que vira o context_latents[0] "
                    "(o contexto alinhado ao canvas -- essencial pra identidade/edicao)."}),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16, "tooltip":
                    "Resolucao do PASSE 2 (a mesma que voce vai usar no KSampler/latent upscalado). "
                    "Tem que bater -- e o que alinha o contexto 0 ao latente que sera denoisado."}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16, "tooltip":
                    "Ver width. Multiplo de 16, igual ao BerniniInfinity."}),
            },
            "optional": {
                "resize_mode": (["stretch", "crop"], {"default": "stretch", "tooltip":
                    "Mesmo parametro do BerniniInfinity -- use o MESMO valor do passe 1 pra nao desalinhar."}),
                "ref_max_size": ("INT", {"default": 848, "min": 16, "max": 8192, "step": 16, "tooltip":
                    "Lado maior das referencias antes de virarem latente de contexto. Mesma logica do BerniniInfinity."}),
                "reference_video": ("IMAGE", {"tooltip": "Video de referencia opcional (identico ao do passe 1)."}),
                "reference_images.reference_image_0": ("IMAGE", {"tooltip":
                    "Imagem de referencia 0 (mesma do passe 1). No prompt: 'from image0'."}),
                "reference_images.reference_image_1": ("IMAGE", {"tooltip": "Imagem de referencia 1."}),
                "reference_images.reference_image_2": ("IMAGE", {"tooltip": "Imagem de referencia 2."}),
                "reference_images.reference_image_3": ("IMAGE", {"tooltip": "Imagem de referencia 3."}),
                "reference_images.reference_image_4": ("IMAGE", {"tooltip": "Imagem de referencia 4."}),
                "reference_images.reference_image_5": ("IMAGE", {"tooltip": "Imagem de referencia 5."}),
                "reference_images.reference_image_6": ("IMAGE", {"tooltip": "Imagem de referencia 6."}),
                "reference_images.reference_image_7": ("IMAGE", {"tooltip": "Imagem de referencia 7."}),
                "ref_influence_vid": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05, "tooltip":
                    "Escala a magnitude do latente do reference_video. 1.0 = neutro (igual ref_influence_vid_off)."}),
                "ref_influence_img": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05, "tooltip":
                    "Escala a magnitude dos latentes das reference_images. 1.0 = neutro (igual ref_influence_img_off)."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "info")
    OUTPUT_TOOLTIPS = (
        "positive com context_latents embutido -- ligue no seu sampler (KSampler, SamplerCustomAdvanced etc.).",
        "negative com context_latents embutido.",
        "Quantas referencias entraram e o tamanho usado.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Bernini Conditioning (Bruxos): monta o context_latents (source_video + referencias) e devolve "
        "positive/negative prontos pra QUALQUER sampler -- sem rodar geracao nenhuma. Existe pra manter a "
        "identidade/edicao guiada por referencia num 2o passe leve (KSampler comum) sem precisar do "
        "BerniniInfinity inteiro (que carrega tiling/janela/mascara/limpeza de VRAM, mais lento pra um refino simples)."
    )

    def run(self, positive, negative, vae, source_video, width, height,
             resize_mode="stretch", ref_max_size=848,
             reference_video=None, ref_influence_vid=1.0, ref_influence_img=1.0,
             **kwargs):
        if not _OK:
            raise RuntimeError("[Bernini Conditioning] torch indisponivel.")
        if positive is None or negative is None:
            raise ValueError(
                "[Bernini Conditioning] positive/negative chegaram vazios (None). Causa comum: o Prompt "
                "Guide (Bruxos) sem CLIP ligado devolve None -- confira o console."
            )

        reference_images = {
            key: value
            for key, value in kwargs.items()
            if key.startswith("reference_images.reference_image_") and value is not None
        }

        target = int(source_video.shape[0])
        aligned = _align_up_4n1(target)
        raw = source_video[:target]
        if aligned != target:
            raw = _mirror_pad_frames(raw, aligned)

        full_source = _resize_source_video(raw, int(width), int(height), resize_mode)
        encoded_source = _encode_video(vae, full_source)
        context_latents = [encoded_source]
        context_latents.extend(
            _collect_reference_latents(
                vae, aligned, int(ref_max_size),
                reference_video=reference_video, reference_images=reference_images,
                scale_vid=float(ref_influence_vid), scale_img=float(ref_influence_img),
            )
        )

        values = {"context_latents": context_latents}
        pos = _clone_conditioning_set_values(positive, values)
        neg = _clone_conditioning_set_values(negative, values)

        n_refs = len(context_latents) - 1
        info = (f"source {int(width)}x{int(height)} ({aligned} frames latentes) | "
                f"{n_refs} referencia(s) extra | ref_max_size={int(ref_max_size)}")
        print(f"[Bernini Conditioning] {info}", flush=True)
        return (pos, neg, info)


NODE_CLASS_MAPPINGS = {"BruxosBerniniConditioning": BruxosBerniniConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosBerniniConditioning": "Bernini Conditioning (Bruxos)"
}
