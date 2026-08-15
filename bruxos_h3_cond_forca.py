# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — MiniMax H3: Forca do Condicionamento
=====================================================
O QUE ISTO CONTROLA
    Quanto o resultado GRUDA nas referencias (imagens, video, audio) em vez de
    inventar. E o unico controle direto disso no H3, e nenhum node de fabrica
    escreve nele -- voce sempre rodou no padrao sem saber que existia.

DE ONDE VEM (lido do core, nao chutado)
    comfy/model_base.py:2100
        if kwargs.get("minimax_visual_cond_noise_aug", None) is not None:
            payload["visual_cond_noise_aug"] = kwargs["minimax_visual_cond_noise_aug"]

    comfy/ldm/minimax/model.py:473
        aug = payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP)  # 0.999
        if aug < 1.0:
            noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
            r = aug * r + (1.0 - aug) * noise

    Ou seja: 'aug' e o PESO DO LATENTE REAL, nao a quantidade de ruido.
        aug = 1.000  -> referencia pura, nenhum ruido        (gruda mais)
        aug = 0.999  -> o padrao do core, 0.1% de ruido
        aug = 0.950  -> 5% de ruido                          (mais liberdade)

    E em model.py:539 o mesmo valor vira o TIMESTEP das linhas de condicao:
        seg_t["cond"] = max(t_v, vis_aug)
    Com aug=1.0 as linhas de referencia ficam cravadas em t=1.0, que e o
    extremo "isto aqui e dado, nao e ruido a ser resolvido".

POR QUE O PADRAO NAO E 1.0
    Os 0.1% de ruido existem para o modelo nao tratar a referencia como
    intocavel -- ajuda quando voce QUER que ele reinterprete. Para style
    transfer e relight, onde a fidelidade de movimento e o objetivo, subir
    para 1.0 costuma ser o que voce quer.

O AUDIO ja vem em 1.0 por padrao (AUDIO_COND_TIMESTEP = 1.0), entao mexer nele
so faz sentido para AFROUXAR.
"""

import logging

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/MiniMax H3"

try:
    import node_helpers
    _OK = True
except Exception:  # pragma: no cover
    node_helpers = None
    _OK = False

_PADRAO_VIS = 0.999   # VISUAL_COND_TIMESTEP no core
_PADRAO_AUD = 1.0     # AUDIO_COND_TIMESTEP no core


class BruxosH3CondForca:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip":
                    "A saida 'positive' do MiniMaxH3ReferenceToVideo (ou do ImageToVideo). "
                    "Ligue este node DEPOIS dele e antes do guider/sampler."}),
                "visual": ("FLOAT", {"default": 0.999, "min": 0.80, "max": 1.0, "step": 0.001,
                    "tooltip":
                    "PESO DO LATENTE REAL das referencias visuais -- nao e a quantidade de ruido.\n\n"
                    "1.000 = referencia pura, zero ruido. GRUDA MAIS: use quando o objetivo e "
                    "seguir o video (style transfer, relight, previz).\n"
                    "0.999 = o padrao do ComfyUI. Os 0.1% de ruido existem para o modelo se "
                    "sentir livre para reinterpretar um pouco.\n"
                    "0.98-0.95 = solta a mao. Use quando a referencia esta mandando demais e "
                    "voce quer mais invencao.\n\n"
                    "A conta no core e literalmente:  r = aug * referencia + (1 - aug) * ruido\n"
                    "Abaixo de 0.90 a referencia vira quase so ruido -- por isso o minimo aqui e 0.80."}),
                "audio": ("FLOAT", {"default": 1.0, "min": 0.80, "max": 1.0, "step": 0.001,
                    "tooltip":
                    "O mesmo, para as referencias de AUDIO.\n\n"
                    "O padrao do core ja e 1.0 (referencia pura), entao mexer aqui so serve para "
                    "AFROUXAR -- por exemplo quando a voz de referencia esta sendo copiada "
                    "literalmente demais e voce quer so o timbre."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_TOOLTIPS = ("O mesmo condicionamento, agora com a forca escrita nele.",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "MiniMax H3 · Forca do Condicionamento (Bruxos): controla o quanto o resultado gruda nas "
        "referencias. O core do ComfyUI le 'minimax_visual_cond_noise_aug' e 'minimax_audio_cond_noise_aug', "
        "mas nenhum node de fabrica escreve neles -- sem este node voce roda sempre no padrao. "
        "MAIOR = mais fiel a referencia. 1.0 = referencia pura, sem ruido."
    )

    def run(self, conditioning, visual=0.999, audio=1.0):
        if not _OK:
            raise RuntimeError(
                "[Bruxos H3 Cond] nao consegui importar 'node_helpers' do ComfyUI. "
                "Este node depende dele para escrever no conditioning.")
        v, a = float(visual), float(audio)
        # So escreve o que foi MEXIDO. Escrever o proprio padrao muda o caminho
        # no core (a chave passa a existir), e eu prefiro que 'nao mexi' seja
        # indistinguivel de 'node ausente'.
        valores = {}
        if abs(v - _PADRAO_VIS) > 1e-9:
            valores["minimax_visual_cond_noise_aug"] = v
        if abs(a - _PADRAO_AUD) > 1e-9:
            valores["minimax_audio_cond_noise_aug"] = a

        if not valores:
            print("[Bruxos H3 Cond] visual=%.3f audio=%.3f -- ambos no padrao, "
                  "condicionamento passou intacto." % (v, a), flush=True)
            return (conditioning,)

        out = node_helpers.conditioning_set_values(conditioning, valores)
        print("[Bruxos H3 Cond] " + " | ".join(
            "%s=%.3f (padrao %.3f)" % (k.replace("minimax_", "").replace("_cond_noise_aug", ""),
                                       val, _PADRAO_VIS if "visual" in k else _PADRAO_AUD)
            for k, val in valores.items()), flush=True)
        if v >= 1.0:
            print("[Bruxos H3 Cond]   visual em 1.0: referencia pura, zero ruido. "
                  "Se o resultado ficar ENGESSADO ou com aparencia de congelado, "
                  "volte para 0.999.", flush=True)
        return (out,)


NODE_CLASS_MAPPINGS = {"BruxosH3CondForca": BruxosH3CondForca}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3CondForca": "MiniMax H3 · Força do Condicionamento (Bruxos)"
}
