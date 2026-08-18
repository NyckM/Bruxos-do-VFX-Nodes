# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Bernini: upscale de LATENTE entre dois passes (2-pass hires)
=============================================================================
PRA QUE SERVE
    Mesma ideia do "MiniMax H3 - Latent Upscale 2-pass", adaptada pro Bernini
    (Wan 2.2): gera a maior parte barato numa resolucao BAIXA, upscala o
    latente, e TERMINA numa resolucao maior com poucos passos de refino.

        passe 1 (BerniniInfinity, res baixa, denoise=1.0)
            -> saida 'latent'
            -> [este node] escala 1.5, metodo bicubic
            -> passe 2 (BerniniInfinity, res ALVO, denoise 0.4-0.6,
                        entrada opcional 'init_latent' = saida daqui)

POR QUE E MAIS SIMPLES QUE O DO H3
    O latente do H3 e um container com dois componentes (video 24ch + audio
    32ch) e o node dele precisa recriar manualmente o ruido do passe 2 via
    model_sampling (porque usa SamplerCustomAdvanced cru, que espera um
    latente JA ruidado).

    O Bernini usa o latente puro do Wan ([B,16,T,H,W], sem audio) e o passe 2
    roda dentro do proprio node 'BerniniInfinity' (BruxosBerniniInfinity),
    que ja tem uma entrada 'init_latent': quando ligada, ele mesmo adiciona o
    ruido certo (via SamplerCustom + o schedule de sigmas do 'denoise' do
    passe 2). Este node, portanto, so precisa fazer UMA coisa: redimensionar
    o latente espacialmente, mantendo o grid par que o VAE do Wan exige.

LIGACAO
    BerniniInfinity #1 (width/height BAIXOS, denoise=1.0)
        latent -> [este node] -> BerniniInfinity #2.init_latent
                                  (width/height ALVO, denoise 0.4-0.6,
                                   MESMO source_video/positive/negative)

CUIDADOS
  * So funciona com o BerniniInfinity em mode=context_window (o init_latent
    do node principal ainda nao suporta mode=sequential).
  * Use a saida 'latent' do passe 1 (nao a 'images').
  * denoise do passe 2 precisa ser < 1.0 -- senao ele regenera do zero e o
    latente que chega aqui e descartado.
"""

import logging

try:
    import torch
    import torch.nn.functional as F
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

try:
    from . import bruxos_bernini_neural_resizer as _neural
except Exception:  # pragma: no cover
    try:
        import bruxos_bernini_neural_resizer as _neural
    except Exception:
        _neural = None

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Bernini"
METODO_NEURAL = "neural (wan2.2, experimental)"

# VAE do Wan comprime 8x no espaco -> a resolucao final (multiplo de 16 em
# pixels, ver width/height do BerniniInfinity) exige LATENTE multiplo de 2.
MULT_LATENTE = 2


def _par(n):
    n = int(round(n))
    return max(MULT_LATENTE, n - (n % MULT_LATENTE))


def _escalar_espacial(t, nh, nw, metodo):
    """[B,C,T,H,W] -> mesma coisa com H,W novos (so espacial; T intocado)."""
    if t.ndim != 5:
        raise ValueError(f"[Bruxos Bernini Upscale] esperava latente 5D [B,C,T,H,W]; veio {tuple(t.shape)}.")
    B, C, T, H, W = t.shape
    x = t.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    kwargs = {"align_corners": False} if metodo in ("bilinear", "bicubic") else {}
    x = F.interpolate(x.float(), size=(nh, nw), mode=metodo, **kwargs)
    return x.reshape(B, T, C, nh, nw).permute(0, 2, 1, 3, 4).to(t.dtype).contiguous()


class BruxosBerniniLatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip":
                    "A saida 'latent' de um BerniniInfinity (passe 1, resolucao baixa, denoise=1.0). "
                    "Latente do Wan: [B,16,T,H,W] (16 canais -- diferente do MiniMax H3, que tem 24)."}),
                "escala": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip":
                    "Quanto crescer no espaco. 1.5 costuma ser um bom negocio (custo bem menor que "
                    "gerar direto no alvo). O resultado e arredondado pra manter o latente PAR.\n"
                    "Combine com o width/height do PASSE 2 (BerniniInfinity #2): eles devem bater com "
                    "(width_passe1 * escala, height_passe1 * escala), arredondados a multiplo de 16."}),
                "metodo": (["bicubic", "bilinear", "nearest-exact", "area", METODO_NEURAL],
                    {"default": "bicubic", "tooltip":
                    "Interpolacao do latente. bicubic segura melhor o detalhe; bilinear e mais macio; "
                    "nearest-exact preserva blocos duros; area e boa pra reduzir (raramente o caso aqui).\n"
                    f"'{METODO_NEURAL}': troca a interpolacao por uma rede treinada (baixa um checkpoint "
                    "~3.9MB sozinha na 1a vez). Ajuda com o fantasma/imagem-duplicada que a interpolacao "
                    "pura causa quando o denoise do passe 2 e baixo -- mas o autor do modelo marca ele "
                    "como WIP/experimental (ganho modesto sobre bilinear). Se falhar (sem internet, etc.), "
                    "cai pro bicubic sozinho e avisa no console."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    OUTPUT_TOOLTIPS = (
        "Latente maior (sem ruido extra) -> ligue em BerniniInfinity #2.init_latent, com denoise < 1.0.",
        "Resolucao antes/depois do latente e em pixels.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Bernini Latent Upscale 2-pass (Bruxos): redimensiona o latente do Wan/Bernini (16ch, sem audio) "
        "entre dois passes do BerniniInfinity, pra gerar a maior parte numa resolucao baixa e SO refinar "
        "na resolucao alvo. Nao precisa mexer com ruido/model_sampling -- quem faz isso e o proprio "
        "BerniniInfinity #2 (via a entrada 'init_latent' + 'denoise' < 1.0)."
    )

    def run(self, latent, escala, metodo):
        if not _OK:
            raise RuntimeError("[Bruxos Bernini Upscale] torch indisponivel.")

        s = latent.get("samples") if isinstance(latent, dict) else latent
        if not torch.is_tensor(s):
            raise ValueError(
                "[Bruxos Bernini Upscale] nao reconheci o latente (esperava um dict com 'samples'). "
                "Ligue a saida 'latent' de um BerniniInfinity."
            )
        if s.ndim != 5:
            raise ValueError(f"[Bruxos Bernini Upscale] esperava latente 5D [B,C,T,H,W]; veio {tuple(s.shape)}.")

        B, C, T, H, W = (int(v) for v in s.shape)
        if C != 16:
            print(f"[Bruxos Bernini Upscale] AVISO: latente com {C} canais (esperava 16, formato Wan/Bernini). "
                  f"Seguindo mesmo assim, mas confira se este latente e mesmo do Bernini "
                  f"(o do MiniMax H3, por exemplo, tem 24 canais e nao e compativel).", flush=True)

        nh, nw = _par(H * float(escala)), _par(W * float(escala))
        metodo_usado = metodo
        if (nh, nw) == (H, W):
            print("[Bruxos Bernini Upscale] AVISO: a escala nao mudou o tamanho do latente "
                  "(escala perto de 1.0 ou latente ja pequeno demais pra arredondar). Repassando sem mudanca.",
                  flush=True)
            out_samples = s
        elif metodo == METODO_NEURAL:
            out_samples = _neural.neural_upscale(s, nh, nw) if _neural is not None else None
            if out_samples is None:
                print("[Bruxos Bernini Upscale] neural indisponivel -- caindo pro metodo bicubic.", flush=True)
                metodo_usado = "bicubic (fallback do neural)"
                out_samples = _escalar_espacial(s, nh, nw, "bicubic")
        else:
            out_samples = _escalar_espacial(s, nh, nw, metodo)

        out = dict(latent) if isinstance(latent, dict) else {}
        out["samples"] = out_samples
        out.pop("noise_mask", None)  # mascara antiga esta no tamanho velho

        info = (f"latente {H}x{W} -> {nh}x{nw} | pixels {W*8}x{H*8} -> {nw*8}x{nh*8} | "
                f"escala pedida {escala} | {metodo_usado} | {T} frame(s) latente(s) intocados")
        print(f"[Bruxos Bernini Upscale] {info}", flush=True)
        return (out, info)


NODE_CLASS_MAPPINGS = {"BruxosBerniniLatentUpscale": BruxosBerniniLatentUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosBerniniLatentUpscale": "Bernini · Latent Upscale 2-pass (Bruxos)"
}
