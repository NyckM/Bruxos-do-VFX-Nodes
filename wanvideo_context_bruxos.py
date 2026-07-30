# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — WanVideo Context (janelas temporais anti-OOM)
=============================================================
Gera o objeto WANVIDCONTEXT que o WanVideo Sampler recebe na entrada
`context_options`. Com ele, o sampler processa o video em JANELAS TEMPORAIS em
vez de tudo de uma vez -> corta o pico de VRAM (o remedio de OOM do MoCha e de
qualquer geracao Wan longa/pesada).

100% AUTOSSUFICIENTE: este node nao importa NADA de terceiros. Ele monta o
objeto de contexto em Python puro (um dict com as chaves que o sampler le).
Pode publicar junto do resto do pacote Bruxos sem arrastar dependencia.

  OBS honesta: o objeto so tem efeito se o grafo usar o WanVideo Sampler (que e
  quem consome `context_options`). O MoCha inteiro roda nesse sampler, entao ele
  ja e requisito pra rodar MoCha -- este node NAO adiciona dependencia nova, so
  gera o contexto sem precisar do node de terceiros pra isso.

POR QUE reduz VRAM (e por que e seguro no MoCha):
  O custo do sampler e dominado pela atencao total, que e O(seq_len^2), e o
  seq_len e LINEAR no numero de frames. Processando em janelas de N frames, o
  seq_len cai pro tamanho da janela -> a memoria da atencao cai com o QUADRADO.
  Ex.: 81f -> ~68k tokens; 49f -> ~26k; 33f -> ~18k. No MoCha o sampler reanexa
  a MASCARA unica e as REFS em CADA janela, entao o rastreamento e a identidade
  ficam intactos bloco a bloco (diferente de tile espacial, que quebraria).

Ligue a saida `context_options` na entrada de mesmo nome do WanVideo Sampler.

Categoria: Bruxos do VFX/Wan
"""

import logging

CAT = "Bruxos do VFX/Wan"

# schedules aceitos pelo agendador de janelas do Wan (context_windows/context.py
# do wrapper): estes 3 nomes sao a interface publica estavel.
_SCHEDULES = ["uniform_standard", "uniform_looped", "static_standard"]


class BruxosWanVideoContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_frames": ("INT", {"default": 49, "min": 5, "max": 1000, "step": 4,
                    "tooltip": "Tamanho da JANELA em frames (o botao principal do OOM). Menor = menos VRAM. Se 81 estoura, tente 49; se ainda estoura, 33. O MoCha foi treinado em 21 e 81 frames, entao 33-49 e zona segura. Use 4n+1 (21, 33, 49, 81)."}),
                "context_overlap": ("INT", {"default": 16, "min": 0, "max": 256, "step": 4,
                    "tooltip": "Sobreposicao entre janelas, em frames. O sampler faz crossfade nessa faixa pra nao deixar emenda. 16 e um bom comeco (menor = mais rapido, mais risco de costura visivel)."}),
                "context_stride": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Passo do agendador de janelas (avancado). 4 e o padrao seguro; mexa so se souber o que faz."}),
                "context_schedule": (_SCHEDULES, {"default": "uniform_standard",
                    "tooltip": "Como as janelas deslizam. uniform_standard = recomendado. uniform_looped = so se o video for pra dar LOOP perfeito. static_standard = janelas fixas (mais previsivel, menos suave)."}),
                "freenoise": ("BOOLEAN", {"default": True,
                    "tooltip": "Embaralha o ruido entre janelas pra reduzir repeticao/piscada nas emendas. Deixe LIGADO."}),
            },
            "optional": {
                "verbose": ("BOOLEAN", {"default": False,
                    "tooltip": "Loga no console (do sampler) o plano de janelas. Util pra conferir se ligou certo."}),
                "reference_latent": ("LATENT", {"tooltip": "[avancado/opcional] Latente de referencia por janela (passthrough pro sampler). Deixe vazio se nao souber."}),
            },
        }

    RETURN_TYPES = ("WANVIDCONTEXT", "STRING")
    RETURN_NAMES = ("context_options", "info")
    OUTPUT_TOOLTIPS = (
        "Ligue na entrada 'context_options' do WanVideo Sampler. Faz o sampler processar em janelas temporais (anti-OOM).",
        "Resumo da configuracao das janelas.",
    )
    FUNCTION = "build"
    CATEGORY = CAT
    DESCRIPTION = (
        "WanVideo Context (Bruxos): janelas temporais pro WanVideo Sampler, pra cortar o pico de VRAM "
        "(anti-OOM do MoCha e de geracoes Wan longas). Menor context_frames = menos VRAM (atencao cai com "
        "o quadrado). No MoCha o sampler mantem mascara+refs em cada janela, entao o rastreamento fica "
        "intacto. Node 100% Bruxos (nao importa nada de terceiros). Ligue em context_options do Sampler."
    )

    def build(self, context_frames, context_overlap, context_stride, context_schedule,
              freenoise, verbose=False, reference_latent=None):
        cf = int(context_frames); co = int(context_overlap); cs = int(context_stride)
        sched = str(context_schedule); fn = bool(freenoise); vb = bool(verbose)

        if sched not in _SCHEDULES:
            logging.info(f"[Bruxos WanContext] schedule '{sched}' desconhecido; usando uniform_standard.")
            sched = "uniform_standard"
        if co >= cf:
            co = max(0, cf - 4)
            logging.info(f"[Bruxos WanContext] overlap>=frames; ajustado overlap={co}.")

        # Objeto de contexto = dict com EXATAMENTE as chaves que o WanVideo
        # Sampler le de context_options (verificado no nodes_sampler.py do
        # wrapper: context_frames/schedule/stride/overlap/freenoise/verbose +
        # reference_latent opcional). Python puro, sem import de terceiros.
        ctx = {
            "context_schedule": sched,
            "context_frames": cf,
            "context_stride": cs,
            "context_overlap": co,
            "freenoise": fn,
            "verbose": vb,
            "reference_latent": reference_latent,
        }

        # estimativa didatica da economia de atencao (janela vs 81f de referencia)
        def _seqrel(frames):
            tlat = (frames - 1) // 4 + 1
            return (tlat * 2 + 1)
        mem_ratio = (_seqrel(cf) / _seqrel(81)) ** 2 * 100.0
        info = (f"janela {cf}f (overlap {co}, stride {cs}, {sched}, freenoise={'on' if fn else 'off'}) "
                f"| atencao ~{mem_ratio:.0f}% do custo de 81f")
        print(f"[Bruxos WanContext] {info}", flush=True)
        return (ctx, info)


NODE_CLASS_MAPPINGS = {"BruxosWanVideoContext": BruxosWanVideoContext}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosWanVideoContext": "WanVideo Context (Bruxos)"}
