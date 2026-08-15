# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — MiniMax H3: grade de frames (o "4n+1" do H3)
============================================================
O Wan exige 4n+1 frames. O H3 tem a SUA propria grade, e ela e bem menos
obvia:

    length % 17 == 5      ->  5, 22, 39, 56, 73, 90, 107, 124, 141, 158,
                              175, 192, 209, 226, 243, 260, 277, ...

De onde vem: e a conta que o node `Math Expression` da template OFICIAL do
ComfyUI aplica antes de mandar o `length` pro MiniMaxH3ReferenceToVideo:

    base = max(5, round(duracao * 24))
    length = base + (5 - base % 17) % 17

IMPORTANTE, pra nao virar folclore: esse grid vem da TEMPLATE oficial, nao de
uma linha do model card. O model card documenta 4-15 s e 24 fps. A grade e o
que a implementacao de referencia impoe -- e por isso vale respeitar.

DUAS COISAS QUE ESTE NODE RESOLVE
---------------------------------
1. Voce pede 100 frames e recebe 107 sem entender. Aqui voce VE a grade, os
   vizinhos, e escolhe pra cima ou pra baixo.

2. O SEU VIDEO DE REFERENCIA tem outro numero de frames que o alvo. Isso e o
   que faz a camera "seguir mas ficar levemente diferente", e o que faz o
   final do video inventar conteudo: quando a referencia acaba antes, o
   modelo fica sem material e preenche sozinho. Ligue os frames aqui e ele
   devolve o clipe JA no tamanho exato.

O SEGUNDO 'EXATO'
-----------------
Resolvendo 24*T ≡ 5 (mod 17) na faixa de 4 a 15 s, existe UM unico valor em
que a contagem cai num segundo redondo:

    192 frames = 8.0000 s

Qualquer outra duracao cai quebrada (5 s -> 124 = 5.17 s). Quando o alvo e
1:1 com um render de Blender, 8 s e a escolha que dispensa reamostragem.
"""

import logging

try:
    import torch
    import torch.nn.functional as F
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/MiniMax H3"

PASSO = 17          # length % PASSO == RESTO
RESTO = 5
FPS = 24.0
DUR_MIN, DUR_MAX = 4.0, 15.0


def grade(minimo=5, maximo=400):
    """Todos os comprimentos validos no intervalo."""
    n = minimo + (RESTO - minimo % PASSO) % PASSO
    return list(range(n, maximo + 1, PASSO))


def valido(n):
    return int(n) % PASSO == RESTO


def indices_pingpong(T, L):
    """Indices 0,1,...,T-1,T-2,...,1,0,1,... ate ter L itens.

    O "reverse pra preencher o fim": em vez de congelar o ultimo frame, o
    clipe volta por onde veio. Nao ha corte duro porque a volta comeca do
    proprio ultimo frame, e nao ha frame duplicado nos pontos de virada --
    e por isso que o periodo e 2T-2 e nao 2T.
    """
    T = int(T)
    if T <= 1:
        return [0] * int(L)
    periodo = 2 * T - 2
    saida = []
    for i in range(int(L)):
        j = i % periodo
        saida.append(j if j < T else periodo - j)
    return saida


def encaixar(n_frames, modo="mais_proximo"):
    """Leva um numero qualquer de frames pro valor valido mais adequado."""
    n = max(RESTO, int(round(float(n_frames))))
    if valido(n):
        return n
    baixo = n - ((n - RESTO) % PASSO)
    alto = baixo + PASSO
    if baixo < RESTO:
        baixo = RESTO
    if modo == "cima":
        return alto
    if modo == "baixo":
        return baixo
    return baixo if (n - baixo) <= (alto - n) else alto


class BruxosH3Frames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "entrada": (["duracao_s", "n_frames"], {"default": "duracao_s", "tooltip":
                    "Como voce quer especificar o tamanho: por SEGUNDOS (o normal) ou direto por "
                    "NUMERO DE FRAMES (util quando voce ja sabe quantos frames o render do Blender tem)."}),
                "duracao_s": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 30.0, "step": 0.1, "tooltip":
                    "[modo duracao_s] Duracao alvo. O H3 foi treinado pra 4-15s.\n"
                    "DICA: 8.0s = 192 frames e a UNICA duracao da faixa que cai num segundo exato. "
                    "Se voce esta casando com um render do Blender, use ela e exporte 192 frames."}),
                "n_frames": ("INT", {"default": 192, "min": 1, "max": 2000, "step": 1, "tooltip":
                    "[modo n_frames] Quantos frames voce tem/quer. Vai ser encaixado na grade do H3."}),
                "arredondar": (["mais_proximo", "cima", "baixo"], {"default": "mais_proximo", "tooltip":
                    "Pra onde ir quando o numero cai entre dois valores validos.\n"
                    "cima  = video um tico mais longo (nunca perde conteudo da referencia)\n"
                    "baixo = mais rapido de gerar"}),
            },
            "optional": {
                "frames": ("IMAGE", {"tooltip":
                    "[opcional] Seu video de REFERENCIA (o blocking, por exemplo). Se ligado, ele sai daqui "
                    "com EXATAMENTE 'length' frames.\n"
                    "E isto que evita o final inventado: quando a referencia acaba antes do alvo, o modelo "
                    "fica sem material e preenche por conta propria."}),
                "ajuste": (["pingpong", "cortar_ou_congelar", "reamostrar", "so_avisar"],
                    {"default": "pingpong", "tooltip":
                    "Como levar seus frames ao tamanho alvo:\n\n"
                    "pingpong (RECOMENDADO) = quando falta, o clipe VOLTA por onde veio (reverse) em vez de "
                    "congelar. A camera continua se movendo, entao o fim fica natural e o modelo continua "
                    "tendo movimento pra seguir. Velocidade preservada, sem corte duro.\n\n"
                    "cortar_ou_congelar = repete o ultimo frame pra completar. Camera PARADA no fim -- e ali "
                    "que o modelo costuma inventar conteudo, porque perde a referencia de movimento.\n\n"
                    "reamostrar = estica/comprime no tempo. Usa todo o conteudo, mas MUDA a velocidade da "
                    "camera. So quando a diferenca e grande e voce aceita alterar o ritmo.\n\n"
                    "so_avisar = nao mexe nos frames, so reporta a diferenca."}),
            },
        }

    RETURN_TYPES = ("INT", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("length", "frames", "duracao_real", "info")
    OUTPUT_TOOLTIPS = (
        "Comprimento valido -> ligue no 'length' do MiniMax H3 (e no Context-IR, se usar).",
        "Seus frames JA no tamanho exato (passthrough se nada foi ligado).",
        "Duracao real em segundos (length / 24).",
        "A grade em volta, o que foi ajustado e os avisos.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "H3 Frames · grade valida (Bruxos): o equivalente do '4n+1' do Wan para o MiniMax H3, cuja grade e "
        "length %% 17 == 5 (5, 22, 39, 56, 73, 90, 107, 124, 141, 192...). Encaixa duracao ou contagem de frames "
        "no valor valido, mostra os vizinhos, e AJUSTA o video de referencia pro mesmo tamanho -- que e o que "
        "evita a camera sair dessincronizada e o final do video inventar conteudo. "
        "8.0s = 192 frames e a unica duracao de 4-15s que cai num segundo exato."
    )

    def run(self, entrada, duracao_s, n_frames, arredondar="mais_proximo",
            frames=None, ajuste="pingpong"):
        avisos = []

        pedido = (float(duracao_s) * FPS) if entrada == "duracao_s" else float(n_frames)
        length = encaixar(pedido, arredondar)
        dur = length / FPS

        # ---- contexto: vizinhos na grade --------------------------------
        g = grade(5, 420)
        i = g.index(length)
        viz = g[max(0, i - 2): i + 3]
        viz_txt = "  ".join(f"[{v}={v/FPS:.2f}s]" if v == length else f"{v}={v/FPS:.2f}s" for v in viz)

        if abs(pedido - length) >= 0.5:
            avisos.append(f"pedido {pedido:.0f} frames -> ajustado pra {length} "
                          f"({'+' if length > pedido else ''}{length - pedido:.0f})")
        if dur < DUR_MIN or dur > DUR_MAX:
            avisos.append(f"DURACAO FORA DA FAIXA: {dur:.2f}s. O H3 foi treinado pra {DUR_MIN}-{DUR_MAX}s; "
                          f"fora disso a qualidade cai e ja vimos crash de kernel.")
        if length == 192:
            avisos.append("192 frames = 8.0000s exatos -- e a unica duracao redonda da faixa. Boa escolha "
                          "pra casar 1:1 com render de Blender.")

        # ---- ajusta os frames, se vieram --------------------------------
        saida = frames
        det = "sem frames ligados"
        if frames is not None:
            if not _OK:
                raise RuntimeError("[Bruxos H3 Frames] torch indisponivel.")
            if frames.ndim != 4:
                raise ValueError(f"[Bruxos H3 Frames] 'frames' precisa ser IMAGE [T,H,W,C]; "
                                 f"veio {tuple(frames.shape)}.")
            T = int(frames.shape[0])
            if T == length:
                det = f"{T} frames -> ja batia com {length}, nao mexi"
            elif ajuste == "so_avisar":
                det = f"{T} frames vs alvo {length} (NAO ajustado: modo 'so_avisar')"
                avisos.append(f"a referencia tem {T} frames e o alvo e {length}. "
                              f"{'Ela ACABA ANTES -- o modelo vai inventar o final.' if T < length else 'Sobra conteudo no fim.'}")
            elif ajuste == "pingpong":
                idx = torch.tensor(indices_pingpong(T, length), dtype=torch.long)
                saida = frames[idx]
                if T > length:
                    det = f"{T} -> {length} CORTANDO {T - length} frame(s) (pingpong so preenche, nao encolhe)"
                else:
                    voltas = (length - 1) // max(1, 2 * T - 2) + 1
                    det = (f"{T} -> {length} em PINGPONG: segue ate o fim e volta pelo caminho "
                           f"({length - T} frame(s) de retorno, {voltas} trecho(s))")
                    avisos.append(
                        f"o trecho final ({(length - T) / FPS:.2f}s) e o seu video DE TRAS PRA FRENTE. "
                        f"A camera continua se movendo -- que e o ponto -- mas ela REFAZ o caminho ao "
                        f"contrario. Se a trajetoria precisa ser sempre pra frente, exporte o render "
                        f"com {length} frames.")
            elif ajuste == "reamostrar":
                idx = torch.linspace(0, T - 1, length).round().long().clamp(0, T - 1)
                saida = frames[idx]
                det = (f"{T} -> {length} por REAMOSTRAGEM (velocidade do movimento mudou "
                       f"{length / T:.3f}x)")
                if abs(length / T - 1.0) > 0.15:
                    avisos.append(f"a reamostragem mudou a velocidade em {abs(1 - length / T) * 100:.0f}%. "
                                  f"Se o movimento de camera importa, prefira exportar o render "
                                  f"ja com {length} frames.")
            else:  # cortar_ou_congelar
                if T > length:
                    saida = frames[:length]
                    det = f"{T} -> {length} CORTANDO {T - length} frame(s) do fim"
                else:
                    falta = length - T
                    saida = torch.cat([frames, frames[-1:].repeat(falta, 1, 1, 1)], dim=0)
                    det = f"{T} -> {length} CONGELANDO o ultimo frame por {falta} frame(s)"
                    avisos.append(f"faltavam {falta} frames ({falta / FPS:.2f}s) e eu congelei o ultimo. "
                                  f"Camera parada no fim. Se preferir usar todo o conteudo, troque "
                                  f"'ajuste' pra 'reamostrar' -- ou exporte o render com {length} frames.")

        info = (f"length={length} ({dur:.4f}s @ {FPS:.0f}fps) | grade %{PASSO}=={RESTO}\n"
                f"vizinhos: {viz_txt}\n"
                f"frames: {det}")
        if avisos:
            info += "\n" + "\n".join(f"- {a}" for a in avisos)

        print(f"[Bruxos H3 Frames] length={length} ({dur:.4f}s) | {det}", flush=True)
        for a in avisos:
            print(f"[Bruxos H3 Frames]   - {a}", flush=True)

        return (int(length), saida, float(dur), info)


NODE_CLASS_MAPPINGS = {"BruxosH3Frames": BruxosH3Frames}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosH3Frames": "MiniMax H3 · Frames (grade válida) (Bruxos)"}
