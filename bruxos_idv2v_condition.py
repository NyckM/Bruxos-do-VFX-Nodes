# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — ID-V2V: sinal de controle "foreground-on-gray"
==============================================================
O QUE E
    O ID-V2V (Netflix/Eyeline Labs, SIGGRAPH Asia 2026) faz restylization
    preservando identidade. O checkpoint PADRAO (`idv2v`) condiciona em UM
    sinal VACE, e o README e explicito sobre qual:

        "foreground-on-gray pixels (SAM3-segmented person)"
        "keeps (relights) whatever is inside the SAM3 mask and FREELY
         REGENERATES the rest of the frame"

    Ou seja: a pessoa recortada, o resto CINZA. O cinza e o que autoriza o
    modelo a reconstruir o cenario a partir do keyframe estilizado.

O ERRO QUE ESTE NODE EVITA
    Mandar o video CRU como control_video. Isso nao e um atalho -- e a receita
    de RELIGHTING do proprio repo:

        "no SAM3 mask, no grayed-out background, and no preprocessing at all.
         The raw source video is used as the condition, SO THE WHOLE SCENE IS
         PRESERVED and only the lighting changes."

    O sintoma e caracteristico: o keyframe manda no frame 0 e vai sendo
    sobrescrito ao longo do clipe. Nao e a forca do keyframe caindo -- e o
    fundo original sendo reafirmado pelo VACE a cada bloco, em todos os steps.

LIGACAO
    Load Video ─ IMAGE ─┬─────────────────────────> [frames]
                        └─> SAM3 (prompt "person") ─> [mask]
                                                        │
                            [este node] ────────────────┘
                                  │ condition
                                  v
                        WanVaceToVideo.control_video

POR QUE 0.5 E NAO PRETO
    O WanVaceToVideo faz `control_video - 0.5` antes de codificar. 0.5 e o
    neutro exato do espaco dele: vira zero, ou seja "sem informacao". Preto
    (0.0) vira -0.5 e o modelo le como conteudo escuro de verdade.
"""

import logging

try:
    import torch
    import torch.nn.functional as F
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

try:
    import cv2
    import numpy as np
    _CV2 = True
except Exception:  # pragma: no cover
    _CV2 = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Wan"

CINZA = 0.5   # neutro do VACE (ver docstring)


def _norm_mask(m, T, H, W):
    """MASK [H,W] / [T,H,W] / IMAGE [T,H,W,C] -> [T,H,W] em 0..1, no tamanho."""
    if m.ndim == 4:
        m = m[..., :3].amax(dim=-1)
    elif m.ndim == 2:
        m = m.unsqueeze(0)
    m = m.float().clamp(0, 1)
    if int(m.shape[-2]) != H or int(m.shape[-1]) != W:
        m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear",
                          align_corners=False).squeeze(1)
    Tm = int(m.shape[0])
    if Tm == 1 and T > 1:
        m = m.repeat(T, 1, 1)
    elif Tm != T:
        idx = torch.linspace(0, Tm - 1, T).round().long().clamp(0, Tm - 1)
        m = m[idx]
    return m


def _grow_blur(m, grow=0, blur=0):
    x = m.unsqueeze(1)
    g = int(grow)
    if g > 0:
        x = F.max_pool2d(x, kernel_size=g * 2 + 1, stride=1, padding=g)
    elif g < 0:
        a = -g
        x = -F.max_pool2d(-x, kernel_size=a * 2 + 1, stride=1, padding=a)
    b = int(blur)
    if b > 0:
        k = b * 2 + 1
        co = torch.arange(k, dtype=torch.float32, device=x.device) - b
        g1 = torch.exp(-(co ** 2) / (2 * (b * 0.5 + 1e-6) ** 2))
        g1 = g1 / g1.sum()
        x = F.conv2d(x, g1.view(1, 1, 1, k), padding=(0, b))
        x = F.conv2d(x, g1.view(1, 1, k, 1), padding=(b, 0))
    return x.squeeze(1).clamp(0, 1)


def _forma_bbox(m):
    """[T,H,W] 0..1 -> caixa envolvente por frame. Puro torch, sem cv2.

    Vem do MaskAugAnnotator do VACE (modo 'bbox'): trocar o recorte exato por
    um container frouxo. Frame sem nada acesso fica vazio, igual ao original
    (la o get_mask_info devolve valid=False e a mascara passa intacta).
    """
    out = torch.zeros_like(m)
    for t in range(int(m.shape[0])):
        nz = m[t] > 0.5
        if not bool(nz.any()):
            out[t] = m[t]
            continue
        linhas = torch.nonzero(nz.any(dim=1)).flatten()
        colunas = torch.nonzero(nz.any(dim=0)).flatten()
        out[t, int(linhas[0]):int(linhas[-1]) + 1,
            int(colunas[0]):int(colunas[-1]) + 1] = 1.0
    return out


def _forma_hull(m):
    """[T,H,W] 0..1 -> fecho convexo por frame (VACE: generate_hull_mask).

    O VACE concatena TODOS os contornos antes do convexHull, entao duas pessoas
    separadas viram um poligono so. Mantive igual de proposito -- e o
    comportamento com que o modelo foi treinado.
    """
    if not _CV2:
        print("[Bruxos ID-V2V] AVISO: cv2 indisponivel, 'hull' caiu pra 'bbox'.", flush=True)
        return _forma_bbox(m)
    dev, dt = m.device, m.dtype
    arr = (m.detach().cpu().numpy() > 0.5).astype(np.uint8)
    saida = np.zeros_like(arr)
    for t in range(arr.shape[0]):
        cont, _ = cv2.findContours(arr[t], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not cont:
            saida[t] = arr[t]
            continue
        hull = cv2.convexHull(np.concatenate(cont))
        cv2.fillPoly(saida[t], [hull], 1)
    return torch.from_numpy(saida).to(device=dev, dtype=dt)


def _parse_indices(txt, n_imgs, T):
    """'0, 40, 80' -> [0, 40, 80]. Tolera ponto-e-virgula, espaco e vazio."""
    if not txt or not str(txt).strip():
        return []
    bruto = str(txt).replace(";", ",").replace(" ", ",")
    idx = []
    for p in bruto.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            idx.append(int(p))
        except ValueError:
            raise ValueError(f"[Bruxos ID-V2V] 'keyframes_idx' tem um valor que nao e "
                             f"numero inteiro: {p!r}. Formato esperado: '0, 40, 80'.")
    fora = [i for i in idx if i < 0 or i >= T]
    if fora:
        raise ValueError(f"[Bruxos ID-V2V] indices de keyframe fora do clipe (0..{T - 1}): "
                         f"{fora}. Lembre que o clipe nativo do ID-V2V tem 81 frames -- um "
                         f"keyframe no frame 80 e o ULTIMO, nao o de numero 80 do video inteiro.")
    if len(idx) != n_imgs:
        raise ValueError(f"[Bruxos ID-V2V] voce ligou {n_imgs} imagem(ns) em 'keyframes' mas "
                         f"listou {len(idx)} indice(s) em 'keyframes_idx'. Precisa ser um indice "
                         f"por imagem, na mesma ordem do batch.")
    return idx


class BruxosIDV2VCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip":
                    "Os frames do VIDEO DE ORIGEM, ja na resolucao de geracao (mesma que vai pro "
                    "WanVaceToVideo)."}),
                "mask": ("MASK", {"tooltip":
                    "Mascara do SUJEITO, por frame -- a saida do SAM3 com prompt 'person'. "
                    "BRANCO = a pessoa (preservada e relightada); PRETO = o que o modelo pode "
                    "regenerar a partir do keyframe.\n"
                    "Se vier invertida, use o 'inverter'."}),
            },
            "optional": {
                "estrutura": ("IMAGE", {"tooltip":
                    "[opcional] Um mapa de ESTRUTURA por frame pra ancorar o movimento de camera SEM "
                    "trazer a aparencia do cenario original. Canny, depth, normais, linhas, ou o "
                    "visualizador de tracking -- qualquer coisa que descreva geometria e se mova "
                    "junto com a camera.\n\n"
                    "E melhor que o 'fundo_forca' pra isso: o RGB original carrega cor e textura "
                    "junto e briga com o keyframe; canny e depth nao carregam nada disso.\n\n"
                    "Vai SO no fundo (fora da mascara). O sujeito nunca e tocado."}),
                "keyframe": ("IMAGE", {"tooltip":
                    "[opcional] A imagem estilizada (a mesma do 'start_image'). So e usada pra MEDIR "
                    "exposicao ou cor media do fundo -- nenhum pixel dela e copiado pro controle. "
                    "Necessaria quando 'fundo_modo' nao e 'cinza_fixo'."}),
                "inverter": ("BOOLEAN", {"default": False, "tooltip":
                    "Troca sujeito por fundo. Ligue se a sua mascara veio com a pessoa em preto."}),
                "crescer": ("INT", {"default": 4, "min": -64, "max": 128, "step": 1, "tooltip":
                    "Dilata (+) ou contrai (-) a mascara, em pixels. Uma folguinha positiva pega "
                    "fios de cabelo e borda de roupa que o SAM3 costuma cortar rente -- e ali que "
                    "aparece halo depois."}),
                "suavizar": ("INT", {"default": 3, "min": 0, "max": 64, "step": 1, "tooltip":
                    "Feather da borda, em pixels. Poucos pixels bastam: borda dura demais vira "
                    "recorte visivel, borda macia demais deixa o fundo original vazar como fantasma."}),
                "cinza": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                    "O valor do fundo. DEIXE EM 0.5.\n"
                    "O WanVaceToVideo faz 'control_video - 0.5' antes de codificar, entao 0.5 vira "
                    "exatamente zero: 'sem informacao aqui, gere o que quiser'. Preto (0.0) vira "
                    "-0.5 e o modelo entende como conteudo escuro REAL."}),
                # ---------------------------------------------------------
                # APPEND-ONLY: widgets NOVOS vao SEMPRE no FIM desta lista.
                # O ComfyUI casa os widgets_values salvos por ORDEM, nao por
                # nome. Inserir no meio desloca TODOS os valores dos grafos ja
                # salvos -- foi exatamente o que aconteceu quando eu pus o
                # 'fundo_forca' antes do 'cinza': um grafo salvo com
                # [inverter, crescer, suavizar, cinza=0.5] passou a ler
                # fundo_forca=0.5, ligando meio fundo original sem o usuario
                # pedir. Acrescente ABAIXO desta nota.
                # ---------------------------------------------------------
                "fundo_forca": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "QUANTO DO FUNDO ORIGINAL SOBREVIVE no controle. E o dial entre liberdade de estilo e "
                    "fidelidade de CAMERA.\n\n"
                    "0.00 = cinza puro. E a receita oficial de restylization: o cenario e reconstruido do "
                    "zero pelo keyframe. MAS o movimento de camera some junto -- paralaxe e percebida pelo "
                    "FUNDO, e sem fundo nao ha pista de que a camera gira.\n"
                    "0.15-0.35 = fantasma do fundo original. Pouco contraste, quase invisivel como aparencia, "
                    "mas geometria suficiente pra camera acompanhar. Comece aqui se o movimento sumiu.\n"
                    "1.00 = fundo original inteiro. Vira a receita de RELIGHTING: a cena e preservada e o "
                    "keyframe so muda a luz.\n\n"
                    "Nota: o variante oficial 'idv2v_with_normal_depth' resolve isso com profundidade, que "
                    "carrega geometria SEM aparencia. Esse checkpoint nao foi convertido pro ComfyUI, entao "
                    "este dial e a aproximacao possivel -- ele vaza um pouco de aparencia junto."}),
                "fundo_modo": (["cinza_fixo", "luminancia_do_keyframe", "cor_do_keyframe"],
                    {"default": "cinza_fixo", "tooltip":
                    "DE ONDE SAI O VALOR DO FUNDO.\n\n"
                    "cinza_fixo = usa o numero do widget 'cinza'. 0.50 e o neutro do VACE.\n\n"
                    "luminancia_do_keyframe = mede a luminancia MEDIA do keyframe e usa ela. Se o seu "
                    "keyframe e uma cena noturna, o fundo do controle fica escuro no mesmo nivel -- o "
                    "modelo ja parte da exposicao certa em vez de partir do meio-tom e ter que escurecer.\n\n"
                    "cor_do_keyframe = usa a COR media (RGB) do keyframe, nao so o brilho. Alem da "
                    "exposicao, leva a dominante -- laranja num incendio, azul numa noite fria. "
                    "E o que mais ajuda a cena gerada nascer na paleta certa.\n\n"
                    "Os dois ultimos precisam do 'keyframe' ligado."}),
                "desvio": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01, "tooltip":
                    "Empurra o valor final do fundo pra mais claro (+) ou mais escuro (-), depois do "
                    "modo escolhido. Use pra afinar sem trocar de modo.\n"
                    "LEMBRE: quanto mais longe de 0.50, MENOS o fundo significa 'vazio' e mais ele "
                    "significa 'conteudo escuro/claro de verdade' -- porque o VACE subtrai 0.5 antes "
                    "de codificar. Isso e util pra guiar exposicao, mas em excesso vira instrucao de "
                    "pintar uma parede lisa."}),
                "estrutura_modo": (["centrado", "aditivo", "bruto"], {"default": "centrado", "tooltip":
                    "COMO o mapa de estrutura vira sinal. Isso decide se ele guia ou manda.\n\n"
                    "centrado (DEPTH, NORMAIS) = subtrai a media do proprio mapa e mantem so a "
                    "VARIACAO. Area plana vira 0.5 exato (nada dito); relevo vira desvio com sinal. "
                    "E o que voce quer pra paralaxe: o modelo ganha a estrutura 3D e o movimento sem "
                    "receber ordem de exposicao.\n\n"
                    "aditivo (CANNY, LINEART, TRACKS) = preto nao diz nada, branco soma. O fundo fica "
                    "0.5 em quase tudo e so as ARESTAS marcam. Perfeito pra mapa esparso.\n\n"
                    "bruto = usa o mapa como imagem, misturado ate o cinza pela forca. Use so se o "
                    "seu mapa ja estiver no range certo."}),
                "estrutura_forca": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip":
                    "QUANTO a estrutura se afasta do neutro 0.5 -- ou seja, quanta ordem ela e.\n\n"
                    "0.10-0.25 = sugestao. A camera acompanha e o modelo continua livre pra inventar "
                    "o cenario. COMECE AQUI.\n"
                    "0.4-0.6 = o contorno comeca a aparecer no resultado; util quando voce QUER a "
                    "arquitetura original mantida.\n"
                    "1.0 = amplitude cheia. Vira controlnet: o modelo desenha em cima das linhas.\n\n"
                    "Lembre da conta do VACE: fundo = 0.5 + forca x sinal, e o VACE subtrai 0.5. "
                    "Com forca 0.2, uma aresta forte vira +0.2 no espaco dele -- presente, mas fraca "
                    "o bastante pra ser reinterpretada."}),
                "keyframes": ("IMAGE", {"tooltip":
                    "[opcional] Batch de frames ESTILIZADOS pra fixar em posicoes especificas do "
                    "clipe -- o caso 'first_last_frame' do repo oficial. Um indice por imagem em "
                    "'keyframes_idx', na mesma ordem.\n\n"
                    "Nao precisa incluir o frame 0: ele ja e fixado pelo 'start_image' do "
                    "WanImageToVideo. Use aqui pro MEIO e pro FIM.\n\n"
                    "Exige que a saida 'control_mask' esteja ligada no 'control_masks' do "
                    "WanVaceToVideo -- e a mascara que faz o pin acontecer."}),
                "keyframes_idx": ("STRING", {"default": "", "multiline": False, "tooltip":
                    "Indices 0-based, separados por virgula: '80' fixa o ultimo frame de um clipe "
                    "de 81; '40, 80' fixa o meio e o fim.\n\n"
                    "COMO FUNCIONA (FrameRefExtractAnnotator do VACE): no frame fixado o controle "
                    "recebe o pixel do keyframe e a control_mask vai a ZERO -- e mask 0 que quer "
                    "dizer 'isto aqui e conteudo real, preserve'. Nos outros frames a mask fica em "
                    "1 e o modelo gera livre."}),
                "mask_modo": (["original", "hull", "bbox"], {"default": "original", "tooltip":
                    "O FORMATO da mascara do sujeito. O VACE foi treinado com os tres (MaskAugAnnotator), "
                    "entao afrouxar nao e gambiarra -- e in-distribution.\n\n"
                    "original = o recorte do SAM3 como veio. Maxima liberdade pro cenario, mas borda "
                    "dura: se o SAM3 corta rente, aparece halo e o sujeito parece colado.\n\n"
                    "hull = fecho convexo do recorte. Envolve o corpo inteiro sem seguir cada dedo e "
                    "cada fio -- some com o aspecto recortado. USE QUANDO o resultado parecer colagem. "
                    "Custo: um pouco do fundo original entra junto, perto do corpo.\n\n"
                    "bbox = caixa envolvente. O mais frouxo; preserva um retangulo inteiro em volta da "
                    "pessoa. So quando 'hull' ainda nao resolver -- come bastante cenario.\n\n"
                    "Aplicado ANTES de 'crescer' e 'suavizar', na mesma ordem do VACE."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "MASK")
    RETURN_NAMES = ("condition", "mask_usada", "info", "control_mask")
    OUTPUT_TOOLTIPS = (
        "O sinal de controle -> ligue no 'control_video' do WanVaceToVideo.",
        "A mascara depois de crescer/suavizar, pra voce conferir num Preview.",
        "Cobertura do sujeito e avisos.",
        "Mascara TEMPORAL -> ligue no 'control_masks' do WanVaceToVideo. Fica 1 (gerar) em todo "
        "frame, e 0 (preservar) nos frames fixados por 'keyframes_idx'. Sem keyframes ela e toda "
        "1, que e o comportamento padrao -- pode ligar sempre, nao muda nada.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "ID-V2V · Condicao foreground-on-gray (Bruxos): monta o sinal de controle que o checkpoint padrao do "
        "ID-V2V espera -- a pessoa segmentada preservada, o resto em cinza 0.5. E o cinza que autoriza o modelo "
        "a regenerar o cenario a partir do keyframe estilizado. "
        "Mandar o video CRU no control_video e a receita de RELIGHTING do repo oficial, que preserva a cena "
        "inteira de proposito -- por isso o estilo do keyframe 'some' ao longo do clipe."
    )

    def run(self, frames, mask, estrutura=None, keyframe=None, inverter=False, crescer=4,
            suavizar=3, fundo_forca=0.0, cinza=0.5, fundo_modo="cinza_fixo", desvio=0.0,
            estrutura_modo="centrado", estrutura_forca=0.20, keyframes=None,
            keyframes_idx="", mask_modo="original"):
        if not _OK:
            raise RuntimeError("[Bruxos ID-V2V] torch indisponivel.")
        if frames.ndim != 4:
            raise ValueError(f"[Bruxos ID-V2V] 'frames' precisa ser IMAGE [T,H,W,C]; "
                             f"veio {tuple(frames.shape)}.")

        T, H, W, C = (int(v) for v in frames.shape)
        m = _norm_mask(mask, T, H, W)
        if inverter:
            m = 1.0 - m
        # ordem do VACE: forma (hull/bbox) PRIMEIRO, expansao depois.
        if mask_modo == "hull":
            m = _forma_hull(m)
        elif mask_modo == "bbox":
            m = _forma_bbox(m)
        if int(crescer) != 0 or int(suavizar) > 0:
            m = _grow_blur(m, int(crescer), int(suavizar))

        # ---- de onde sai o valor do fundo -------------------------------
        # Rec.709: o olho pesa verde muito mais que azul. Media simples de RGB
        # daria uma "luminancia" errada -- ceu azul pareceria mais claro do que e.
        nota_modo = f"cinza fixo {float(cinza):.3f}"
        base = torch.tensor([float(cinza)] * 3, dtype=torch.float32)
        if fundo_modo != "cinza_fixo":
            if keyframe is None:
                print(f"[Bruxos ID-V2V] AVISO: fundo_modo='{fundo_modo}' precisa do 'keyframe' "
                      f"ligado. Caindo pra cinza fixo {float(cinza):.2f}.", flush=True)
                nota_modo = f"cinza fixo {float(cinza):.3f} (keyframe ausente)"
            else:
                kf = keyframe[0, :, :, :3].float()
                if fundo_modo == "cor_do_keyframe":
                    base = kf.reshape(-1, 3).mean(dim=0)
                    nota_modo = (f"cor media do keyframe "
                                 f"R{base[0]:.2f} G{base[1]:.2f} B{base[2]:.2f}")
                else:  # luminancia_do_keyframe
                    lum = float((kf[..., 0] * 0.2126 + kf[..., 1] * 0.7152 +
                                 kf[..., 2] * 0.0722).mean())
                    base = torch.tensor([lum] * 3, dtype=torch.float32)
                    nota_modo = f"luminancia do keyframe {lum:.3f} (Rec.709)"

        base = (base + float(desvio)).clamp(0, 1).to(frames.dtype).to(frames.device)
        lum_final = float(base[0] * 0.2126 + base[1] * 0.7152 + base[2] * 0.0722)

        a = m.unsqueeze(-1).to(frames.dtype)
        # o fundo vai de cinza puro (f=0) ate o original (f=1). Valores baixos
        # deixam so um fantasma: geometria suficiente pra camera, aparencia
        # fraca demais pra competir com o keyframe.
        f = float(fundo_forca)
        fundo = base + f * (frames - base)

        # ---- estrutura: geometria SEM aparencia -------------------------
        # A conta toda vive em torno de 0.5 porque o VACE subtrai 0.5 antes de
        # codificar. Entao "quanto o sinal se afasta de 0.5" E "quanta ordem
        # ele e". Forca baixa = sugestao; forca alta = controlnet.
        nota_est = "sem estrutura"
        ef = float(estrutura_forca)
        if estrutura is not None and ef > 1e-6:
            e = estrutura
            if int(e.shape[1]) != H or int(e.shape[2]) != W:
                e = F.interpolate(e.permute(0, 3, 1, 2).float(), size=(H, W),
                                  mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            e = e[..., :3].float().clamp(0, 1)
            Te = int(e.shape[0])
            if Te == 1 and T > 1:
                e = e.repeat(T, 1, 1, 1)
            elif Te != T:
                idx = torch.linspace(0, Te - 1, T).round().long().clamp(0, Te - 1)
                e = e[idx]
                print(f"[Bruxos ID-V2V] estrutura tinha {Te} frames e o video tem {T} -> "
                      f"reamostrada. Se o mapa nao acompanhar a camera, gere ele com a MESMA "
                      f"contagem de frames.", flush=True)
            if estrutura_modo == "centrado":
                # media POR FRAME: area plana vira 0 (nada dito), relevo vira desvio com sinal
                media = e.mean(dim=(1, 2), keepdim=True)
                sinal = (e - media) * 2.0
                nota_est = f"centrado (depth/normais) forca {ef:.2f}"
            elif estrutura_modo == "aditivo":
                sinal = e                      # preto = 0 = nada; branco = +1
                nota_est = f"aditivo (canny/tracks) forca {ef:.2f}"
            else:
                sinal = (e - 0.5) * 2.0
                nota_est = f"bruto forca {ef:.2f}"
            fundo = (fundo + ef * sinal).clamp(0, 1)

        cond = frames * a + fundo * (1.0 - a)

        # ---- keyframes: pin temporal (FrameRefExtractAnnotator do VACE) ---
        # No frame fixado: pixel REAL do keyframe + control_mask 0 ("preserve").
        # Nos demais: o que ja montamos + control_mask 1 ("gere"). E so isso --
        # nao ha nada de arquitetural em fixar frame, e a mascara temporal.
        control_mask = torch.ones((T, H, W), dtype=torch.float32, device=frames.device)
        nota_kf = "nenhum"
        if keyframes is not None:
            n_kf = int(keyframes.shape[0])
            idx = _parse_indices(keyframes_idx, n_kf, T)
            for j, i in enumerate(idx):
                kfi = keyframes[j:j + 1, :, :, :3].float()
                if int(kfi.shape[1]) != H or int(kfi.shape[2]) != W:
                    kfi = F.interpolate(kfi.permute(0, 3, 1, 2), size=(H, W),
                                        mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                cond[i] = kfi[0].to(cond.dtype).to(cond.device)
                control_mask[i] = 0.0
            nota_kf = f"{len(idx)} fixado(s) em {idx}"
        elif str(keyframes_idx).strip():
            print("[Bruxos ID-V2V] AVISO: 'keyframes_idx' preenchido mas nenhuma imagem ligada "
                  "em 'keyframes'. Ignorado.", flush=True)

        cob = float(m.mean()) * 100.0
        avisos = []
        if cob < 1.0:
            avisos.append(f"a mascara cobre so {cob:.1f}% do quadro -- praticamente TUDO vira cinza e "
                          f"o modelo perde a ancora de identidade. Confira o prompt do SAM3 e o 'inverter'.")
        elif cob > 85.0:
            avisos.append(f"a mascara cobre {cob:.1f}% do quadro -- quase nada fica cinza, entao sobra "
                          f"pouca liberdade pro keyframe reconstruir a cena. Se a pessoa nao ocupa tudo isso, "
                          f"a mascara provavelmente esta invertida.")
        if abs(lum_final - 0.5) > 0.12:
            avisos.append(
                f"o fundo ficou com luminancia {lum_final:.2f}, longe do neutro 0.50. O VACE subtrai "
                f"0.5 antes de codificar, entao isso NAO e mais 'vazio' -- e uma instrucao de "
                f"{'claro' if lum_final > 0.5 else 'escuro'} de verdade. Otimo pra guiar exposicao; "
                f"em excesso o modelo pinta uma superficie lisa no lugar de cenario.")

        if f >= 0.95:
            avisos.append("fundo_forca perto de 1.0: isso e a receita de RELIGHTING (cena preservada). "
                          "Se voce quer trocar o cenario pelo keyframe, baixe pra 0.0-0.3.")
        if mask_modo != "original" and cob > 60.0:
            avisos.append(f"mask_modo='{mask_modo}' inflou a mascara pra {cob:.1f}% do quadro. "
                          f"Sobra pouco cinza pro keyframe reconstruir. Se ficou assim, volte pra "
                          f"'hull' (se estava em bbox) ou pra 'original'.")
        info = (f"{T} frames {W}x{H} | sujeito {cob:.1f}% ({mask_modo}) | estrutura: {nota_est} | "
                f"keyframes: {nota_kf} | fundo: {nota_modo}"
                f"{f' desvio {desvio:+.2f}' if abs(desvio) > 1e-6 else ''} "
                f"-> lum {lum_final:.3f} | fundo_forca {f:.2f} | "
                f"crescer {crescer} | suavizar {suavizar}")
        print(f"[Bruxos ID-V2V] {info}", flush=True)
        for x in avisos:
            print(f"[Bruxos ID-V2V]   AVISO: {x}", flush=True)
        if avisos:
            info += "\n" + "\n".join(f"- {x}" for x in avisos)
        return (cond.clamp(0, 1), m, info, control_mask)


NODE_CLASS_MAPPINGS = {"BruxosIDV2VCondition": BruxosIDV2VCondition}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosIDV2VCondition": "ID-V2V · Condição foreground-on-gray (Bruxos)"
}
