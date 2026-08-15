# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Inpaint por BBox: cortar pela mascara, gerar, colar de volta
============================================================================
POR QUE ESTE PAR EXISTE
    O MiniMax H3 NAO tem entrada de mascara. Conferido no core:

        comfy/model_base.py:2071  class MiniMaxH3.extra_conds
            minimax_token_tags | minimax_keyframes | minimax_refs
            minimax_visual_cond_noise_aug | minimax_audio_cond_noise_aug | seed

    Nao ha canal de inpaint como no VACE. Entao o caminho e geometrico:
    corta a regiao da mascara, gera SO ela na resolucao cheia do modelo, e
    recompoe. De quebra a regiao ganha muito mais pixels do que teria no
    quadro inteiro -- um rosto que ocupava 8% do frame passa a ocupar 100%.

O QUE ISTO NAO E
    Nao e inpaint de verdade: o modelo nao ve o entorno e nao sabe que esta
    preenchendo um buraco. Ele gera o recorte inteiro. A costura de volta e
    responsabilidade do Stitch, e e ali que mora a diferenca entre "colado" e
    "integrado".

A DECISAO QUE MAIS IMPORTA: UNIAO vs POR FRAME
    Mascara de video mexe a cada frame. Se o corte seguisse cada uma, o
    recorte tremeria e o modelo receberia um enquadramento instavel -- que e
    exatamente o que faz gerador de video derivar.
    Por isso o padrao e UNIAO: uma caixa so, que cobre a mascara em TODOS os
    frames. Corte estavel, movimento preservado dentro dele.
"""

import logging

try:
    import torch
    import torch.nn.functional as F
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Mascara"

_ASPECTOS = ["livre", "original", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9"]
_RAZAO = {"1:1": 1.0, "16:9": 16/9, "9:16": 9/16, "4:3": 4/3, "3:4": 3/4, "21:9": 21/9}


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


def _caixa(mask2d, limiar):
    """[H,W] -> (x0,y0,x1,y1) inclusivo, ou None se a mascara estiver vazia."""
    nz = mask2d > limiar
    if not bool(nz.any()):
        return None
    linhas = torch.nonzero(nz.any(dim=1)).flatten()
    colunas = torch.nonzero(nz.any(dim=0)).flatten()
    return (int(colunas[0]), int(linhas[0]), int(colunas[-1]), int(linhas[-1]))


class BruxosMaskBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "imagens": ("IMAGE", {"tooltip": "Os frames originais [B,H,W,C]."}),
                "mascara": ("MASK", {"tooltip":
                    "A mascara do SAM3 (ou qualquer outra). BRANCO = a regiao que vai ser "
                    "recortada e regenerada."}),
                "margem": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip":
                    "Folga em volta da caixa, como fracao do lado dela. 0.25 = 25% de cada lado.\n\n"
                    "NAO deixe em 0. O modelo precisa ver CONTEXTO em volta do alvo -- um rosto "
                    "recortado rente gera sem pescoco, sem cabelo e sem a luz do ambiente, e a "
                    "colagem denuncia. 0.2-0.4 costuma ser o certo para rosto."}),
                "multiplo": ("INT", {"default": 32, "min": 1, "max": 64, "step": 1, "tooltip":
                    "Arredonda o recorte para multiplo disso. 32 para MiniMax H3 e LTX, 16 para "
                    "Wan 2.1/2.2-14B. Fora da grade o encoder arredonda por conta e o Stitch "
                    "volta desalinhado."}),
            },
            "optional": {
                "modo": (["uniao", "por_frame"], {"default": "uniao", "tooltip":
                    "uniao = UMA caixa cobrindo a mascara de TODOS os frames. Corte estavel, e o "
                    "movimento do sujeito acontece dentro dele. E o que voce quer em video.\n\n"
                    "por_frame = a caixa segue a mascara quadro a quadro. O recorte TREME, e "
                    "enquadramento instavel e justamente o que faz gerador de video derivar. "
                    "So use para imagem unica."}),
                "aspecto": (_ASPECTOS, {"default": "livre", "tooltip":
                    "Trava a proporcao do recorte, expandindo o lado menor.\n"
                    "'original' copia a proporcao do quadro de entrada -- util quando o modelo "
                    "foi treinado naquela proporcao."}),
                "lado_minimo": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32,
                    "tooltip":
                    "Piso para o menor lado do recorte, em pixels. Evita mandar um recorte "
                    "minusculo para o modelo quando a mascara e pequena. 0 = sem piso."}),
                "limiar": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "A partir de que valor um pixel da mascara conta como dentro."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BRUXOS_BBOX", "INT", "INT", "STRING")
    RETURN_NAMES = ("recorte", "mascara_recorte", "bbox", "largura", "altura", "info")
    OUTPUT_TOOLTIPS = (
        "Os frames cortados -> mande para o gerador.",
        "A mascara cortada junto, no mesmo enquadramento.",
        "A geometria do corte -> ligue no Stitch. NAO edite no meio do caminho.",
        "Largura do recorte (ja na grade).",
        "Altura do recorte.",
        "Tamanho, posicao e avisos.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Mascara -> BBox -> Recorte (Bruxos): corta a regiao da mascara para voce gerar SO ela na "
        "resolucao cheia do modelo. O MiniMax H3 nao tem entrada de mascara -- este e o caminho. "
        "Use com o 'Stitch' para recompor."
    )

    def run(self, imagens, mascara, margem=0.25, multiplo=32, modo="uniao",
            aspecto="livre", lado_minimo=0, limiar=0.5):
        if not _OK:
            raise RuntimeError("[Bruxos BBox] torch indisponivel.")
        if imagens.ndim != 4:
            raise ValueError("[Bruxos BBox] 'imagens' precisa ser IMAGE [B,H,W,C]; veio %s."
                             % (tuple(imagens.shape),))
        T, H, W, _ = (int(v) for v in imagens.shape)
        m = _norm_mask(mascara, T, H, W)
        avisos = []

        if modo == "por_frame" and T > 1:
            avisos.append("modo 'por_frame' com %d frames: o recorte vai TREMER e o gerador "
                          "tende a derivar. Em video use 'uniao'." % T)

        # ---- caixa bruta -------------------------------------------------
        if modo == "uniao" or T == 1:
            cx = _caixa(m.amax(dim=0), limiar)
        else:
            caixas = [c for c in (_caixa(m[t], limiar) for t in range(T)) if c]
            cx = (min(c[0] for c in caixas), min(c[1] for c in caixas),
                  max(c[2] for c in caixas), max(c[3] for c in caixas)) if caixas else None
        if cx is None:
            raise ValueError(
                "[Bruxos BBox] a mascara esta VAZIA (nenhum pixel acima de %.2f). "
                "Confira o prompt do SAM3, ou se a mascara nao veio invertida." % limiar)

        x0, y0, x1, y1 = cx
        cw, ch = (x1 - x0 + 1), (y1 - y0 + 1)
        cobertura = 100.0 * cw * ch / (W * H)

        # ---- margem ------------------------------------------------------
        mx, my = cw * float(margem), ch * float(margem)
        x0, x1 = x0 - mx, x1 + mx
        y0, y1 = y0 - my, y1 + my

        # ---- aspecto -----------------------------------------------------
        alvo = None
        if aspecto == "original":
            alvo = W / H
        elif aspecto in _RAZAO:
            alvo = _RAZAO[aspecto]
        if alvo:
            cw, ch = (x1 - x0), (y1 - y0)
            cx_c, cy_c = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            if cw / max(ch, 1e-6) < alvo:
                cw = ch * alvo
            else:
                ch = cw / alvo
            x0, x1 = cx_c - cw / 2, cx_c + cw / 2
            y0, y1 = cy_c - ch / 2, cy_c + ch / 2

        # ---- lado minimo, grade, e prender dentro do quadro ---------------
        def _fechar(a, b, limite):
            """Arredonda o tamanho pra grade e encaixa dentro de [0, limite]."""
            tam = int(round(b - a))
            if lado_minimo > 0:
                tam = max(tam, int(lado_minimo))
            tam = max(int(multiplo), (tam // int(multiplo)) * int(multiplo))
            if tam > limite:                       # nao cabe: usa o limite na grade
                tam = max(int(multiplo), (limite // int(multiplo)) * int(multiplo))
            centro = (a + b) / 2.0
            ini = int(round(centro - tam / 2.0))
            ini = max(0, min(ini, limite - tam))
            return ini, ini + tam

        nx0, nx1 = _fechar(x0, x1, W)
        ny0, ny1 = _fechar(y0, y1, H)
        rw, rh = nx1 - nx0, ny1 - ny0

        recorte = imagens[:, ny0:ny1, nx0:nx1, :].contiguous()
        mrec = m[:, ny0:ny1, nx0:nx1].contiguous()

        bbox = {"x": nx0, "y": ny0, "w": rw, "h": rh, "W": W, "H": H, "T": T,
                "multiplo": int(multiplo)}

        if rw < 64 or rh < 64:
            avisos.append("recorte de %dx%d e minusculo; suba 'lado_minimo' ou a 'margem', "
                          "senao o gerador recebe quase nenhuma informacao." % (rw, rh))
        if float(margem) < 0.05:
            avisos.append("margem quase zero: o modelo nao vera contexto em volta do alvo e a "
                          "colagem tende a denunciar (sem pescoco, sem cabelo, luz errada).")
        if rw >= W and rh >= H:
            avisos.append("o recorte virou o quadro INTEIRO -- a margem ou o aspecto comeram "
                          "toda a economia. Reduza a margem.")

        ganho = (rw * rh) / max(1.0, cw * ch) if cw and ch else 1.0
        info = ("mascara cobria %.1f%% do quadro | recorte %dx%d em (%d,%d) | grade %d | %s"
                % (cobertura, rw, rh, nx0, ny0, multiplo, modo))
        print("[Bruxos BBox] %s" % info, flush=True)
        for a in avisos:
            print("[Bruxos BBox]   AVISO: %s" % a, flush=True)
        if avisos:
            info += "\n" + "\n".join("- %s" % a for a in avisos)
        return (recorte, mrec, bbox, rw, rh, info)


class BruxosMaskStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "originais": ("IMAGE", {"tooltip":
                    "Os frames ORIGINAIS inteiros -- os mesmos que entraram no BBox."}),
                "gerado": ("IMAGE", {"tooltip":
                    "O resultado do gerador sobre o recorte. Pode vir em outro tamanho: e "
                    "reescalado para a caixa automaticamente."}),
                "bbox": ("BRUXOS_BBOX", {"tooltip":
                    "A saida 'bbox' do node de recorte. E ela que sabe onde colar."}),
                "suavizar": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1, "tooltip":
                    "Feather da borda de colagem, em pixels.\n\n"
                    "E o que separa 'colado' de 'integrado'. Zero deixa uma linha reta visivel "
                    "no contorno da caixa. Comece em 10-20."}),
            },
            "optional": {
                "mascara": ("MASK", {"tooltip":
                    "[opcional] A mascara do SUJEITO, no quadro inteiro. Se ligada, a colagem "
                    "acontece SO dentro dela (mais o feather) em vez de na caixa toda -- o "
                    "entorno dentro da caixa volta a ser o original.\n\n"
                    "Use quando so o sujeito devia mudar. Sem ela, tudo dentro da caixa e "
                    "substituido."}),
                "expandir": ("INT", {"default": 0, "min": -64, "max": 128, "step": 1, "tooltip":
                    "Dilata (+) ou contrai (-) a mascara antes do feather, em pixels. "
                    "So vale se a 'mascara' estiver ligada."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Stitch pelo BBox (Bruxos): cola o recorte gerado de volta no quadro inteiro, com feather "
        "na borda. Par do 'Mascara -> BBox -> Recorte'."
    )

    def run(self, originais, gerado, bbox, suavizar=12, mascara=None, expandir=0):
        if not _OK:
            raise RuntimeError("[Bruxos Stitch] torch indisponivel.")
        if not isinstance(bbox, dict) or "x" not in bbox:
            raise ValueError("[Bruxos Stitch] 'bbox' invalido -- ligue a saida do node de recorte.")
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        T, H, W, C = (int(v) for v in originais.shape)
        avisos = []

        g = gerado
        if int(g.shape[1]) != h or int(g.shape[2]) != w:
            avisos.append("o gerado veio %dx%d e a caixa e %dx%d -- reescalei. Se a proporcao "
                          "diferir, ha distorcao: gere no tamanho que o BBox informou."
                          % (int(g.shape[2]), int(g.shape[1]), w, h))
            g = F.interpolate(g.permute(0, 3, 1, 2).float(), size=(h, w),
                              mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        Tg = int(g.shape[0])
        if Tg != T:
            if Tg == 1:
                g = g.repeat(T, 1, 1, 1)
            else:
                avisos.append("o gerado tem %d frames e o original %d -- reamostrei no tempo."
                              % (Tg, T))
                idx = torch.linspace(0, Tg - 1, T).round().long().clamp(0, Tg - 1)
                g = g[idx]

        # ---- alfa: feather a partir da borda da caixa ---------------------
        f = int(suavizar)
        if f > 0:
            yy = torch.arange(h, dtype=torch.float32).view(-1, 1)
            xx = torch.arange(w, dtype=torch.float32).view(1, -1)
            dy = torch.minimum(yy, (h - 1) - yy)
            dx = torch.minimum(xx, (w - 1) - xx)
            a = (torch.minimum(dx, dy) / float(f)).clamp(0, 1)
            a = 0.5 - 0.5 * torch.cos(a * 3.141592653589793)   # cosseno, borda suave
        else:
            a = torch.ones((h, w), dtype=torch.float32)
        a = a.to(originais.device).unsqueeze(0)                # [1,h,w]

        # ---- se veio mascara, a colagem obedece ela ----------------------
        if mascara is not None:
            mm = _norm_mask(mascara, T, H, W)[:, y:y + h, x:x + w]
            e = int(expandir)
            if e != 0:
                k = abs(e) * 2 + 1
                mm = mm.unsqueeze(1)
                mm = (F.max_pool2d(mm, k, 1, abs(e)) if e > 0
                      else -F.max_pool2d(-mm, k, 1, abs(e))).squeeze(1)
            if f > 0:
                b = f
                co = torch.arange(b * 2 + 1, dtype=torch.float32, device=mm.device) - b
                g1 = torch.exp(-(co ** 2) / (2 * (b * 0.5 + 1e-6) ** 2))
                g1 = g1 / g1.sum()
                mm = mm.unsqueeze(1)
                mm = F.conv2d(mm, g1.view(1, 1, 1, -1), padding=(0, b))
                mm = F.conv2d(mm, g1.view(1, 1, -1, 1), padding=(b, 0)).squeeze(1)
            a = a * mm.clamp(0, 1)

        out = originais.clone()
        alvo = out[:, y:y + h, x:x + w, :].float()
        out[:, y:y + h, x:x + w, :] = (
            alvo * (1.0 - a.unsqueeze(-1)) + g.float().to(alvo.device) * a.unsqueeze(-1)
        ).to(out.dtype)

        info = ("colado %dx%d em (%d,%d) sobre %dx%d | feather %d px | %s"
                % (w, h, x, y, W, H, f,
                   "limitado pela mascara" if mascara is not None else "caixa inteira"))
        print("[Bruxos Stitch] %s" % info, flush=True)
        for m_ in avisos:
            print("[Bruxos Stitch]   AVISO: %s" % m_, flush=True)
        if avisos:
            info += "\n" + "\n".join("- %s" % m_ for m_ in avisos)
        return (out.clamp(0, 1), info)


NODE_CLASS_MAPPINGS = {
    "BruxosMaskBBox": BruxosMaskBBox,
    "BruxosMaskStitch": BruxosMaskStitch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosMaskBBox": "Máscara → BBox → Recorte (Bruxos)",
    "BruxosMaskStitch": "Stitch pelo BBox (Bruxos)",
}
