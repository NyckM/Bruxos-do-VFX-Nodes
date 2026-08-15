# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Video Tiler: ladrilho em PIXEL pra upscale (LTX / MiniMax H3)
=============================================================================
POR QUE ESTE PACOTE EXISTE (e por que o `ltx_tiled_bruxos.py` nao servia)
    O `BruxosLTXTiledGuider` corta o LATENTE e funde as predicoes de ruido a
    cada passo. Isso e otimo pra GERAR coerente -- os ladrilhos se enxergam
    durante o processo -- mas ele nao tem onde enfiar um condicionamento
    DIFERENTE por ladrilho.

    Upscale E condicionamento por ladrilho: cada pedaco precisa ser guiado
    pelo seu proprio recorte da fonte. Sao arquiteturas diferentes:

        step-fused (latente)      | ladrilho em pixel (aqui)
        --------------------------|---------------------------------
        corta o latente           | corta a IMAGEM
        funde a cada passo        | funde no fim
        condicionamento global    | UM RAMO COMPLETO POR LADRILHO
        serve pra gerar           | serve pra UPSCALAR

CREDITO
    A geometria de posicionamento par (equiparticao inteira das passadas) e a
    rampa de costura por "distancia a borda interna mais proxima" vieram do
    comfyui-video-tiler de maDcaDDie2000 (Apache-2.0). Reescrevi em vez de
    copiar, mas a ideia central e dele e merece o credito.

O QUE MUDA AQUI
    * Um TILE_CONFIG proprio ("BXT1"), sem versoes legadas pra carregar.
    * Helper de GRADE POR MODELO: LTX comprime 32x no espaco, o H3 comprime
      16x + patch 2x2. Ladrilho fora da grade da costura ou erro de shape, e
      nenhum slicer generico sabe qual modelo voce esta usando.
    * Avisos altos onde da pra errar em silencio: ladrilho maior que o quadro,
      sobreposicao zero, contagem de ladrilhos que nao bate na fusao.

FLUXO EM MEMORIA
    Load Video ─> Tile Slice ─┬─ tiles (LISTA) ─> [seu ramo de upscale] ─┐
                              └─ tile_config ───────────────────────────┼─> Tile Merge
    (referencia) ─> Tile Ref Slice (mesma geometria) ─> ramo             ┘

FLUXO EM DISCO (quando nem a lista de ladrilhos cabe)
    passe 1: Disk Job ─> Disk Get Tile ─> [upscale] ─> Disk Save Tile
             (uma execucao por indice)
    passe 2: Disk Merge (le os .pt um por um e funde)
"""

import json
import logging
import math
import os

try:
    import torch
    import torch.nn.functional as F
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Tiler"
MARCA = "BXT1"

# Compressao espacial efetiva de cada modelo. Ladrilho tem que ser multiplo
# disso, senao o encoder arredonda e a costura desalinha.
GRADE = {
    "LTX 2.x  (32)": 32,
    "MiniMax H3  (32)": 32,
    "Bernini / Wan 2.2-14B  (16)": 16,
    "Wan 2.1 / 2.2-14B  (16)": 16,
    "Wan 2.2-5B  (32)": 32,
    "SDXL / generico  (8)": 8,
}


# ---------------------------------------------------------------------------
# geometria
# ---------------------------------------------------------------------------
def _snap(v, m):
    return max(m, (int(v) // m) * m)


def _posicoes(tamanho, ladrilho, sobrep_min, m):
    """Inicios de ladrilho no eixo, com o ULTIMO encostando na borda.

    Usa equiparticao INTEIRA do resto entre as passadas:
        passo_i = (resto*(i+1))//n - (resto*i)//n
    Assim a soma fecha exata e passadas vizinhas diferem no maximo 1 pixel --
    sem aquele ladrilho final espremido que aparece quando voce vai somando
    passo fixo e joga o resto todo na ultima junta.
    """
    ladrilho = max(1, int(ladrilho))
    tamanho = max(1, int(tamanho))
    m = max(1, min(64, int(m)))
    if ladrilho >= tamanho:
        return [0], 0
    resto = tamanho - ladrilho
    sobrep_min = max(0, min(int(sobrep_min), ladrilho // 2))
    passo_max = max(m, ladrilho - sobrep_min)

    n_min = max(1, math.ceil(resto / passo_max))
    for n in range(n_min, n_min + max(8, resto // max(m, 1) + 4) + 1):
        passos = [(resto * (i + 1)) // n - (resto * i) // n for i in range(n)]
        if not passos or max(passos) > passo_max or min(passos) < 1:
            continue
        xs = [0]
        for p in passos:
            xs.append(xs[-1] + p)
        if xs[-1] != resto:
            continue
        return xs, ladrilho - max(passos)

    # fallback: passo fixo (pode deixar a ultima junta desigual)
    passo = max(1, ladrilho - sobrep_min)
    xs, i = [0], 0
    while xs[-1] + ladrilho < tamanho:
        i += passo
        xs.append(min(i, tamanho - ladrilho))
        if xs[-1] == xs[-2]:
            break
    return sorted(set(xs)), max(0, ladrilho - passo)


def _ordem(padrao, nr, nc):
    if padrao == "coluna":
        return [(r, c) for c in range(nc) for r in range(nr)]
    if padrao == "espiral":
        saida, cima, baixo, esq, dir_ = [], 0, nr - 1, 0, nc - 1
        while cima <= baixo and esq <= dir_:
            for c in range(esq, dir_ + 1):
                saida.append((cima, c))
            cima += 1
            for r in range(cima, baixo + 1):
                saida.append((r, dir_))
            dir_ -= 1
            if cima <= baixo:
                for c in range(dir_, esq - 1, -1):
                    saida.append((baixo, c))
                baixo -= 1
            if esq <= dir_:
                for r in range(baixo, cima - 1, -1):
                    saida.append((r, esq))
                esq += 1
        return saida
    return [(r, c) for r in range(nr) for c in range(nc)]


def montar_layout(W, H, lw, lh, mult, sobrep_frac, padrao):
    """-> (lista de dicts, config, sobrep_x, sobrep_y)"""
    m = max(1, min(64, int(mult)))
    lw = _snap(max(m, lw), m)
    lh = _snap(max(m, lh), m)
    sx = min(lw // 2, _snap(int(lw * sobrep_frac), m)) if sobrep_frac > 0 else 0
    sy = min(lh // 2, _snap(int(lh * sobrep_frac), m)) if sobrep_frac > 0 else 0
    xs, ox = _posicoes(W, lw, sx, m)
    ys, oy = _posicoes(H, lh, sy, m)
    nc, nr = len(xs), len(ys)

    tiles = []
    for i, (r, c) in enumerate(_ordem(padrao, nr, nc)):
        x, y = xs[c], ys[r]
        w, h = min(lw, W - x), min(lh, H - y)
        if w > 0 and h > 0:
            tiles.append({"x": x, "y": y, "w": w, "h": h, "col": c, "row": r, "ordem": i})
    cfg = (MARCA, int(W), int(H), int(lw), int(lh), int(ox), int(oy), m,
           tuple((t["x"], t["y"], t["w"], t["h"], t["col"], t["row"], t["ordem"]) for t in tiles))
    return tiles, cfg, ox, oy


def ler_config(cfg):
    while isinstance(cfg, (list, tuple)) and len(cfg) == 1 and cfg[0] != MARCA:
        cfg = cfg[0]
    if not (isinstance(cfg, (list, tuple)) and len(cfg) == 9 and cfg[0] == MARCA):
        raise ValueError(
            "[Bruxos Tiler] 'tile_config' invalido. Ele tem que vir do 'Tile Slice (Bruxos)' -- "
            "nao e compativel com o TILE_CONFIG de outros pacotes de ladrilho."
        )
    _, W, H, lw, lh, ox, oy, m, td = cfg
    tiles = [{"x": t[0], "y": t[1], "w": t[2], "h": t[3],
              "col": t[4], "row": t[5], "ordem": t[6]} for t in td]
    return int(W), int(H), int(lw), int(lh), int(ox), int(oy), int(m), tiles


# ---------------------------------------------------------------------------
# fusao
# ---------------------------------------------------------------------------
def _rampa(w, h, x, y, W, H, fx, fy, dev):
    """Alfa do topo: distancia a borda INTERNA mais proxima, normalizada por
    eixo, o MINIMO das duas, e uma subida cosseno.

    O minimo (em vez do produto) e o detalhe que importa: multiplicar as
    rampas dos dois eixos cria um poco de peso nos CANTOS, onde as duas caem
    juntas -- e canto escuro na junta e o artefato classico de ladrilho."""
    if fx <= 0 and fy <= 0:
        return torch.ones((h, w), dtype=torch.float32, device=dev)
    ly = torch.arange(h, device=dev, dtype=torch.float32).view(-1, 1).expand(h, w)
    lx = torch.arange(w, device=dev, dtype=torch.float32).view(1, -1).expand(h, w)
    ns = []
    if x > 0 and fx > 0:
        ns.append(lx / fx)
    if y > 0 and fy > 0:
        ns.append(ly / fy)
    if x + w < W and fx > 0:
        ns.append(((w - 1) - lx) / fx)
    if y + h < H and fy > 0:
        ns.append(((h - 1) - ly) / fy)
    if not ns:
        return torch.ones((h, w), dtype=torch.float32, device=dev)
    t = torch.min(torch.stack(ns, 0), 0).values.clamp(0, 1)
    return 0.5 * (1.0 - torch.cos(math.pi * t))


def _curva(a, modo):
    if modo == "suave_entrada":
        return a * a
    if modo == "suave_saida":
        return 1.0 - (1.0 - a) * (1.0 - a)
    if modo == "suave_ambos":
        return a * a * (3.0 - 2.0 * a)
    return a


def _norm_tile(t):
    while isinstance(t, (list, tuple)) and len(t) > 0:
        t = t[0]
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t


def fundir(tiles, cfg, feather, curva="linear", modo="media_ponderada", dispositivo="auto"):
    W, H, lw, lh, ox, oy, m, specs = ler_config(cfg)
    tl = [_norm_tile(t) for t in tiles]
    if not tl:
        raise ValueError("[Bruxos Tiler] lista de ladrilhos VAZIA. O ramo de upscale nao devolveu nada.")
    if len(tl) != len(specs):
        raise ValueError(
            f"[Bruxos Tiler] chegaram {len(tl)} ladrilhos mas o tile_config descreve {len(specs)}.\n"
            f"Isso quase sempre e o ramo de upscale mudando o numero de itens da lista (algum node "
            f"que junta batch, ou um Preview no meio). A ordem TEM que ser a mesma do slicer."
        )
    p = tl[0]
    B, C = int(p.shape[0]), int(p.shape[3])
    dev = (torch.device("cpu") if dispositivo == "cpu" else
           (torch.device("cuda:0") if (dispositivo == "cuda" and torch.cuda.is_available()) else p.device))

    f = max(0.0, min(0.5, float(feather)))
    fx = min(ox, _snap(int(lw * f), m)) if (ox > 0 and f > 0) else 0
    fy = min(oy, _snap(int(lh * f), m)) if (oy > 0 and f > 0) else 0

    ordenados = sorted(enumerate(specs), key=lambda kv: kv[1]["ordem"])

    if modo == "media_ponderada":
        num = torch.zeros((B, H, W, C), dtype=torch.float32, device=dev)
        den = torch.zeros((B, H, W), dtype=torch.float32, device=dev)
        for k, (i, s) in enumerate(ordenados):
            t = tl[i].to(dev)
            x, y, w, h = s["x"], s["y"], s["w"], s["h"]
            a = (torch.ones((h, w), dtype=torch.float32, device=dev) if k == 0
                 else _curva(_rampa(w, h, x, y, W, H, fx, fy, dev), curva))
            num[:, y:y + h, x:x + w, :] += t.to(torch.float32) * a.unsqueeze(0).unsqueeze(-1)
            den[:, y:y + h, x:x + w] += a.unsqueeze(0)
        return (num / den.unsqueeze(-1).clamp(min=1e-8)).to(p.dtype)

    saida = torch.zeros((B, H, W, C), dtype=p.dtype, device=dev)
    coberto = torch.zeros((B, H, W), dtype=torch.bool, device=dev)
    for k, (i, s) in enumerate(ordenados):
        t = tl[i].to(dev)
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        if k == 0:
            saida[:, y:y + h, x:x + w, :] = t
            coberto[:, y:y + h, x:x + w] = True
            continue
        a = _curva(_rampa(w, h, x, y, W, H, fx, fy, dev), curva)
        cov = coberto[:, y:y + h, x:x + w]
        # onde ninguem escreveu ainda, cola opaco: rampa so vale sobre pixel ja pintado
        a = torch.where(cov, a.unsqueeze(0).expand(B, h, w),
                        torch.ones((), dtype=torch.float32, device=dev)).unsqueeze(-1)
        reg = saida[:, y:y + h, x:x + w, :].to(torch.float32)
        saida[:, y:y + h, x:x + w, :] = (reg * (1 - a) + t.to(torch.float32) * a).to(p.dtype)
        coberto[:, y:y + h, x:x + w] = True
    return saida


# ===========================================================================
# 1) GRADE — o helper que os pacotes genericos nao tem
# ===========================================================================
class BruxosTileGrade:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "modelo": (list(GRADE.keys()), {"default": "LTX 2.x  (32)", "tooltip":
                    "Qual modelo vai processar os ladrilhos. Define o MULTIPLO da grade.\n"
                    "LTX 2.x comprime 32x no espaco. O MiniMax H3 comprime 16x no VAE e ainda "
                    "faz patch 2x2 no transformer = 32x efetivo. Wan 2.1/2.2-14B usa VAE 8x + "
                    "patch 2x2 = 16x. Wan 2.2-5B usa VAE 16x + patch 2x2 = 32x.\n"
                    "Ladrilho fora da grade e arredondado pelo encoder, e a costura desalinha "
                    "por alguns pixels -- aparece como linha fantasma nas juntas."}),
                "largura_alvo": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "tooltip":
                    "Tamanho de ladrilho que voce QUER. Vai ser arredondado pra baixo ate a grade."}),
                "altura_alvo": ("INT", {"default": 576, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "imagem": ("IMAGE", {"tooltip":
                    "[opcional] Se ligada, o node mede o quadro e ja calcula quantos ladrilhos vao "
                    "sair, avisando se o ladrilho e maior que a imagem (nesse caso vira 1 ladrilho "
                    "so e o tiling nao serve pra nada)."}),
                "sobreposicao": (["1/8", "1/4", "3/8", "1/2"], {"default": "1/4", "tooltip":
                    "Sobreposicao MINIMA entre vizinhos, como fracao do ladrilho. Mais sobreposicao "
                    "= costura mais macia e mais ladrilhos (mais tempo)."}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("largura", "altura", "multiplo", "sobreposicao", "info")
    OUTPUT_TOOLTIPS = ("-> 'ladrilho_w' do Tile Slice.", "-> 'ladrilho_h'.",
                       "-> 'multiplo'.", "-> 'sobreposicao'.", "O que foi ajustado e por que.")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Tile Grade (Bruxos): calcula o tamanho de ladrilho VALIDO pro modelo que vai processar. "
        "Cada modelo tem um multiplo de compressao (LTX 32x, H3 32x efetivo, Wan-14B 16x) e ladrilho "
        "fora da grade vira costura fantasma. Ligue a imagem pra ele conferir a contagem tambem."
    )

    def run(self, modelo, largura_alvo, altura_alvo, imagem=None, sobreposicao="1/4"):
        m = GRADE[modelo]
        lw, lh = _snap(largura_alvo, m), _snap(altura_alvo, m)
        frac = {"1/8": .125, "1/4": .25, "3/8": .375, "1/2": .5}[sobreposicao]
        avisos = []
        if lw != largura_alvo or lh != altura_alvo:
            avisos.append(f"ajustado {largura_alvo}x{altura_alvo} -> {lw}x{lh} (multiplo de {m})")
        extra = ""
        if imagem is not None:
            H, W = int(imagem.shape[1]), int(imagem.shape[2])
            if lw >= W and lh >= H:
                avisos.append(f"o ladrilho {lw}x{lh} e MAIOR que o quadro {W}x{H} -- vai sair 1 "
                              f"ladrilho so, e o tiling nao economiza nada. Diminua o ladrilho.")
            _, cfg, ox, oy = montar_layout(W, H, lw, lh, m, frac, "linha")
            n = len(cfg[8])
            extra = f" | quadro {W}x{H} -> {n} ladrilho(s), sobreposicao real {ox}x{oy}px"
            if W % m or H % m:
                # NAO e cosmetico: o ladrilho de borda herda a sobra (ex.: 808 numa grade
                # de 32), o encoder arredonda pra baixo (800) e o merge tenta escrever 800
                # numa fatia de 808 -> RuntimeError la na frente, depois de todo o sampling.
                avisos.append(
                    f"ISSO VAI QUEBRAR O MERGE. O quadro ({W}x{H}) nao e multiplo de {m}: "
                    f"{'largura ' + str(W) + ' -> ' + str(W - W % m) if W % m else ''}"
                    f"{' e ' if (W % m and H % m) else ''}"
                    f"{'altura ' + str(H) + ' -> ' + str(H - H % m) if H % m else ''}. "
                    f"O ladrilho de borda fica com a sobra, o encoder arredonda pra baixo, "
                    f"e o merge estoura com 'tensor (X) must match existing size (Y)' -- so "
                    f"que DEPOIS de todo o sampling. Ponha um resize com divisible_by={m} "
                    f"antes do slicer (o ImageResizeKJv2 com width/height em 0 mantem o "
                    f"tamanho e so corta a sobra).")
        info = f"{modelo} | ladrilho {lw}x{lh} | multiplo {m} | sobrep {sobreposicao}{extra}"
        print(f"[Bruxos Tile Grade] {info}", flush=True)
        for a in avisos:
            print(f"[Bruxos Tile Grade]   AVISO: {a}", flush=True)
        if avisos:
            info += "\n" + "\n".join(f"- {a}" for a in avisos)
        return (lw, lh, m, frac, info)


# ===========================================================================
# 2) SLICE
# ===========================================================================
class BruxosTileSlice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "imagens": ("IMAGE", {"tooltip": "O batch de frames [B,H,W,C] a ser ladrilhado."}),
                "ladrilho_w": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "ladrilho_h": ("INT", {"default": 576, "min": 64, "max": 8192, "step": 8}),
                "multiplo": ("INT", {"default": 32, "min": 1, "max": 64, "step": 1, "tooltip":
                    "Grade de alinhamento. Use o 'Tile Grade (Bruxos)' pra acertar por modelo."}),
                "sobreposicao": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip": "Sobreposicao MINIMA entre vizinhos, fracao do ladrilho."}),
            },
            "optional": {
                "padrao": (["linha", "coluna", "espiral"], {"default": "linha", "tooltip":
                    "Ordem de percurso. Importa na fusao 'sobrepor': o ultimo ladrilho fica por cima. "
                    "'espiral' deixa as juntas do centro por baixo, o que ajuda quando o sujeito esta "
                    "no meio. Na 'media_ponderada' a ordem quase nao importa."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TILE_CONFIG", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("ladrilhos", "tile_config", "mapa", "quantidade", "info")
    OUTPUT_IS_LIST = (True, False, False, False, False)
    OUTPUT_TOOLTIPS = (
        "LISTA de ladrilhos -> mande pro seu ramo de upscale. O ComfyUI roda o ramo uma vez por item.",
        "A geometria -> ligue no Tile Merge E no Tile Ref Slice.",
        "Mapa visual das posicoes e da ordem, pra conferir antes de gastar meia hora.",
        "Quantos ladrilhos sairam.", "Resumo.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Tile Slice (Bruxos): corta o video em ladrilhos com sobreposicao, em PIXEL. As posicoes usam "
        "equiparticao inteira -- as passadas entre vizinhos diferem no maximo 1px e o ultimo ladrilho "
        "encosta na borda, sem aquele pedaco espremido no fim. Saida em LISTA: cada ladrilho passa "
        "pelo seu ramo de upscale inteiro, e e isso que permite condicionamento por ladrilho."
    )

    def run(self, imagens, ladrilho_w, ladrilho_h, multiplo, sobreposicao, padrao="linha"):
        if not _OK:
            raise RuntimeError("[Bruxos Tiler] torch indisponivel.")
        if imagens.ndim != 4:
            raise ValueError(f"[Bruxos Tiler] 'imagens' precisa ser [B,H,W,C]; veio {tuple(imagens.shape)}.")
        B, H, W, C = (int(v) for v in imagens.shape)
        tiles, cfg, ox, oy = montar_layout(W, H, ladrilho_w, ladrilho_h, multiplo, sobreposicao, padrao)

        recortes = [imagens[:, t["y"]:t["y"] + t["h"], t["x"]:t["x"] + t["w"], :] for t in tiles]

        # mapa: retangulos coloridos por ordem (frio -> quente)
        mapa = torch.full((1, H, W, 3), 0.12, dtype=torch.float32)
        n = max(1, len(tiles))
        for t in tiles:
            u = t["ordem"] / max(1, n - 1)
            cor = torch.tensor([0.15 + 0.8 * u, 0.45, 0.95 - 0.7 * u], dtype=torch.float32)
            x1, y1, x2, y2 = t["x"], t["y"], t["x"] + t["w"], t["y"] + t["h"]
            e = max(2, min(H, W) // 300)
            for a1, a2, b1, b2 in ((y1, y1 + e, x1, x2), (y2 - e, y2, x1, x2),
                                   (y1, y2, x1, x1 + e), (y1, y2, x2 - e, x2)):
                mapa[0, a1:a2, b1:b2, :] = cor

        avisos = []
        if len(tiles) == 1:
            avisos.append("saiu 1 ladrilho so -- o ladrilho cobre o quadro inteiro e o tiling nao "
                          "esta economizando nada. Diminua 'ladrilho_w/h'.")
        if ox == 0 and oy == 0 and len(tiles) > 1:
            avisos.append("sobreposicao ZERO entre vizinhos: a fusao nao tem onde fazer rampa e a "
                          "junta vai ser um corte seco. Suba 'sobreposicao'.")
        if W % int(multiplo) or H % int(multiplo):
            avisos.append(f"o quadro {W}x{H} nao e multiplo de {multiplo}; o ultimo ladrilho de cada "
                          f"eixo sai menor e pode desalinhar no encoder.")

        info = (f"{B} frames {W}x{H} -> {len(tiles)} ladrilho(s) de ate "
                f"{cfg[3]}x{cfg[4]} | sobreposicao real {ox}x{oy}px | multiplo {multiplo} | {padrao}")
        print(f"[Bruxos Tile Slice] {info}", flush=True)
        for a in avisos:
            print(f"[Bruxos Tile Slice]   AVISO: {a}", flush=True)
        if avisos:
            info += "\n" + "\n".join(f"- {a}" for a in avisos)
        return (recortes, cfg, mapa, len(tiles), info)


# ===========================================================================
# 3) REF SLICE — o node que faltava no LTX tiled
# ===========================================================================
class BruxosTileRefSlice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "referencia": ("IMAGE", {"tooltip":
                    "A imagem/video de REFERENCIA, no MESMO tamanho do que foi ladrilhado. "
                    "Se o tamanho diferir, ela e redimensionada antes de cortar."}),
                "tile_config": ("TILE_CONFIG", {"forceInput": True, "tooltip":
                    "O MESMO tile_config do slicer. E o que garante que o recorte da referencia "
                    "cai exatamente sobre o recorte do video."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("ladrilhos_ref", "quantidade")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_TOOLTIPS = ("LISTA de ladrilhos da referencia, na MESMA ordem -> condicionamento do "
                       "seu ramo (guia de imagem do LTX, ref do H3, etc).", "Quantos.")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Tile Ref Slice (Bruxos): corta uma referencia com a MESMA geometria do slicer. "
        "E a peca que faltava pra upscale ladrilhado: cada ladrilho precisa ser condicionado no seu "
        "proprio recorte da fonte, e nao no quadro inteiro. Sem isso o ladrilho gera em vez de upscalar."
    )

    def run(self, referencia, tile_config):
        W, H, _lw, _lh, _ox, _oy, _m, specs = ler_config(tile_config)
        r = referencia
        src_h, src_w = int(r.shape[1]), int(r.shape[2])
        saida = []
        for t in specs:
            # Recorta primeiro na resolucao original usando coordenadas normalizadas.
            # So depois amplia o recorte para o tamanho final do tile. Isso evita criar
            # temporariamente um video FULL FRAME em alta resolucao (muito pesado para
            # guias clay de dezenas de frames).
            sx0 = max(0, min(src_w - 1, int(round(t["x"] * src_w / W))))
            sy0 = max(0, min(src_h - 1, int(round(t["y"] * src_h / H))))
            sx1 = max(sx0 + 1, min(src_w, int(round((t["x"] + t["w"]) * src_w / W))))
            sy1 = max(sy0 + 1, min(src_h, int(round((t["y"] + t["h"]) * src_h / H))))
            recorte = r[:, sy0:sy1, sx0:sx1, :]
            if int(recorte.shape[1]) != t["h"] or int(recorte.shape[2]) != t["w"]:
                recorte = F.interpolate(
                    recorte.permute(0, 3, 1, 2).float(),
                    size=(t["h"], t["w"]), mode="bicubic", align_corners=False,
                ).permute(0, 2, 3, 1).clamp(0, 1).to(referencia.dtype)
            saida.append(recorte)
        if src_w != W or src_h != H:
            print(f"[Bruxos Tile Ref] referencia leve {src_w}x{src_h} -> "
                  f"crop local + resize por tile no canvas {W}x{H}.", flush=True)
        print(f"[Bruxos Tile Ref] {len(saida)} ladrilho(s) de referencia cortados.", flush=True)
        return (saida, len(saida))


# ===========================================================================
# 4) MERGE
# ===========================================================================
class BruxosTileMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_config": ("TILE_CONFIG", {"forceInput": True}),
                "ladrilhos": ("IMAGE", {"tooltip": "A lista processada, na ordem do slicer."}),
                "feather": ("FLOAT", {"default": 0.125, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip":
                    "Largura da rampa de costura, como fracao do ladrilho (max 50%). "
                    "A rampa e limitada pela sobreposicao real: nao adianta pedir 0.4 se a "
                    "sobreposicao e 1/8. 0.10-0.20 costuma resolver."}),
            },
            "optional": {
                "curva": (["linear", "suave_entrada", "suave_saida", "suave_ambos"],
                    {"default": "linear", "tooltip":
                    "Remapeia a rampa. 'suave_ambos' (smoothstep) deixa a transicao mais lenta no "
                    "meio e mais rapida nas pontas -- ajuda quando a junta ainda aparece como faixa."}),
                "modo": (["media_ponderada", "sobrepor"], {"default": "media_ponderada", "tooltip":
                    "media_ponderada = soma(peso x cor)/soma(peso). Simetrico, nao depende da ordem, "
                    "e o mais seguro pra upscale.\n"
                    "sobrepor = ordem de pintor com portao de cobertura. Preserva melhor detalhe do "
                    "ultimo ladrilho, mas a junta pode ficar assimetrica."}),
                "dispositivo": (["auto", "cpu", "cuda"], {"default": "cpu", "tooltip":
                    "Onde montar o quadro final. 'cpu' e o padrao aqui de proposito: o buffer de "
                    "saida inteiro (B x H x W x C) e grande, e em video longo ele sozinho estoura a "
                    "VRAM depois de todo o trabalho ja feito. 'cuda' e mais rapido se couber."}),
            },
        }

    INPUT_IS_LIST = (False, True)
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("imagem",)
    OUTPUT_TOOLTIPS = ("O quadro inteiro remontado [B,H,W,C].",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Tile Merge (Bruxos): remonta o quadro a partir dos ladrilhos processados. A rampa de costura "
        "usa a distancia a borda interna mais proxima e um cosseno -- e o MINIMO entre os eixos, nao o "
        "produto, o que evita o poco de peso nos cantos que vira mancha escura na junta."
    )

    def run(self, tile_config, ladrilhos, feather, curva=None, modo=None, dispositivo=None):
        def _u(v, d):
            while isinstance(v, (list, tuple)):
                v = v[0] if v else d
            return v if v is not None else d
        cfg = tile_config[0] if (isinstance(tile_config, list) and tile_config and
                                 tile_config[0] != MARCA) else tile_config
        lst = list(ladrilhos) if isinstance(ladrilhos, (list, tuple)) else [ladrilhos]
        f, cv = _u(feather, 0.125), _u(curva, "linear")
        md, dv = _u(modo, "media_ponderada"), _u(dispositivo, "cpu")
        print(f"[Bruxos Tile Merge] {len(lst)} ladrilho(s) | feather {f} | {cv} | {md} | {dv}", flush=True)
        out = fundir(lst, cfg, f, cv, md, dv)
        print(f"[Bruxos Tile Merge] pronto: {tuple(out.shape)}", flush=True)
        return (out,)


# ===========================================================================
# 5) COLOR MATCH
# ===========================================================================
class BruxosTileColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fundido": ("IMAGE", {"tooltip": "A saida do Tile Merge."}),
                "referencia": ("IMAGE", {"tooltip":
                    "O MESMO conteudo antes do upscale (baixa resolucao serve). Redimensionado "
                    "internamente. Batch 1 se propaga pro batch inteiro."}),
            },
            "optional": {
                "sigma": ("FLOAT", {"default": 14.0, "min": 0.0, "max": 256.0, "step": 0.5, "tooltip":
                    "Raio (px) do desfoque que separa baixa de alta frequencia. ALTO = so a cor "
                    "ampla e corrigida e o detalhe do upscale fica intacto. BAIXO demais e voce "
                    "importa a textura borrada da referencia junto."}),
                "puxada": ("FLOAT", {"default": 0.58, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                    "Quanto a cor de baixa frequencia segue a referencia. 1.0 = cor da referencia."}),
                "detalhe": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                    "Ganho da alta frequencia do fundido. 1.0 = mantem todo o detalhe do upscale."}),
                "travar_luminancia": ("BOOLEAN", {"default": True, "tooltip":
                    "Reescala o RGB depois da puxada pra luminancia Rec.709 continuar a do fundido. "
                    "Sem isso, uma referencia mais escura ACHATA a cena inteira junto com a cor."}),
                "limite_luma": ("FLOAT", {"default": 4.0, "min": 1.05, "max": 10.0, "step": 0.05,
                    "tooltip": "Teto do multiplicador ao travar luminancia, pra nao explodir pixel escuro."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("imagem",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Tile Color Match (Bruxos): alinha a COR do resultado a uma referencia sem perder o detalhe "
        "gerado. Separa baixa e alta frequencia por gaussiana, puxa so a baixa pra referencia e "
        "devolve a alta do upscale. Serve pra costura com tom diferente entre ladrilhos e tambem pro "
        "desvio de cor de VAE (o do Wan 2.2 de 48 canais, por exemplo)."
    )

    def _blur(self, x, s):
        s = float(max(s, 1e-3))
        r = max(1, int(math.ceil(3 * s)))
        c = x.shape[1]
        modo = "reflect" if (r < x.shape[2] and r < x.shape[3]) else "replicate"
        xs = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        k = torch.exp(-0.5 * (xs / s) ** 2)
        k = k / k.sum()
        n = k.numel()
        x = F.conv2d(F.pad(x, (0, 0, r, r), mode=modo), k.view(1, 1, n, 1).expand(c, 1, n, 1), groups=c)
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=modo), k.view(1, 1, 1, n).expand(c, 1, 1, n), groups=c)
        return x

    def run(self, fundido, referencia, sigma=14.0, puxada=0.58, detalhe=1.0,
            travar_luminancia=True, limite_luma=4.0):
        if fundido.shape[-1] != 3:
            raise ValueError("[Bruxos Color Match] espera RGB (C=3).")
        B, H, W, _ = (int(v) for v in fundido.shape)
        ref = referencia.to(fundido.device, fundido.dtype)
        if int(ref.shape[0]) == 1 and B > 1:
            ref = ref.expand(B, -1, -1, -1)
        elif int(ref.shape[0]) != B:
            raise ValueError(f"[Bruxos Color Match] referencia tem batch {int(ref.shape[0])} e o "
                             f"fundido tem {B}. Tem que bater, ou a referencia ter batch 1.")
        # F.pad/conv2d na CUDA usam indice 32-bit: fatiar evita estourar em video longo
        por_frame = max(1, 3 * H * W)
        passo = max(1, (1 << 30) // por_frame)
        trab = torch.float32 if fundido.dtype in (torch.float16, torch.bfloat16) else fundido.dtype
        partes = []
        for s in range(0, B, passo):
            e = min(s + passo, B)
            m = fundido[s:e].permute(0, 3, 1, 2).contiguous()
            r = ref[s:e].permute(0, 3, 1, 2).contiguous()
            if r.shape[2] != H or r.shape[3] != W:
                r = F.interpolate(r, size=(H, W), mode="bicubic", align_corners=False)
            m32, r32 = m.to(trab), r.to(trab)
            if float(sigma) <= 0.05:
                lm, lr = m32, r32
            else:
                lm, lr = self._blur(m32, sigma), self._blur(r32, sigma)
            out = ((1 - puxada) * lm + puxada * lr) + float(detalhe) * (m32 - lm)
            out = out.to(fundido.dtype).permute(0, 2, 3, 1)
            if travar_luminancia:
                def _y(t):
                    return 0.2126 * t[..., 0] + 0.7152 * t[..., 1] + 0.0722 * t[..., 2]
                k = (_y(fundido[s:e]) / (_y(out) + 1e-6)).clamp(1.0 / limite_luma, limite_luma)
                out = (out * k.unsqueeze(-1)).clamp(0, 1)
            partes.append(out.contiguous())
        r = torch.cat(partes, 0)
        print(f"[Bruxos Color Match] {B} frames | sigma {sigma} | puxada {puxada} | "
              f"detalhe {detalhe} | luma {'travada' if travar_luminancia else 'livre'}", flush=True)
        return (r,)


# ===========================================================================
# 6) DISCO — um ladrilho por execucao
# ===========================================================================
def _pasta_job(nome):
    try:
        import folder_paths
        base = folder_paths.get_temp_directory()
    except Exception:
        base = os.path.join(os.path.expanduser("~"), "bruxos_tiles")
    p = os.path.join(base, "bruxos_tiles", "".join(c for c in str(nome) if c.isalnum() or c in "-_"))
    os.makedirs(p, exist_ok=True)
    return p


class BruxosTileDiskJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "tile_config": ("TILE_CONFIG", {"forceInput": True}),
            "nome_job": ("STRING", {"default": "upscale_01", "tooltip":
                "Nome da pasta deste trabalho. MUDE ao comecar um video novo -- reaproveitar o nome "
                "mistura ladrilhos de execucoes diferentes e a fusao sai remendada."}),
        }}
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("job", "quantidade", "info")
    FUNCTION = "run"
    CATEGORY = CAT + "/Disco"
    DESCRIPTION = ("Tile Disk Job (Bruxos): grava o manifesto da geometria em disco. Os outros nodes "
                   "de disco leem dele. Use quando nem a lista de ladrilhos processados cabe na memoria.")

    def run(self, tile_config, nome_job):
        W, H, lw, lh, ox, oy, m, specs = ler_config(tile_config)
        p = _pasta_job(nome_job)
        man = {"marca": MARCA, "w": W, "h": H, "ladrilho": [lw, lh],
               "sobrep": [ox, oy], "multiplo": m, "n": len(specs), "tiles": specs}
        with open(os.path.join(p, "manifesto.json"), "w", encoding="utf-8") as fp:
            json.dump(man, fp, indent=1)
        salvos = len([f for f in os.listdir(p) if f.startswith("tile_") and f.endswith(".pt")])
        info = f"job '{nome_job}' | {len(specs)} ladrilho(s) | {salvos} ja salvo(s) | {p}"
        print(f"[Bruxos Tile Disk] {info}", flush=True)
        return (p, len(specs), info)


class BruxosTileDiskGet:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "imagens": ("IMAGE",),
            "job": ("STRING", {"forceInput": True}),
            "indice": ("INT", {"default": 0, "min": 0, "max": 9999, "tooltip":
                "Qual ladrilho processar NESTA execucao. Incremente e enfileire de novo."}),
        }}
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("ladrilho", "indice", "info")
    FUNCTION = "run"
    CATEGORY = CAT + "/Disco"
    DESCRIPTION = "Tile Disk Get (Bruxos): recorta UM ladrilho do video original, pelo indice do manifesto."

    def run(self, imagens, job, indice):
        with open(os.path.join(job, "manifesto.json"), encoding="utf-8") as fp:
            man = json.load(fp)
        n = man["n"]
        i = max(0, min(int(indice), n - 1))
        if i != int(indice):
            print(f"[Bruxos Tile Disk] indice {indice} fora de 0..{n-1}; usando {i}.", flush=True)
        t = man["tiles"][i]
        rec = imagens[:, t["y"]:t["y"] + t["h"], t["x"]:t["x"] + t["w"], :]
        info = f"ladrilho {i+1}/{n} em ({t['x']},{t['y']}) {t['w']}x{t['h']}"
        print(f"[Bruxos Tile Disk] {info}", flush=True)
        return (rec, i, info)


class BruxosTileDiskSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ladrilho": ("IMAGE",),
            "job": ("STRING", {"forceInput": True}),
            "indice": ("INT", {"forceInput": True}),
        }, "optional": {
            "sobrescrever": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CAT + "/Disco"
    DESCRIPTION = ("Tile Disk Save (Bruxos): salva o ladrilho processado como .pt (tensor exato, sem "
                   "perda). PNG perderia a precisao e o range.")

    def run(self, ladrilho, job, indice, sobrescrever=True):
        cam = os.path.join(job, f"tile_{int(indice):05d}.pt")
        if os.path.exists(cam) and not sobrescrever:
            info = f"ja existia, mantido: {os.path.basename(cam)}"
        else:
            torch.save(ladrilho.cpu(), cam)
            info = f"salvo {os.path.basename(cam)} {tuple(ladrilho.shape)}"
        with open(os.path.join(job, "manifesto.json"), encoding="utf-8") as fp:
            man = json.load(fp)
        falta = man["n"] - len([f for f in os.listdir(job) if f.startswith("tile_") and f.endswith(".pt")])
        info += f" | faltam {falta}"
        print(f"[Bruxos Tile Disk] {info}", flush=True)
        return (info,)


class BruxosTileDiskMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "job": ("STRING", {"forceInput": True}),
            "feather": ("FLOAT", {"default": 0.125, "min": 0.0, "max": 0.5, "step": 0.005}),
        }, "optional": {
            "curva": (["linear", "suave_entrada", "suave_saida", "suave_ambos"], {"default": "linear"}),
            "modo": (["media_ponderada", "sobrepor"], {"default": "media_ponderada"}),
            "dispositivo": (["cpu", "auto", "cuda"], {"default": "cpu"}),
            "exigir_todos": ("BOOLEAN", {"default": True, "tooltip":
                "PARA com erro claro se faltar ladrilho. Desligue so pra ver um resultado parcial -- "
                "o que faltar vira buraco preto."}),
        }}
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("imagem", "info")
    FUNCTION = "run"
    CATEGORY = CAT + "/Disco"
    DESCRIPTION = ("Tile Disk Merge (Bruxos): le os .pt um por um e funde. So um ladrilho fica na "
                   "memoria por vez; o buffer final ainda existe inteiro, entao prefira dispositivo=cpu.")

    def run(self, job, feather, curva="linear", modo="media_ponderada",
            dispositivo="cpu", exigir_todos=True):
        with open(os.path.join(job, "manifesto.json"), encoding="utf-8") as fp:
            man = json.load(fp)
        n = man["n"]
        arqs = [os.path.join(job, f"tile_{i:05d}.pt") for i in range(n)]
        faltando = [i for i, a in enumerate(arqs) if not os.path.exists(a)]
        if faltando and exigir_todos:
            raise ValueError(
                f"[Bruxos Tile Disk Merge] faltam {len(faltando)} de {n} ladrilhos: "
                f"{faltando[:12]}{'...' if len(faltando) > 12 else ''}.\n"
                f"Rode o passe 1 pros indices que faltam, ou desligue 'exigir_todos' pra ver parcial."
            )
        specs = man["tiles"]
        cfg = (MARCA, man["w"], man["h"], man["ladrilho"][0], man["ladrilho"][1],
               man["sobrep"][0], man["sobrep"][1], man["multiplo"],
               tuple((t["x"], t["y"], t["w"], t["h"], t["col"], t["row"], t["ordem"]) for t in specs))
        tiles = []
        for i, a in enumerate(arqs):
            if os.path.exists(a):
                tiles.append(torch.load(a, map_location="cpu"))
            else:
                t = specs[i]
                tiles.append(torch.zeros((1, t["h"], t["w"], 3), dtype=torch.float32))
        out = fundir(tiles, cfg, feather, curva, modo, dispositivo)
        info = (f"fundidos {n - len(faltando)}/{n} de '{os.path.basename(job)}' -> {tuple(out.shape)}"
                + (f" | {len(faltando)} FALTANDO (buracos pretos)" if faltando else ""))
        print(f"[Bruxos Tile Disk Merge] {info}", flush=True)
        return (out, info)


# ---------------------------------------------------------------------------
# COLISAO DE NOME -- por que este registro NAO e "BruxosTileMerge"
#   O tile_nodes.py ja registrava "BruxosTileMerge" (o merge por TILE_PLAN, com
#   entradas 'tiles'/'plan'/'blend'). O __init__ carrega este modulo DEPOIS e usa
#   dict.update(), entao registrar o mesmo nome aqui SOBRESCREVIA aquele em
#   silencio: os grafos antigos continuavam pedindo 'BruxosTileMerge' e recebiam
#   uma classe com entradas totalmente diferentes -- "input tile_config not
#   found", widgets deslocados, e nenhuma mensagem dizendo o porque.
#   Os dois nodes coexistem de proposito (um funde por TILE_PLAN, o outro por
#   TILE_CONFIG), entao a chave deste e outra. NAO renomeie de volta.
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "BruxosTileGrade": BruxosTileGrade,
    "BruxosTileSlice": BruxosTileSlice,
    "BruxosTileRefSlice": BruxosTileRefSlice,
    "BruxosVideoTileMerge": BruxosTileMerge,
    "BruxosTileColorMatch": BruxosTileColorMatch,
    "BruxosTileDiskJob": BruxosTileDiskJob,
    "BruxosTileDiskGet": BruxosTileDiskGet,
    "BruxosTileDiskSave": BruxosTileDiskSave,
    "BruxosTileDiskMerge": BruxosTileDiskMerge,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosTileGrade": "Tile Grade · por modelo (Bruxos)",
    "BruxosTileSlice": "Tile Slice (Bruxos)",
    "BruxosTileRefSlice": "Tile Ref Slice · guia por ladrilho (Bruxos)",
    "BruxosVideoTileMerge": "Video Tile Merge · costura (Bruxos)",
    "BruxosTileColorMatch": "Tile Color Match (Bruxos)",
    "BruxosTileDiskJob": "Tile Disk · Job (Bruxos)",
    "BruxosTileDiskGet": "Tile Disk · Get (Bruxos)",
    "BruxosTileDiskSave": "Tile Disk · Save (Bruxos)",
    "BruxosTileDiskMerge": "Tile Disk · Merge (Bruxos)",
}
