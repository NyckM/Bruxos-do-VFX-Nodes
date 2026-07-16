# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Tiles (Wan 2.2 / upscale por ladrilho)
======================================================
Substitui o subgraph "Tile Settings" inteiro (Rounding up num, Set dimension
properly, Padding, imageSplitTiles, imageTilesFromBatch, ImageComposite+,
Split Images, Total tiles...) por 3 nodes:

    [Tile Split] -> (por tile) -> seu Wan/upscale -> [Tile Merge]
                 \-> [Tile Select] (dentro do For Loop, pega o tile N)

Como funciona:
  - Voce diz QUANTOS ladrilhos quer: tile_count_width x tile_count_height.
      2 x 2  -> 4 tiles      8 x 8 -> 64 tiles
      1 x 1  -> 1 tile = a imagem inteira (passa direto, sem cortar)
  - O TAMANHO de cada tile e calculado automaticamente (largura/colunas), ja
    alinhado ao 'divisible_by' (o Wan quer multiplo de 16).
  - 'tile_padding' = sobreposicao entre os tiles vizinhos. Na hora de juntar,
    o Merge faz um degrade (feather) nessa faixa -> sem linha de emenda.
  - Funciona em VIDEO (lote de frames): cada tile carrega TODOS os frames.
  - O Merge aceita tiles UPSCALADOS: ele detecta a escala sozinha (ex.: tile
    saiu 2x maior -> a imagem final sai 2x maior).
"""

import logging

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

CAT = "Bruxos do VFX/Tiles"


def _ceil_div(a, d):
    return ((int(a) + d - 1) // d) * d


def _pad_to_div(img, div):
    """Preenche a imagem (replicando a borda) ate W e H virarem multiplos de div.
    Sem isto, uma fonte 1920x1080 (1080 nao e multiplo de 16) perderia 8 linhas."""
    H, W = int(img.shape[1]), int(img.shape[2])
    Hp, Wp = _ceil_div(H, div), _ceil_div(W, div)
    if (Hp, Wp) == (H, W):
        return img, W, H, Wp, Hp
    x = img.permute(0, 3, 1, 2)
    x = torch.nn.functional.pad(x, (0, Wp - W, 0, Hp - H), mode="replicate")
    return x.permute(0, 2, 3, 1).contiguous(), W, H, Wp, Hp


def _plan_tiles(W, H, cols, rows, padding, div):
    """Tiles de tamanho UNIFORME, multiplo de div, cobrindo o canvas inteiro.
    W,H aqui ja sao o canvas PADDED (multiplos de div)."""
    cols, rows = max(1, int(cols)), max(1, int(rows))
    pad = max(0, int(padding))

    # tamanho do tile = nucleo + padding dos dois lados, arredondado PRA CIMA
    tw = min(W, _ceil_div(-(-W // cols) + (2 * pad if cols > 1 else 0), div))
    th = min(H, _ceil_div(-(-H // rows) + (2 * pad if rows > 1 else 0), div))

    tiles = []
    for r in range(rows):
        for c in range(cols):
            cx0 = (W * c) // cols          # nucleo (divisao exata)
            cy0 = (H * r) // rows
            cx1 = (W * (c + 1)) // cols
            cy1 = (H * (r + 1)) // rows
            # centraliza o tile no nucleo e empurra pra dentro do canvas
            x0 = min(max(0, cx0 - (pad if c > 0 else 0)), W - tw)
            y0 = min(max(0, cy0 - (pad if r > 0 else 0)), H - th)
            tiles.append({
                "x0": int(x0), "y0": int(y0),
                "x1": int(x0 + tw), "y1": int(y0 + th),
                "cx0": cx0, "cy0": cy0, "cx1": cx1, "cy1": cy1,
                "col": c, "row": r,
            })
    return tiles, int(tw), int(th)


class BruxosTileSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Imagem ou VIDEO (lote de frames) a ser dividido em ladrilhos."}),
                "tile_count_width": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Quantas COLUNAS de ladrilho. O tamanho de cada um e calculado sozinho. 1 = nao corta na horizontal."}),
                "tile_count_height": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Quantas LINHAS de ladrilho. 2x2 = 4 tiles; 8x8 = 64. 1x1 = a imagem inteira, sem cortar."}),
                "tile_padding": ("INT", {"default": 80, "min": 0, "max": 512, "step": 8,
                    "tooltip": "Sobreposicao (px) entre ladrilhos vizinhos. O Merge usa essa faixa pra fazer o degrade e sumir com a emenda. 0 = corte seco (aparece linha)."}),
            },
            "optional": {
                "divisible_by": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Alinha o tamanho de cada tile a um multiplo disto. O Wan exige 16. Nao mexa sem motivo."}),
                "upscale_by": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Amplia a imagem ANTES de cortar (ex.: 2.0 = corta o dobro do tamanho). Deixe 1.0 se voce ja mandou a imagem no tamanho final."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TILE_PLAN", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("tiles", "plan", "total_tiles", "tile_width", "tile_height", "info")
    OUTPUT_TOOLTIPS = (
        "TODOS os ladrilhos empilhados (tile-major: tile0 c/ todos os frames, depois tile1...). "
        "Se for 1 frame e 1 tile, e a propria imagem.",
        "O 'mapa' dos cortes. Ligue no Tile Select e no Tile Merge.",
        "Quantos ladrilhos ao todo (colunas x linhas). Use no 'total' do For Loop.",
        "Largura de cada ladrilho (calculada).",
        "Altura de cada ladrilho (calculada).",
        "Resumo: tamanho da fonte, grade, tamanho do tile, sobreposicao.",
    )
    FUNCTION = "split"
    CATEGORY = CAT
    DESCRIPTION = ("Corta a imagem/video em ladrilhos pela CONTAGEM (2x2, 8x8...). O tamanho de cada "
                   "ladrilho e calculado automaticamente e alinhado ao multiplo de 16 do Wan. 1x1 passa "
                   "a imagem inteira. Ligue o 'plan' no Tile Merge pra costurar de volta sem emenda.")

    def split(self, image, tile_count_width, tile_count_height, tile_padding,
              divisible_by=16, upscale_by=1.0):
        if not _HAS_TORCH:
            raise RuntimeError("[Bruxos Tiles] torch indisponivel.")
        img = image.float()
        T = int(img.shape[0])

        if float(upscale_by) > 1.0001:
            nh = int(round(img.shape[1] * float(upscale_by)))
            nw = int(round(img.shape[2] * float(upscale_by)))
            img = torch.nn.functional.interpolate(
                img.permute(0, 3, 1, 2), size=(nh, nw), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1).clamp(0, 1)

        cols, rows = max(1, int(tile_count_width)), max(1, int(tile_count_height))
        div = max(1, int(divisible_by))

        # preenche ate multiplo de div (1080 -> 1088), senao perderiamos linhas
        img, W, H, Wp, Hp = _pad_to_div(img, div)

        tiles_meta, tw, th = _plan_tiles(Wp, Hp, cols, rows, int(tile_padding), div)

        crops = [img[:, t["y0"]:t["y1"], t["x0"]:t["x1"], :] for t in tiles_meta]
        out = torch.cat(crops, dim=0) if len(crops) > 1 else crops[0]

        plan = {
            "W": W, "H": H, "Wp": Wp, "Hp": Hp, "T": T,
            "cols": cols, "rows": rows,
            "padding": int(tile_padding), "div": div,
            "tile_w": tw, "tile_h": th,
            "tiles": tiles_meta,
        }
        pad_msg = f" (canvas {Wp}x{Hp} p/ multiplo de {div})" if (Wp, Hp) != (W, H) else ""
        info = (f"fonte {W}x{H} x{T}f{pad_msg} | grade {cols}x{rows} = {len(tiles_meta)} tile(s) | "
                f"tile {tw}x{th} | sobreposicao {int(tile_padding)}px")
        if cols == 1 and rows == 1:
            info += " | 1x1 = imagem inteira (sem corte)"
        print(f"[Bruxos Tile Split] {info}", flush=True)
        return (out.clamp(0, 1), plan, len(tiles_meta), tw, th, info)


class BruxosTileSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE", {"tooltip": "A saida 'tiles' do Tile Split."}),
                "plan": ("TILE_PLAN", {"tooltip": "A saida 'plan' do Tile Split."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Qual ladrilho processar agora (0 = primeiro). Ligue o 'index' do For Loop Start aqui."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("tile", "x", "y", "info")
    OUTPUT_TOOLTIPS = (
        "Este ladrilho, com TODOS os frames do video. Manda pro seu Wan/upscale.",
        "Posicao X do ladrilho na imagem original.",
        "Posicao Y do ladrilho na imagem original.",
        "Qual tile de quantos, e onde ele fica.",
    )
    FUNCTION = "select"
    CATEGORY = CAT
    DESCRIPTION = ("Pega UM ladrilho (com todos os frames) da pilha do Tile Split. Use dentro do For Loop: "
                   "ligue o 'index' do loop aqui e mande a saida pro seu sampler.")

    def select(self, tiles, plan, index=0):
        n = len(plan["tiles"])
        T = int(plan["T"])
        i = max(0, min(n - 1, int(index)))
        tile = tiles[i * T:(i + 1) * T]
        t = plan["tiles"][i]
        info = f"tile {i + 1}/{n} (col {t['col']}, lin {t['row']}) em ({t['x0']},{t['y0']})"
        print(f"[Bruxos Tile Select] {info}", flush=True)
        return (tile, int(t["x0"]), int(t["y0"]), info)


def _feather(th, tw, t, plan, scale, device, dtype):
    """Peso do tile: 1 no meio, degrade nas bordas que encostam em outro tile."""
    wy = torch.ones(th, device=device, dtype=dtype)
    wx = torch.ones(tw, device=device, dtype=dtype)
    f = int(round(plan["padding"] * scale))
    if f > 0:
        ramp_x = min(f, tw // 2)
        ramp_y = min(f, th // 2)
        if ramp_x > 0:
            r = torch.linspace(0, 1, ramp_x + 2, device=device, dtype=dtype)[1:-1]
            if t["col"] > 0:
                wx[:ramp_x] = r
            if t["col"] < plan["cols"] - 1:
                wx[-ramp_x:] = r.flip(0)
        if ramp_y > 0:
            r = torch.linspace(0, 1, ramp_y + 2, device=device, dtype=dtype)[1:-1]
            if t["row"] > 0:
                wy[:ramp_y] = r
            if t["row"] < plan["rows"] - 1:
                wy[-ramp_y:] = r.flip(0)
    return (wy.view(th, 1) * wx.view(1, tw)).clamp(1e-6, 1.0)


class BruxosTileMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE", {"tooltip": "Os ladrilhos JA PROCESSADOS, na mesma ordem do Tile Split (tile-major). Se voce usou For Loop, acumule os tiles e ligue aqui."}),
                "plan": ("TILE_PLAN", {"tooltip": "A saida 'plan' do Tile Split (o mapa dos cortes)."}),
            },
            "optional": {
                "blend": ("BOOLEAN", {"default": True,
                    "tooltip": "LIGADO: degrade (feather) na sobreposicao -> emenda invisivel. DESLIGADO: corte seco (mostra a linha; use so pra depurar)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    OUTPUT_TOOLTIPS = (
        "A imagem/video remontado, com as emendas suavizadas. Se os tiles vieram upscalados, a saida ja sai maior.",
        "Resumo: tamanho final, escala detectada, nº de tiles.",
    )
    FUNCTION = "merge"
    CATEGORY = CAT
    DESCRIPTION = ("Costura os ladrilhos de volta numa imagem/video so, com degrade na sobreposicao "
                   "(sem linha de emenda). Detecta sozinho se os tiles foram UPSCALADOS e devolve a "
                   "imagem final no tamanho ampliado.")

    def merge(self, tiles, plan, blend=True):
        if not _HAS_TORCH:
            raise RuntimeError("[Bruxos Tiles] torch indisponivel.")
        n = len(plan["tiles"])
        T = int(plan["T"])
        t_img = tiles.float()

        got = int(t_img.shape[0])
        if got != n * T:
            # tolera lote incompleto/diferente: usa o que der
            per = max(1, got // max(1, n))
            logging.info(f"[Bruxos Tile Merge] esperava {n * T} imagens ({n} tiles x {T} frames), "
                         f"recebi {got}; usando {per} frame(s) por tile.")
            T = per

        th, tw = int(t_img.shape[1]), int(t_img.shape[2])
        scale = tw / float(plan["tile_w"])          # upscale detectado
        # monta no canvas PADDED e no fim corta de volta ao tamanho real
        Wp = int(plan.get("Wp", plan["W"]))
        Hp = int(plan.get("Hp", plan["H"]))
        ow, oh = int(round(Wp * scale)), int(round(Hp * scale))
        fw, fh = int(round(plan["W"] * scale)), int(round(plan["H"] * scale))

        dev, dt = t_img.device, t_img.dtype
        acc = torch.zeros((T, oh, ow, 3), device=dev, dtype=dt)
        wsum = torch.zeros((T, oh, ow, 1), device=dev, dtype=dt)

        for i in range(min(n, got // max(1, T))):
            t = plan["tiles"][i]
            chunk = t_img[i * T:(i + 1) * T, ..., :3]
            x0 = int(round(t["x0"] * scale))
            y0 = int(round(t["y0"] * scale))
            x1, y1 = min(ow, x0 + tw), min(oh, y0 + th)
            cw, ch = x1 - x0, y1 - y0
            if cw <= 0 or ch <= 0:
                continue
            if blend:
                w = _feather(th, tw, t, plan, scale, dev, dt)[:ch, :cw].view(1, ch, cw, 1)
            else:
                w = torch.ones((1, ch, cw, 1), device=dev, dtype=dt)
            acc[:, y0:y1, x0:x1, :] += chunk[:, :ch, :cw, :] * w
            wsum[:, y0:y1, x0:x1, :] += w

        out = acc / wsum.clamp(min=1e-6)
        out = out[:, :fh, :fw, :]          # tira o padding do canvas
        info = (f"{fw}x{fh} x{T}f | {n} tile(s) de {tw}x{th} | escala {scale:.2f}x | "
                f"blend={'sim' if blend else 'nao'}")
        print(f"[Bruxos Tile Merge] {info}", flush=True)
        return (out.clamp(0, 1), info)


class BruxosTileAccumulate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile": ("IMAGE", {"tooltip": "O ladrilho JA PROCESSADO desta volta do laco (a saida do seu Wan/upscale)."}),
            },
            "optional": {
                "acumulado": ("IMAGE", {"tooltip": "A pilha das voltas anteriores. Na 1a volta pode ficar VAZIO (ou ligue o value1 do For Loop Start). No fim do laco, mande a saida daqui de volta pro For Loop End."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("acumulado", "count", "info")
    OUTPUT_TOOLTIPS = (
        "A pilha com este ladrilho somado aos anteriores. Ligue no For Loop End (initial_value) "
        "e, no fim do laco, mande pro Tile Merge.",
        "Quantos ladrilhos ja foram acumulados (x frames).",
        "Resumo do acumulo.",
    )
    FUNCTION = "acc"
    CATEGORY = CAT
    DESCRIPTION = ("Empilha os ladrilhos processados a cada volta do For Loop, na ordem certa. "
                   "E a peca que fecha o laco: sem ela, o Tile Merge so receberia UM tile. "
                   "Saida -> For Loop End; no fim, -> Tile Merge.")

    def acc(self, tile, acumulado=None):
        t = tile.float()
        if acumulado is None or not hasattr(acumulado, "shape") or int(acumulado.shape[0]) == 0:
            out = t
        else:
            a = acumulado.float()
            if a.shape[1:] != t.shape[1:]:
                raise RuntimeError(
                    f"[Bruxos Tile Accumulate] os ladrilhos tem tamanhos diferentes "
                    f"({tuple(a.shape[1:])} vs {tuple(t.shape[1:])}). Todos os tiles precisam sair do "
                    f"Wan com o MESMO tamanho. Confira se a resolucao do sampler e igual pra todos."
                )
            out = torch.cat([a, t], dim=0)
        info = f"acumulados: {int(out.shape[0])} imagem(ns) ({tuple(out.shape[1:3])})"
        print(f"[Bruxos Tile Accumulate] {info}", flush=True)
        return (out.clamp(0, 1), int(out.shape[0]), info)


NODE_CLASS_MAPPINGS = {
    "BruxosTileSplit": BruxosTileSplit,
    "BruxosTileSelect": BruxosTileSelect,
    "BruxosTileAccumulate": BruxosTileAccumulate,
    "BruxosTileMerge": BruxosTileMerge,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosTileSplit": "Tile Split (Bruxos)",
    "BruxosTileSelect": "Tile Select (Bruxos)",
    "BruxosTileAccumulate": "Tile Accumulate (Bruxos)",
    "BruxosTileMerge": "Tile Merge (Bruxos)",
}
