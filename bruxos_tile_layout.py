"""Layout espacial customizado para o tiled step-fused do Bernini/Wan."""

from __future__ import annotations

import json

import torch

from .bruxos_video_tiler import MARCA


CAT = "Bruxos do VFX/Bernini/Tiles"
MAX_TILES = 24

DEFAULT_LAYOUT = {
    "version": 1,
    "tiles": [
        {"id": 1, "x0": 0.00, "y0": 0.00, "x1": 0.56, "y1": 0.56},
        {"id": 2, "x0": 0.44, "y0": 0.00, "x1": 1.00, "y1": 0.56},
        {"id": 3, "x0": 0.00, "y0": 0.44, "x1": 0.56, "y1": 1.00},
        {"id": 4, "x0": 0.44, "y0": 0.44, "x1": 1.00, "y1": 1.00},
        {"id": 5, "x0": 0.25, "y0": 0.25, "x1": 0.75, "y1": 0.75},
    ],
}


def _clamp(value):
    return max(0.0, min(1.0, float(value)))


def _parse_layout(raw):
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(str(raw or ""))
        except Exception as exc:
            raise ValueError(f"[Bruxos Tile Layout] JSON invalido: {exc}") from exc
    source = data.get("tiles", data) if isinstance(data, dict) else data
    if not isinstance(source, list):
        raise ValueError("[Bruxos Tile Layout] o layout precisa conter uma lista 'tiles'.")
    if not 1 <= len(source) <= MAX_TILES:
        raise ValueError(f"[Bruxos Tile Layout] use entre 1 e {MAX_TILES} tiles.")
    tiles = []
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            raise ValueError(f"[Bruxos Tile Layout] tile {index + 1} nao e um objeto.")
        try:
            x0, y0 = _clamp(item["x0"]), _clamp(item["y0"])
            x1, y1 = _clamp(item["x1"]), _clamp(item["y1"])
        except Exception as exc:
            raise ValueError(f"[Bruxos Tile Layout] coordenadas invalidas no tile {index + 1}.") from exc
        if x1 - x0 < 0.02 or y1 - y0 < 0.02:
            raise ValueError(f"[Bruxos Tile Layout] tile {index + 1} e pequeno ou invertido.")
        tiles.append({
            "id": int(item.get("id", index + 1)),
            "x0": round(x0, 6), "y0": round(y0, 6),
            "x1": round(x1, 6), "y1": round(y1, 6),
            "weight": max(0.05, min(8.0, float(item.get("weight", 1.0)))),
        })
    return tiles


def _coverage(tiles, samples=128):
    # Valida em coordenadas normalizadas; o motor repete uma validacao exata no
    # grid latente antes de amostrar.
    missing = 0
    for yi in range(samples):
        y = (yi + 0.5) / samples
        for xi in range(samples):
            x = (xi + 0.5) / samples
            if not any(t["x0"] <= x <= t["x1"] and t["y0"] <= y <= t["y1"] for t in tiles):
                missing += 1
    return 1.0 - missing / float(samples * samples)


class BruxosTileLayoutCustom:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "canvas_width": ("INT", {"default": 832, "min": 64, "max": 8192, "step": 16,
                                      "tooltip": "Proporcao visual do editor; o layout final e normalizado."}),
            "canvas_height": ("INT", {"default": 480, "min": 64, "max": 8192, "step": 16}),
            "layout_json": ("STRING", {"default": json.dumps(DEFAULT_LAYOUT, separators=(",", ":")),
                                        "multiline": True,
                                        "tooltip": "Estado persistente do editor visual."}),
        }}

    RETURN_TYPES = ("BRUXOS_TILE_LAYOUT", "STRING")
    RETURN_NAMES = ("tile_layout", "info")
    FUNCTION = "build"
    CATEGORY = CAT
    DESCRIPTION = (
        "Editor visual de tiles normalizados. Ligue tile_layout ao Bernini com guidance_mode=tiled; "
        "quando conectado, substitui tile_w x tile_h."
    )

    def build(self, canvas_width, canvas_height, layout_json):
        tiles = _parse_layout(layout_json)
        coverage = _coverage(tiles)
        if coverage < 0.9999:
            raise ValueError(
                f"[Bruxos Tile Layout] o layout cobre somente {coverage * 100:.2f}% do canvas. "
                "Aumente/mova os tiles ate nao restarem buracos."
            )
        layout = {
            "version": 1,
            "tiles": tiles,
            "editor_width": int(canvas_width),
            "editor_height": int(canvas_height),
            "coverage": coverage,
        }
        largest = max((t["x1"] - t["x0"]) * (t["y1"] - t["y0"]) for t in tiles)
        info = (f"{len(tiles)} tiles custom | cobertura {coverage * 100:.1f}% | "
                f"maior tile {largest * 100:.1f}% da area")
        return layout, info


class BruxosTileSliceCustomLayout:
    """Converte o layout normalizado do editor na geometria BXT1 do tiler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "imagens": ("IMAGE", {"tooltip": "Video/imagem completa que sera recortada."}),
            "tile_layout": ("BRUXOS_TILE_LAYOUT", {"forceInput": True}),
            "multiplo": ("INT", {"default": 32, "min": 1, "max": 64, "step": 1,
                                   "tooltip": "MiniMax H3: 32; Bernini/Wan 14B: 16."}),
        }}

    RETURN_TYPES = ("IMAGE", "TILE_CONFIG", "IMAGE", "INT", "STRING", "INT", "INT")
    RETURN_NAMES = ("tiles", "tile_config", "mapa", "quantidade", "info",
                    "larguras", "alturas")
    OUTPUT_IS_LIST = (True, False, False, False, False, True, True)
    FUNCTION = "run"
    CATEGORY = "Bruxos do VFX/Tiler"
    DESCRIPTION = (
        "Recorta partes diferentes usando o Bernini Custom Tile Layout e cria o TILE_CONFIG "
        "compativel com Tile Merge, Tile Ref Slice e os nodes SSD do Video Tiler."
    )

    @staticmethod
    def _borda(valor, tamanho, multiplo):
        valor = max(0.0, min(1.0, float(valor)))
        if valor <= 0.0:
            return 0
        if valor >= 1.0:
            return int(tamanho)
        return max(0, min(int(tamanho), int(round(valor * tamanho / multiplo)) * multiplo))

    def run(self, imagens, tile_layout, multiplo):
        if getattr(imagens, "ndim", 0) != 4:
            raise ValueError("[Bruxos Custom Tile Slice] imagens precisa ser [B,H,W,C].")
        if not isinstance(tile_layout, dict) or not tile_layout.get("tiles"):
            raise ValueError("[Bruxos Custom Tile Slice] tile_layout invalido ou vazio.")

        _B, H, W, _C = (int(v) for v in imagens.shape)
        m = max(1, min(64, int(multiplo)))
        if W % m or H % m:
            raise ValueError(
                f"[Bruxos Custom Tile Slice] quadro {W}x{H} fora da grade {m}. "
                f"Redimensione antes para {W - W % m}x{H - H % m}."
            )

        specs = []
        for ordem, src in enumerate(tile_layout["tiles"]):
            x0 = self._borda(src["x0"], W, m)
            y0 = self._borda(src["y0"], H, m)
            x1 = self._borda(src["x1"], W, m)
            y1 = self._borda(src["y1"], H, m)
            if x1 <= x0:
                x1 = min(W, x0 + m)
            if y1 <= y0:
                y1 = min(H, y0 + m)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"[Bruxos Custom Tile Slice] tile {ordem + 1} colapsou na grade {m}.")
            specs.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                          "col": -1, "row": -1, "ordem": ordem})

        cobertura = torch.zeros((H, W), dtype=torch.bool)
        mapa = torch.full((1, H, W, 3), 0.10, dtype=torch.float32)
        n = len(specs)
        for i, s in enumerate(specs):
            x, y, w, h = s["x"], s["y"], s["w"], s["h"]
            cobertura[y:y + h, x:x + w] = True
            u = i / max(1, n - 1)
            cor = torch.tensor([0.15 + 0.8 * u, 0.50, 0.95 - 0.7 * u])
            e = max(2, min(H, W) // 300)
            mapa[0, y:y + e, x:x + w] = cor
            mapa[0, y + h - e:y + h, x:x + w] = cor
            mapa[0, y:y + h, x:x + e] = cor
            mapa[0, y:y + h, x + w - e:x + w] = cor
        if not bool(cobertura.all()):
            faltam = int((~cobertura).sum().item())
            raise ValueError(
                f"[Bruxos Custom Tile Slice] o arredondamento para a grade {m} deixou "
                f"{faltam} pixels sem cobertura. Aumente levemente a sobreposicao no editor."
            )

        oxs, oys = [], []
        for i, a in enumerate(specs):
            for b in specs[i + 1:]:
                ix = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
                iy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
                if ix > 0 and iy > 0:
                    if a["x"] != b["x"] or a["w"] != b["w"]:
                        oxs.append(ix)
                    if a["y"] != b["y"] or a["h"] != b["h"]:
                        oys.append(iy)
        ox = min(oxs) if oxs else 0
        oy = min(oys) if oys else 0
        lw = max(s["w"] for s in specs)
        lh = max(s["h"] for s in specs)
        cfg = (MARCA, W, H, lw, lh, int(ox), int(oy), m,
               tuple((s["x"], s["y"], s["w"], s["h"], s["col"], s["row"], s["ordem"])
                     for s in specs))
        recortes = [imagens[:, s["y"]:s["y"] + s["h"],
                            s["x"]:s["x"] + s["w"], :] for s in specs]
        info = (f"custom {n} tiles | quadro {W}x{H} | grade {m} | "
                f"overlap estimado {ox}x{oy}px")
        print(f"[Bruxos Custom Tile Slice] {info}", flush=True)
        return (recortes, cfg, mapa, n, info,
                [s["w"] for s in specs], [s["h"] for s in specs])


NODE_CLASS_MAPPINGS = {
    "BruxosTileLayoutCustom": BruxosTileLayoutCustom,
    "BruxosTileSliceCustomLayout": BruxosTileSliceCustomLayout,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosTileLayoutCustom": "Bernini Custom Tile Layout (Bruxos)",
    "BruxosTileSliceCustomLayout": "Tile Slice Custom Layout (Bruxos)",
}
