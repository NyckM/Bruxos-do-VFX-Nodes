# -*- coding: utf-8 -*-
"""
Bruxos do VFX - Editor de Pontos (SAM3).

Editor visual (canvas no navegador) para marcar pontos de SELECAO (verde)
e NEGACAO (roxo) sobre um frame, alem de caixas (bbox). Serve para gerar
`initial_mask` / prompts de ponto para o rastreamento SAM3, sem depender
de texto ("face center") que pode perder o alvo ao longo do video.

Compativel por formato com o "Frames Editor" do comfyui-easy-sam3: produz
`positive_coords` / `negative_coords` (STRING, JSON de {"x":..,"y":..}),
`bboxes` (BBOX) e `frame_index` (INT), prontos para ligar em
`Sam3VideoSegmentation` / `Sam3ImageSegmentation` (positive_coords /
negative_coords / bboxes).
"""

import json
import random
import hashlib

import numpy as np
import torch

try:
    import folder_paths
    _HAS_FP = True
except Exception:
    _HAS_FP = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    from server import PromptServer
    _HAS_SERVER = True
except Exception:
    _HAS_SERVER = False


CAT = "Bruxos do VFX/SAM3"


def _tensor_to_pil_list(images):
    """IMAGE (B,H,W,3) float 0..1 -> lista de PIL.Image RGB."""
    out = []
    arr = images.detach().cpu().numpy()
    for i in range(arr.shape[0]):
        frame = np.clip(arr[i] * 255.0, 0, 255).astype(np.uint8)
        out.append(Image.fromarray(frame, mode="RGB"))
    return out


class BruxosPointsEditor:
    """Editor de Pontos (Bruxos) - marca selecao (verde) / negacao (roxo)."""

    _state = {"last_hash": None, "cached_preview": None}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frames de entrada (video ou imagem) para marcar os pontos."}),
                # widget escondido: o JS grava aqui o JSON com os cliques
                "info": ("STRING", {"default": "", "multiline": False}),
                "preview_rescale": ("FLOAT", {
                    "default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Reduz o preview p/ carregar mais rapido; as coordenadas voltam pra escala original.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "BBOX", "INT")
    RETURN_NAMES = ("positive_coords", "negative_coords", "bboxes", "frame_index")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT
    DESCRIPTION = (
        "Marque no frame: clique ESQUERDO = ponto verde (selecionar), clique "
        "DIREITO = ponto roxo (negar). Arraste com Shift p/ caixa. Saidas prontas "
        "p/ Sam3VideoSegmentation / Sam3ImageSegmentation (positive_coords / "
        "negative_coords / bboxes)."
    )

    def run(self, images, info="", preview_rescale=1.0):
        positive_coords = None
        negative_coords = None
        bboxes = []
        frame_index = 0

        needs_scaling = 0.0 < preview_rescale < 1.0
        scale_factor = (1.0 / preview_rescale) if needs_scaling else 1.0

        if info:
            try:
                data = json.loads(info)
            except (json.JSONDecodeError, TypeError):
                data = None
            if data:
                pos = data.get("positive_coords")
                neg = data.get("negative_coords")
                box = data.get("bbox")
                frame_index = int(data.get("frame_index", 0) or 0)

                if needs_scaling and pos:
                    pos = [{"x": c["x"] * scale_factor, "y": c["y"] * scale_factor} for c in pos]
                if needs_scaling and neg:
                    neg = [{"x": c["x"] * scale_factor, "y": c["y"] * scale_factor} for c in neg]

                if box:
                    for b in box:
                        if needs_scaling:
                            x, y, w, h = (b["x"] * scale_factor, b["y"] * scale_factor,
                                          b["w"] * scale_factor, b["h"] * scale_factor)
                        else:
                            x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                        bboxes.append([x, y, x + w, y + h])

                if pos is not None:
                    positive_coords = json.dumps(pos, ensure_ascii=False)
                if neg is not None:
                    negative_coords = json.dumps(neg, ensure_ascii=False)

        # ---- preview (reaproveita o cache se os frames nao mudaram) ----
        preview_str = ""
        if _HAS_PIL and _HAS_FP and _HAS_SERVER:
            preview_images = images
            if needs_scaling:
                _, h, w, _ = images.shape
                nh, nw = max(1, int(h * preview_rescale)), max(1, int(w * preview_rescale))
                pil_list = _tensor_to_pil_list(images)
                resized = [im.resize((nw, nh), Image.LANCZOS) for im in pil_list]
                arr = np.stack([np.array(im, dtype=np.float32) / 255.0 for im in resized], axis=0)
                preview_images = torch.from_numpy(arr)

            images_hash = hashlib.md5(preview_images.detach().cpu().numpy().tobytes()).hexdigest()
            rescale_hash = f"{images_hash}_{preview_rescale}"

            if self._state.get("last_hash") == rescale_hash and self._state.get("cached_preview"):
                preview_str = self._state["cached_preview"]
                is_init = False
            else:
                saved = []
                out_dir = folder_paths.get_temp_directory()
                import os
                os.makedirs(out_dir, exist_ok=True)
                prefix = "bruxos_points_" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
                pil_list = _tensor_to_pil_list(preview_images)
                for idx, im in enumerate(pil_list):
                    fname = f"{prefix}_{idx:05d}.png"
                    im.save(os.path.join(out_dir, fname), compress_level=4)
                    saved.append({"filename": fname, "subfolder": "", "type": "temp"})
                preview_str = json.dumps(saved, ensure_ascii=False)
                self._state["last_hash"] = rescale_hash
                self._state["cached_preview"] = preview_str
                is_init = True
        else:
            is_init = True

        return {
            "ui": {"preview": [{"preview_str": preview_str, "is_init": is_init}]},
            "result": (positive_coords, negative_coords, bboxes, frame_index),
        }


NODE_CLASS_MAPPINGS = {
    "BruxosPointsEditor": BruxosPointsEditor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosPointsEditor": "Editor de Pontos SAM3 (Bruxos)",
}
