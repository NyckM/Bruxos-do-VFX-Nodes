# -*- coding: utf-8 -*-
"""Expansao temporal: poucos frames-guia viram uma timeline maior para o Bernini."""

from __future__ import annotations

import torch


class BruxosTemporalAnchorSetup:
    """Calcula uma timeline inteira e alimenta os widgets ligados no workflow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "requested_output_frames": ("INT", {"default": 81, "min": 1, "max": 4097, "step": 1,
                "tooltip": "Quantidade desejada. O modo auto corrige para a grade temporal 4n+1."}),
            "output_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
            "anchor_stride": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1,
                "tooltip": "2 = uma ancora para cada dois frames finais."}),
            "grid_mode": (["nearest_4n1", "ceil_4n1", "floor_4n1", "strict_4n1"],
                          {"default": "nearest_4n1"}),
            "max_chunk": ("INT", {"default": 81, "min": 5, "max": 4097, "step": 4,
                "tooltip": "Acima deste total, o Bernini divide em janelas. 81 e seguro para VRAM."}),
            "chunk_overlap": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
        }}

    RETURN_TYPES = ("INT", "FLOAT", "INT", "FLOAT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "anchor_frames", "anchor_fps", "output_frames", "output_fps",
        "chunk_size", "overlap", "duration_seconds", "info",
    )
    FUNCTION = "calculate"
    CATEGORY = "Bruxos do VFX/Bernini"
    DESCRIPTION = (
        "Planeja automaticamente Load Video, Temporal Anchors, chunk do Bernini e FPS do Combine."
    )

    @staticmethod
    def _grid_4n1(value, mode):
        value = max(1, int(value))
        if value % 4 == 1:
            return value
        floor_value = max(1, value - ((value - 1) % 4))
        ceil_value = floor_value + 4
        if mode == "strict_4n1":
            raise ValueError(
                f"[Bruxos Anchor Setup] {value} nao e 4n+1. "
                f"Use {floor_value} ou {ceil_value}."
            )
        if mode == "ceil_4n1":
            return ceil_value
        if mode == "floor_4n1":
            return floor_value
        return floor_value if value - floor_value < ceil_value - value else ceil_value

    def calculate(self, requested_output_frames=81, output_fps=24.0, anchor_stride=2,
                  grid_mode="nearest_4n1", max_chunk=81, chunk_overlap=8):
        target = self._grid_4n1(requested_output_frames, grid_mode)
        stride = max(1, int(anchor_stride))
        fps = max(0.001, float(output_fps))

        # Inclui as duas extremidades. Para 81/stride 2: (80/2)+1 = 41.
        anchors = ((target - 1) + stride - 1) // stride + 1
        actual_stride = (target - 1) / max(1, anchors - 1)
        anchor_fps = fps / actual_stride if target > 1 else fps

        limit = max(5, int(max_chunk))
        limit = max(5, 1 + 4 * ((limit - 1) // 4))
        chunk = target if target <= limit else limit
        overlap = min(max(0, int(chunk_overlap)), max(0, chunk - 1))
        # Duracao do arquivo salvo (mesma convencao usada pelos nodes de video: F / fps).
        duration = target / fps
        adjusted = "" if target == int(requested_output_frames) else f" (ajustado de {requested_output_frames})"
        info = (
            f"saida {target}{adjusted} @ {fps:g} fps | "
            f"clay {anchors} @ {anchor_fps:g} fps | stride real {actual_stride:.3f} | "
            f"chunk {chunk}, overlap {overlap} | duracao {duration:.3f}s"
        )
        print(f"[Bruxos Anchor Setup] {info}", flush=True)
        return anchors, anchor_fps, target, fps, chunk, overlap, duration, info


class BruxosTemporalAnchorExpand:
    """Distribui frames de referencia pela timeline e preenche os intervalos.

    Para 41 -> 81, os 41 frames originais ocupam exatamente 0, 2, 4, ... 80.
    Assim o Bernini recebe um source de 81 frames e realmente denoisa 81 frames,
    mas o arquivo clay precisa fornecer apenas 41 ancoras.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anchor_frames": ("IMAGE", {"tooltip":
                    "Frames-guia em ordem temporal. Ex.: clay a 12 fps com 41 frames."}),
                "output_frames": ("INT", {"default": 81, "min": 1, "max": 4097, "step": 4,
                    "tooltip": "Total entregue ao Bernini. Para Wan/Bernini use 4n+1: 41, 81, 121..."}),
                "interpolation": (["linear", "nearest", "hold_previous"], {"default": "linear",
                    "tooltip": "linear e recomendado para clay. Os frames-ancora permanecem exatos quando a razao casa."}),
            },
            "optional": {
                "require_4n1": ("BOOLEAN", {"default": True,
                    "tooltip": "Recusa uma saida fora da grade temporal 4n+1 do Bernini/Wan."}),
                "expected_anchor_frames": ("INT", {"default": 0, "min": 0, "max": 4097, "step": 1,
                    "tooltip": "Ligue anchor_frames do Anchor Setup para validar se o arquivo tinha frames suficientes."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("expanded_frames", "frame_count", "info")
    FUNCTION = "expand"
    CATEGORY = "Bruxos do VFX/Bernini"
    DESCRIPTION = (
        "Espalha N frames clay como ancoras por uma timeline maior. "
        "41 -> 81 preserva cada frame original nas posicoes pares e interpola as impares."
    )

    @staticmethod
    def _validate(images, target, require_4n1):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("[Bruxos Temporal Anchors] anchor_frames precisa ser IMAGE [F,H,W,C].")
        if int(images.shape[0]) < 1:
            raise ValueError("[Bruxos Temporal Anchors] nenhuma ancora recebida.")
        if target < 1:
            raise ValueError("[Bruxos Temporal Anchors] output_frames precisa ser positivo.")
        if require_4n1 and target % 4 != 1:
            before = target - ((target - 1) % 4)
            after = before + 4
            raise ValueError(
                f"[Bruxos Temporal Anchors] {target} nao e 4n+1. "
                f"Use {before} ou {after}."
            )

    def expand(self, anchor_frames, output_frames=81, interpolation="linear", require_4n1=True,
               expected_anchor_frames=0):
        target = int(output_frames)
        self._validate(anchor_frames, target, bool(require_4n1))
        source_count = int(anchor_frames.shape[0])
        expected = int(expected_anchor_frames)
        if expected > 0 and source_count != expected:
            raise ValueError(
                f"[Bruxos Temporal Anchors] eram esperadas {expected} ancoras, "
                f"mas o Load Video entregou {source_count}. O arquivo pode ser curto demais "
                f"para a duracao escolhida."
            )

        if target == source_count:
            info = f"{source_count} ancoras -> {target} frames (sem expansao)"
            return anchor_frames, target, info

        # Posicao de cada frame de saida na timeline das ancoras. align_corners
        # temporal: primeira e ultima ancoras sempre coincidem com as extremidades.
        positions = torch.linspace(
            0.0, float(source_count - 1), target,
            device=anchor_frames.device, dtype=torch.float32,
        )
        lo = positions.floor().long().clamp(0, source_count - 1)
        hi = positions.ceil().long().clamp(0, source_count - 1)

        out_shape = (target,) + tuple(anchor_frames.shape[1:])
        expanded = torch.empty(out_shape, dtype=anchor_frames.dtype, device=anchor_frames.device)

        # Blocos pequenos evitam manter dois videos temporarios completos durante o lerp.
        block = 8
        with torch.no_grad():
            for start in range(0, target, block):
                end = min(target, start + block)
                if interpolation == "nearest":
                    indices = positions[start:end].round().long().clamp(0, source_count - 1)
                    expanded[start:end] = anchor_frames.index_select(0, indices)
                elif interpolation == "hold_previous":
                    expanded[start:end] = anchor_frames.index_select(0, lo[start:end])
                else:
                    left = anchor_frames.index_select(0, lo[start:end])
                    right = anchor_frames.index_select(0, hi[start:end])
                    alpha = (positions[start:end] - lo[start:end].float()).to(anchor_frames.dtype)
                    alpha = alpha.view(-1, 1, 1, 1)
                    expanded[start:end] = torch.lerp(left, right, alpha)

        exact_positions = torch.linspace(0.0, float(target - 1), source_count).round().long()
        ratio = (target - 1) / max(1, source_count - 1)
        info = (
            f"{source_count} ancoras -> {target} frames | {interpolation} | "
            f"espacamento medio {ratio:.3f} | ancoras em "
            f"{int(exact_positions[0])}..{int(exact_positions[-1])}"
        )
        print(f"[Bruxos Temporal Anchors] {info}", flush=True)
        return expanded, target, info


NODE_CLASS_MAPPINGS = {
    "BruxosTemporalAnchorSetup": BruxosTemporalAnchorSetup,
    "BruxosTemporalAnchorExpand": BruxosTemporalAnchorExpand,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosTemporalAnchorSetup": "Bernini Anchor Timeline Auto (Bruxos)",
    "BruxosTemporalAnchorExpand": "Bernini Temporal Anchors Expand (Bruxos)",
}
