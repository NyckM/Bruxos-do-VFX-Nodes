# -*- coding: utf-8 -*-
"""Planejamento leve de referencia clay para MiniMax H3.

A imagem de aparencia continua independente e em alta resolucao. O clay e
carregado em resolucao menor e com poucas ancoras temporais; outro node
reconstroi a timeline completa antes do condicionamento nativo do H3.
"""

from __future__ import annotations


PASSO_H3 = 17
RESTO_H3 = 5


def _grade_h3(value: int, mode: str) -> int:
    value = max(RESTO_H3, int(value))
    if value % PASSO_H3 == RESTO_H3:
        return value
    lower = value - ((value - RESTO_H3) % PASSO_H3)
    lower = max(RESTO_H3, lower)
    upper = lower + PASSO_H3
    if mode == "strict_h3":
        raise ValueError(
            f"[Bruxos H3 Clay] {value} nao pertence a grade H3. "
            f"Use {lower} ou {upper} (length % 17 == 5)."
        )
    if mode == "ceil_h3":
        return upper
    if mode == "floor_h3":
        return lower
    return lower if value - lower < upper - value else upper


def _multiple_32(value: int) -> int:
    """Arredonda para o multiplo de 32 mais proximo, sem dimensao zero."""
    return max(32, int(round(max(32, int(value)) / 32.0)) * 32)


class BruxosH3ClayReferenceSetup:
    """Calcula todos os numeros ligados ao Load Video e a timeline H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "requested_output_frames": ("INT", {
                "default": 121, "min": 5, "max": 4097, "step": 1,
                "tooltip": "O modo auto converte para a grade do H3. 121 vira 124.",
            }),
            "output_fps": ("FLOAT", {
                "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
            }),
            "anchor_stride": ("INT", {
                "default": 2, "min": 1, "max": 16, "step": 1,
                "tooltip": "2 carrega aproximadamente metade dos frames do clay.",
            }),
            "grid_mode": (["ceil_h3", "nearest_h3", "floor_h3", "strict_h3"], {
                "default": "ceil_h3",
            }),
            "clay_width": ("INT", {
                "default": 352, "min": 32, "max": 4096, "step": 32,
                "tooltip": "Resolucao leve da referencia, nao da saida. Multiplo de 32.",
            }),
            "clay_height": ("INT", {
                "default": 640, "min": 32, "max": 4096, "step": 32,
                "tooltip": "Resolucao leve da referencia, nao da imagem de aparencia.",
            }),
        }}

    RETURN_TYPES = ("INT", "FLOAT", "INT", "FLOAT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "anchor_frames", "anchor_fps", "output_frames", "output_fps",
        "clay_width", "clay_height", "duration_seconds", "info",
    )
    FUNCTION = "calculate"
    CATEGORY = "Bruxos do VFX/MiniMax H3"
    DESCRIPTION = (
        "Mantem a imagem em alta e planeja uma referencia clay leve. Liga o Load Video "
        "a aproximadamente metade do FPS e da resolucao, preservando o mesmo intervalo temporal."
    )

    def calculate(self, requested_output_frames=121, output_fps=24.0, anchor_stride=2,
                  grid_mode="ceil_h3", clay_width=352, clay_height=640):
        target = _grade_h3(requested_output_frames, grid_mode)
        fps = max(0.001, float(output_fps))
        stride = max(1, int(anchor_stride))

        # Inclui primeira e ultima amostras. Para 124/stride 2: 63 ancoras.
        anchors = ((target - 1) + stride - 1) // stride + 1
        actual_stride = (target - 1) / max(1, anchors - 1)
        anchor_fps = fps / actual_stride if target > 1 else fps

        width = _multiple_32(clay_width)
        height = _multiple_32(clay_height)
        duration = target / fps
        adjusted = "" if target == int(requested_output_frames) else f" (ajustado de {requested_output_frames})"
        info = (
            f"H3 {target}{adjusted} @ {fps:g} fps | "
            f"clay {anchors} ancoras @ {anchor_fps:.5g} fps, {width}x{height} | "
            f"stride real {actual_stride:.3f} | imagem de aparencia permanece separada/em alta | "
            f"duracao {duration:.3f}s"
        )
        print(f"[Bruxos H3 Clay] {info}", flush=True)
        return anchors, anchor_fps, target, fps, width, height, duration, info


NODE_CLASS_MAPPINGS = {
    "BruxosH3ClayReferenceSetup": BruxosH3ClayReferenceSetup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3ClayReferenceSetup": "MiniMax H3 Clay Reference Auto (Bruxos)",
}
