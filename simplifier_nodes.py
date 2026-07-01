# -*- coding: utf-8 -*-
"""Bruxos do VFX - Super-nodes 'simplificadores' do MM Upscale.
Encapsulam subgraphs inteiros do workflow em um node só:
  - BruxosUpscaleConfig  -> subgraph 'Settings' (sliders de resolucao/passos/etc.)
  - BruxosBatchBlend     -> subgraph 'Blend Frames' (junta batches com transicao)
"""

import logging

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

CAT = "Bruxos do VFX/Upscale"


# =============================================================================
# Config de Upscale (Bruxos)  ->  subgraph "Settings"
# =============================================================================
class BruxosUpscaleConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_width": ("INT", {"default": 960, "min": 16, "max": 8192, "step": 8,
                    "tooltip": "Largura alvo do upscale."}),
                "target_height": ("INT", {"default": 960, "min": 16, "max": 8192, "step": 8,
                    "tooltip": "Altura alvo do upscale."}),
                "frames_por_iteracao": ("INT", {"default": 81, "min": 5, "max": 1000, "step": 4,
                    "tooltip": "Frames processados por iteração. WAN exige múltiplo de 4 + 1 (81, 85, 89...). O node ajusta se não for."}),
                "overlap_frames": ("INT", {"default": 48, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Frames de sobreposição entre iterações (pro blend costurar). Deve ser menor que frames_por_iteracao."}),
                "alta_qualidade": ("BOOLEAN", {"default": False,
                    "tooltip": "OFF = 1 passo (rápido, qualidade normal). ON = 2 passos (mais detalhe, mais lento)."}),
                "criatividade": ("FLOAT", {"default": 0.408, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Ajusta o shift do scheduler. Menor = mais fiel à fonte; maior = mais generativo/detalhe (pode gerar ruído)."}),
                "salvar_sequencia": ("BOOLEAN", {"default": False,
                    "tooltip": "Salvar a sequência de imagens em disco a cada iteração."}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("width", "height", "frames_iter", "overlap", "steps", "shift", "salvar")
    OUTPUT_TOOLTIPS = (
        "Largura alvo.", "Altura alvo.",
        "Frames por iteração (ajustado p/ 4n+1).", "Overlap entre iterações.",
        "Passos do sampler (1 ou 2, de alta_qualidade).",
        "Shift/criatividade p/ o ModelSamplingSD3.",
        "Salvar sequência (bool).",
    )
    FUNCTION = "build"
    CATEGORY = CAT
    DESCRIPTION = ("Config central do MM Upscale num node só (substitui o subgraph 'Settings'): "
                   "resolução alvo, frames por iteração, overlap, passos (alta qualidade), criatividade e salvar sequência.")

    def build(self, target_width, target_height, frames_por_iteracao, overlap_frames,
              alta_qualidade, criatividade, salvar_sequencia):
        f = int(frames_por_iteracao)
        # forca 4n + 1 (exigencia do WAN)
        if (f - 1) % 4 != 0:
            f = ((f - 1) // 4) * 4 + 1
            logging.info(f"[Bruxos UpscaleConfig] frames_por_iteracao ajustado p/ 4n+1: {f}")
        ov = int(overlap_frames)
        if ov >= f:
            ov = max(0, f - 1)
            logging.info(f"[Bruxos UpscaleConfig] overlap reduzido p/ < frames_iter: {ov}")
        steps = 2 if bool(alta_qualidade) else 1
        return (int(target_width), int(target_height), int(f), int(ov),
                int(steps), float(criatividade), bool(salvar_sequencia))


# =============================================================================
# Blend de Batches (Bruxos)  ->  subgraph "Blend Frames"
# =============================================================================
def _match_hw(b, ref):
    """Redimensiona b [T,H,W,C] p/ H,W de ref."""
    if b.shape[1:3] == ref.shape[1:3]:
        return b
    x = b.permute(0, 3, 1, 2)
    x = torch.nn.functional.interpolate(x, size=(ref.shape[1], ref.shape[2]),
                                        mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


class BruxosBatchBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_a": ("IMAGE", {"tooltip": "Primeiro lote (ex.: iteração A / passo anterior)."}),
                "batch_b": ("IMAGE", {"tooltip": "Segundo lote (ex.: iteração B / passo seguinte)."}),
                "modo": (["transicao", "misturar", "media"], {"default": "transicao",
                    "tooltip": "transicao = concatena A→B com fade nas bordas (junta iterações). misturar = mescla ponderada quadro a quadro. media = média simples."}),
                "transicao_frames": ("INT", {"default": 4, "min": 0, "max": 240, "step": 1,
                    "tooltip": "[modo transicao] Nº de frames do crossfade entre A e B (costura o overlap). Igual ao ImageBatchJoinWithTransition."}),
                "mistura": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "[modo misturar] Peso de B (0 = só A, 1 = só B)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "blend"
    CATEGORY = CAT
    DESCRIPTION = ("Junta/mistura dois lotes de imagens (substitui o subgraph 'Blend Frames'): "
                   "transição com fade entre iterações, mistura ponderada ou média.")

    def blend(self, batch_a, batch_b, modo="transicao", transicao_frames=4, mistura=0.5):
        if not _HAS_TORCH:
            raise RuntimeError("[Bruxos BatchBlend] torch indisponível.")
        a = batch_a.float()
        b = _match_hw(batch_b.float(), a)

        if modo == "misturar":
            n = min(a.shape[0], b.shape[0])
            w = float(mistura)
            out = a[:n] * (1.0 - w) + b[:n] * w
            return (out.clamp(0, 1),)

        if modo == "media":
            n = min(a.shape[0], b.shape[0])
            return ((a[:n] + b[:n]) * 0.5).clamp(0, 1),

        # modo == "transicao": concatena A -> B com crossfade de `transicao_frames`
        t = int(transicao_frames)
        if t <= 0:
            return (torch.cat([a, b], dim=0).clamp(0, 1),)
        t = min(t, a.shape[0], b.shape[0])
        head = a[:a.shape[0] - t]                       # A sem a cauda
        a_tail = a[a.shape[0] - t:]                     # cauda de A (t frames)
        b_head = b[:t]                                  # cabeça de B (t frames)
        # pesos lineares 0..1 ao longo da transicao
        wts = torch.linspace(0, 1, steps=t, dtype=a.dtype).view(t, 1, 1, 1)
        mid = a_tail * (1.0 - wts) + b_head * wts       # crossfade
        tail = b[t:]                                    # resto de B
        out = torch.cat([head, mid, tail], dim=0)
        return (out.clamp(0, 1),)


NODE_CLASS_MAPPINGS = {
    "BruxosUpscaleConfig": BruxosUpscaleConfig,
    "BruxosBatchBlend": BruxosBatchBlend,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosUpscaleConfig": "Config de Upscale (Bruxos)",
    "BruxosBatchBlend": "Blend de Batches (Bruxos)",
}
