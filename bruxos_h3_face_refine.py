"""Extensoes H3 para refino temporal de rostos pequenos.

O conceito de denoise por tamanho de rosto foi adaptado de
ComfyUI-H3-FaceRefine, Copyright (c) 2026 Carasibana, licença MIT.
Usa o TRACK_CROP dos Bruxos e preserva explicitamente a mascara de audio.
"""
import torch
import torch.nn.functional as F

import comfy.nested_tensor


CAT = "Bruxos do VFX/MiniMax H3/Face Refine"


def _nested_members(value):
    if isinstance(value, comfy.nested_tensor.NestedTensor) or getattr(value, "is_nested", False):
        return list(value.unbind())
    values = getattr(value, "tensors", None)
    if isinstance(values, (list, tuple)):
        return list(values)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _smooth_curve(values, window):
    window = max(1, int(window))
    if window <= 1 or values.numel() <= 2:
        return values
    if window % 2 == 0:
        window += 1
    radius = window // 2
    coords = torch.arange(window, dtype=torch.float32) - radius
    sigma = max(window / 6.0, 0.5)
    kernel = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    padded = F.pad(values.view(1, 1, -1), (radius, radius), mode="replicate")
    return F.conv1d(padded, kernel.view(1, 1, -1))[0, 0]


class BruxosH3FacePerFrameDenoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "av_latent": ("LATENT", {"tooltip": "Latente H3 AV, de preferencia depois do Audio Lock."}),
            "crop_data": ("TRACK_CROP", {"tooltip": "Saida do Tracked Crop / SAM3 (Bruxos)."}),
            "strength_small_face": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "strength_large_face": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
            "scale_mode": (["absolute_px", "relative_to_clip"], {"default": "absolute_px"}),
            "face_px_small": ("FLOAT", {"default": 30.0, "min": 4.0, "max": 400.0, "step": 1.0}),
            "face_px_large": ("FLOAT", {"default": 120.0, "min": 8.0, "max": 800.0, "step": 1.0}),
            "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.1}),
            "smooth_frames": ("INT", {"default": 9, "min": 1, "max": 61, "step": 2}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    FUNCTION = "apply"
    CATEGORY = CAT
    DESCRIPTION = (
        "Varia o denoise no tempo: rosto pequeno recebe mais regeneracao; rosto grande preserva detalhe. "
        "A mascara do audio permanece intacta para nao quebrar lipsync/audio nativo do H3."
    )

    def apply(self, av_latent, crop_data, strength_small_face=1.0,
              strength_large_face=.35, scale_mode="absolute_px",
              face_px_small=30.0, face_px_large=120.0, gamma=1.0,
              smooth_frames=9):
        samples = av_latent.get("samples") if isinstance(av_latent, dict) else None
        members = _nested_members(samples)
        if not members:
            raise ValueError("[Bruxos H3 Face] esperado LATENT AV do MiniMax H3 (NestedTensor).")
        video = members[0]
        if video.dim() != 5:
            raise ValueError(f"[Bruxos H3 Face] video latent deve ser 5D; veio {tuple(video.shape)}")
        latent_t = int(video.shape[-3])

        heights = crop_data.get("face_heights") or []
        if not heights:
            # Compatibilidade com TRACK_CROP anterior: aproxima a altura do
            # alvo pela mascara recortada e pela escala da janela.
            windows = crop_data.get("windows") or []
            heights = [float(v[3]) / 2.0 for v in windows]
        if not heights:
            raise ValueError("[Bruxos H3 Face] crop_data nao possui medidas do rosto.")
        face = torch.tensor(heights, dtype=torch.float32)

        if str(scale_mode) == "relative_to_clip":
            lo, hi = float(face.min()), float(face.max())
        else:
            lo, hi = float(face_px_small), float(face_px_large)
        if hi - lo < 1e-6:
            t = torch.zeros_like(face)
        else:
            t = ((face - lo) / (hi - lo)).clamp(0, 1)
        t = t.pow(float(gamma))
        strength = float(strength_small_face) + (
            float(strength_large_face) - float(strength_small_face)) * t
        strength = _smooth_curve(strength, smooth_frames).clamp(0, 1)

        temporal = F.interpolate(strength.view(1, 1, -1), size=latent_t,
                                 mode="linear", align_corners=True)
        temporal = temporal.view(1, 1, latent_t, 1, 1).to(video.device, torch.float32)
        vmask = temporal.expand(video.shape[0], video.shape[1], latent_t,
                                video.shape[-2], video.shape[-1]).contiguous()

        previous = av_latent.get("noise_mask") if isinstance(av_latent, dict) else None
        previous_members = _nested_members(previous)
        if previous_members:
            # Substitui SO video; audio continua exatamente como chegou.
            previous_members[0] = vmask.to(previous_members[0].dtype)
            mask_members = previous_members
            audio_note = "audio mask preservada"
        else:
            mask_members = [vmask.to(video.dtype)]
            if len(members) > 1:
                mask_members.append(torch.zeros_like(members[1]))
            audio_note = "audio mask criada em zero" if len(members) > 1 else "sem componente de audio"
        new_mask = comfy.nested_tensor.NestedTensor(tuple(mask_members))

        out = dict(av_latent)
        out["noise_mask"] = new_mask
        report = (f"face {float(face.min()):.0f}-{float(face.max()):.0f}px | "
                  f"strength {float(strength.max()):.2f}->{float(strength.min()):.2f} | "
                  f"{len(face)} frames -> {latent_t} tempos latentes | {scale_mode} | {audio_note}")
        print(f"[Bruxos H3 Face Denoise] {report}", flush=True)
        return (out, report)


NODE_CLASS_MAPPINGS = {
    "BruxosH3FacePerFrameDenoise": BruxosH3FacePerFrameDenoise,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3FacePerFrameDenoise": "H3 Face Denoise por Frame (Bruxos)",
}
