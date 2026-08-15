"""FLUX.2 Klein green-screen inpaint/outpaint para o ComfyUI nativo.

FLUX.2 Klein nao possui mask_image nativo. A LoRA
fal/flux-2-klein-4B-outpaint-lora aprende que verde puro #00FF00 e a regiao
que deve ser preenchida. Este node prepara essa referencia e injeta o latent
nos conditionings do Flux2, sem carregar uma segunda pipeline Diffusers.
"""
import math

import torch
import torch.nn.functional as F

import comfy.model_management
import node_helpers


CAT = "Bruxos do VFX/Flux2 Klein"
MODES = ["inpaint (mask branca)", "outpaint (padding)", "outpaint (tamanho alvo)"]
POSITIONS = ["center", "top-left", "top", "top-right", "left", "right",
             "bottom-left", "bottom", "bottom-right"]


def _ceil16(value):
    return max(16, int(math.ceil(max(1, int(value)) / 16.0)) * 16)


def _resize_images(images, width, height):
    if int(images.shape[2]) == width and int(images.shape[1]) == height:
        return images
    return F.interpolate(images.movedim(-1, 1), size=(height, width),
                         mode="bicubic", align_corners=False,
                         antialias=True).movedim(1, -1).clamp(0, 1)


def _resize_masks(mask, batch, width, height):
    m = mask.detach().float()
    if m.dim() == 2:
        m = m[None]
    if m.shape[0] == 1 and batch > 1:
        m = m.expand(batch, -1, -1)
    elif m.shape[0] != batch:
        ids = torch.linspace(0, m.shape[0] - 1, batch).round().long()
        m = m.index_select(0, ids)
    return F.interpolate(m[:, None], size=(height, width), mode="nearest")[:, 0]


def _offset(position, canvas_w, canvas_h, source_w, source_h):
    x = 0 if "left" in position else (canvas_w - source_w if "right" in position
                                       else (canvas_w - source_w) // 2)
    y = 0 if "top" in position else (canvas_h - source_h if "bottom" in position
                                      else (canvas_h - source_h) // 2)
    return max(0, x), max(0, y)


class BruxosFlux2KleinFill:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "image": ("IMAGE",),
            "mode": (MODES, {"default": MODES[0]}),
            "prompt": ("STRING", {"default":
                "Fill the green spaces according to the image, seamless natural result",
                "multiline": True}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
        }, "optional": {
            "mask": ("MASK", {"tooltip": "No inpaint: branco = substituir; preto = preservar."}),
            "pad_left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
            "pad_right": ("INT", {"default": 256, "min": 0, "max": 8192, "step": 16}),
            "pad_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
            "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
            "target_width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
            "target_height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
            "position": (POSITIONS, {"default": "center"}),
            "shrink_to_fit": ("BOOLEAN", {"default": True,
                "tooltip": "Reduz a imagem se ela nao couber no tamanho alvo; nunca amplia."}),
            "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "invert_mask": ("BOOLEAN", {"default": False}),
            "reference_image": ("IMAGE", {"tooltip":
                "Segunda foto: guia o conteudo que sera criado dentro da mascara branca."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "IMAGE", "MASK",
                    "INT", "INT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "green_reference", "fill_mask",
                    "width", "height", "info")
    FUNCTION = "prepare"
    CATEGORY = CAT
    DESCRIPTION = (
        "Prepara inpaint/outpaint para FLUX.2 Klein Base 4B + LoRA fal outpaint. "
        "A regiao branca da mascara vira verde puro #00FF00, a imagem e codificada "
        "como reference_latent e o latent vazio de destino ja sai no tamanho correto. "
        "Use a LoRA flux-outpaint-lora.safetensors em aproximadamente 1.1, CFG 4 e 50 steps."
    )

    def prepare(self, clip, vae, image, mode, prompt, negative_prompt,
                mask=None, pad_left=0, pad_right=256, pad_top=0, pad_bottom=0,
                target_width=1024, target_height=1024, position="center",
                shrink_to_fit=True, mask_threshold=.5, invert_mask=False,
                reference_image=None):
        src = image.detach().float().clamp(0, 1)
        batch, src_h, src_w, channels = src.shape
        if channels < 3:
            src = src[..., :1].expand(-1, -1, -1, 3)
        else:
            src = src[..., :3]

        if str(mode).startswith("inpaint"):
            if mask is None:
                raise ValueError("[Flux2 Klein Fill] conecte uma MASK para usar inpaint.")
            out_w, out_h = _ceil16(src_w), _ceil16(src_h)
            prepared = _resize_images(src, out_w, out_h)
            fill = _resize_masks(mask, batch, out_w, out_h)
            if invert_mask:
                fill = 1.0 - fill
            fill = (fill > float(mask_threshold)).float()
        else:
            if "padding" in str(mode):
                out_w = _ceil16(src_w + int(pad_left) + int(pad_right))
                out_h = _ceil16(src_h + int(pad_top) + int(pad_bottom))
                x, y = max(0, int(pad_left)), max(0, int(pad_top))
            else:
                out_w, out_h = _ceil16(target_width), _ceil16(target_height)
                if shrink_to_fit and (src_w > out_w or src_h > out_h):
                    scale = min(out_w / src_w, out_h / src_h)
                    nw, nh = max(1, round(src_w * scale)), max(1, round(src_h * scale))
                    src = _resize_images(src, nw, nh)
                    src_h, src_w = nh, nw
                x, y = _offset(position, out_w, out_h, src_w, src_h)

            # Canvas com verde EXATO. Se a imagem ultrapassar o canvas, corta
            # apenas o excedente; o usuario pode habilitar shrink_to_fit.
            prepared = torch.zeros((batch, out_h, out_w, 3), dtype=src.dtype, device=src.device)
            prepared[..., 1] = 1.0
            fill = torch.ones((batch, out_h, out_w), dtype=src.dtype, device=src.device)
            copy_w = max(0, min(src_w, out_w - x))
            copy_h = max(0, min(src_h, out_h - y))
            if copy_w and copy_h:
                prepared[:, y:y + copy_h, x:x + copy_w] = src[:, :copy_h, :copy_w]
                fill[:, y:y + copy_h, x:x + copy_w] = 0.0

        green = torch.zeros_like(prepared)
        green[..., 1] = 1.0
        prepared = prepared * (1.0 - fill[..., None]) + green * fill[..., None]
        prepared = prepared.contiguous()

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(str(prompt)))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(str(negative_prompt)))
        ref_samples = vae.encode(prepared)
        reference_latents = [ref_samples]
        reference_note = ""
        if reference_image is not None:
            guide = reference_image.detach().float().clamp(0, 1)
            if int(guide.shape[-1]) < 3:
                guide = guide[..., :1].expand(-1, -1, -1, 3)
            else:
                guide = guide[..., :3]
            # A segunda referencia nao precisa ter a mesma proporcao da base.
            # Apenas alinhamos cada lado ao grid de 16 exigido pelo VAE Flux2.
            guide_h, guide_w = int(guide.shape[1]), int(guide.shape[2])
            guide = _resize_images(guide, _ceil16(guide_w), _ceil16(guide_h))
            reference_latents.append(vae.encode(guide.contiguous()))
            reference_note = " | image2 ligada como guia do preenchimento"
        pos = node_helpers.conditioning_set_values(
            positive, {"reference_latents": reference_latents}, append=True)
        neg = node_helpers.conditioning_set_values(
            negative, {"reference_latents": reference_latents}, append=True)
        latent = {"samples": torch.zeros(
            [batch, 128, out_h // 16, out_w // 16],
            device=comfy.model_management.intermediate_device())}
        info = (f"{str(mode).split(' (')[0]} | {src_w}x{src_h} -> {out_w}x{out_h} | "
                "verde #00FF00 = preencher" + reference_note +
                " | recomendado: Klein Base 4B, LoRA 1.1, CFG 4, 50 steps")
        return (pos, neg, latent, prepared, fill.contiguous(),
                int(out_w), int(out_h), info)


NODE_CLASS_MAPPINGS = {"BruxosFlux2KleinFill": BruxosFlux2KleinFill}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosFlux2KleinFill": "Flux2 Klein Inpaint + Outpaint (Bruxos)"
}
