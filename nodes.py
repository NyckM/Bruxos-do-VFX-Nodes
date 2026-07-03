import copy
import math

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
from comfy_extras.nodes_custom_sampler import BasicScheduler, KSamplerSelect, SamplerCustom, SplitSigmas


def _clone_conditioning_set_values(conditioning, values):
    updated = []
    for item in conditioning:
        if len(item) != 2:
            updated.append(copy.deepcopy(item))
            continue
        cond, metadata = item
        metadata = copy.deepcopy(metadata)
        metadata.update(values)
        updated.append([cond, metadata])
    return updated


def _resize_long_edge(image, max_size, stride=16):
    h, w = image.shape[1], image.shape[2]
    scale = min(max_size / max(h, w), 1.0)
    nh = max(stride, round(h * scale / stride) * stride)
    nw = max(stride, round(w * scale / stride) * stride)
    return comfy.utils.common_upscale(
        image[:, :, :, :3].movedim(-1, 1), nw, nh, "area", "disabled"
    ).movedim(1, -1)


# Modo de encaixe quando o aspect ratio do video difere do width/height pedido:
#   stretch -> "disabled": estica (sem cortar; pode distorcer um pouco)
#   crop    -> "center":   corta as bordas pra preservar o aspect ratio
_FIT_CROP = {"stretch": "disabled", "crop": "center"}


def _resize_source_video(video, width, height, mode="stretch"):
    crop = _FIT_CROP.get(mode, "disabled")
    return comfy.utils.common_upscale(
        video[:, :, :, :3].movedim(-1, 1), width, height, "area", crop
    ).movedim(1, -1)


def _video_frame_count(video):
    if video is None or not hasattr(video, "shape") or len(video.shape) < 1:
        raise ValueError("source_video must be a ComfyUI IMAGE/video tensor with frames on dimension 0.")
    return int(video.shape[0])


def _split_video(video, chunk_size, overlap):
    frame_count = _video_frame_count(video)
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    step = chunk_size - overlap
    chunks = []
    ranges = []
    for start in range(0, frame_count, step):
        end = min(start + chunk_size, frame_count)
        if end <= start:
            continue
        chunks.append(video[start:end])
        ranges.append((start, end))
        if end == frame_count:
            break
    return chunks, ranges


def _latent_shape(frame_count, width, height, batch_size):
    return [batch_size, 16, ((frame_count - 1) // 4) + 1, height // 8, width // 8]


def _make_empty_latent(frame_count, width, height, batch_size):
    return torch.zeros(
        _latent_shape(frame_count, width, height, batch_size),
        device=comfy.model_management.intermediate_device(),
    )


def _encode_video(vae, video):
    encoded = vae.encode(video)
    if isinstance(encoded, dict) and "samples" in encoded:
        return encoded["samples"]
    return encoded


# ----------------------------------------------------------------------
# ALINHAMENTO TEMPORAL (4n+1) -- por que o video "perde" frames:
#   o Wan VAE comprime o tempo em ~4x. N frames viram T_lat = ((N-1)//4)+1
#   latentes, e o decode devolve (T_lat-1)*4 + 1 frames. So sobrevivem
#   comprimentos da forma 4n+1 (1,5,9,...,109,113,...). 111 -> 28 latentes
#   -> 109 frames. A correcao: por dentro trabalhamos no proximo 4n+1
#   (padding espelhado) e no final cortamos de volta ao alvo do usuario.
# ----------------------------------------------------------------------
def _lat_len(frames):
    """Numero de frames latentes para `frames` frames de pixel (compressao 4x)."""
    return ((int(frames) - 1) // 4) + 1


def _align_up_4n1(n):
    """Proximo comprimento valido para o grid temporal do Wan VAE (4n+1)."""
    n = int(n)
    if n < 1:
        return 1
    r = (n - 1) % 4
    return n if r == 0 else n + (4 - r)


def _mirror_pad_frames(video, target_len):
    """Estende o video no eixo temporal (dim 0) ate target_len por reflexao
    (espelho ping-pong, igual ao truque do Kijai no Wan Animate), evitando o
    frame congelado que a simples duplicacao do ultimo frame causaria."""
    cur = int(video.shape[0])
    target_len = int(target_len)
    if cur >= target_len:
        return video
    need = target_len - cur
    if cur == 1:
        pad = video[-1:].repeat(need, *([1] * (video.dim() - 1)))
        return torch.cat([video, pad], dim=0)
    idx = []
    i = cur - 2          # comeca refletindo a partir do penultimo
    direction = -1
    while len(idx) < need:
        idx.append(i)
        i += direction
        if i < 0:                 # bate na borda inicial e volta
            i = 1
            direction = 1
        elif i > cur - 1:         # bate na borda final e volta
            i = cur - 2
            direction = -1
    pad = video[idx]
    return torch.cat([video, pad], dim=0)


# ----------------------------------------------------------------------
# MASCARA (gerar so na area selecionada / otimizar por bbox)
#   - aceita MASK [T,H,W] / [H,W] ou um IMAGE colorido [T,H,W,C] (estilo
#     SCAIL2ColoredMask). Para colorido, qualquer pixel != preto = regiao.
# ----------------------------------------------------------------------
def _normalize_mask(mask):
    if mask is None:
        return None
    m = mask
    if m.dim() == 4:                       # IMAGE [T,H,W,C] (mascara colorida)
        m = m[..., :3].amax(dim=-1)        # qualquer canal aceso = dentro
    elif m.dim() == 2:                     # [H,W]
        m = m.unsqueeze(0)
    return m.float().clamp(0.0, 1.0)


def _grow_blur_mask(m, grow=0, blur=0):
    """m: [T,H,W] -> [T,H,W]. grow>0 dilata, grow<0 contrai, blur suaviza a borda."""
    x = m.unsqueeze(1)                      # [T,1,H,W]
    grow = int(grow)
    if grow > 0:
        k = grow * 2 + 1
        x = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=1, padding=grow)
    elif grow < 0:
        g = -grow
        k = g * 2 + 1
        x = -torch.nn.functional.max_pool2d(-x, kernel_size=k, stride=1, padding=g)
    blur = int(blur)
    if blur > 0:
        k = blur * 2 + 1
        coords = torch.arange(k, dtype=torch.float32, device=m.device) - blur
        sigma = blur * 0.5 + 1e-6
        g1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        g1d = (g1d / g1d.sum())
        kh = g1d.view(1, 1, k, 1)
        kw = g1d.view(1, 1, 1, k)
        x = torch.nn.functional.conv2d(x, kh, padding=(blur, 0))
        x = torch.nn.functional.conv2d(x, kw, padding=(0, blur))
    return x.squeeze(1).clamp(0.0, 1.0)


def _resize_mask_spatial_temporal(m, T, H, W, mode="stretch"):
    """m: [Tm,Hm,Wm] -> [T,H,W]. Usa o MESMO encaixe (stretch/crop) do video,
    pra mascara e fonte ficarem alinhadas. Tempo por amostragem nearest."""
    Tm = int(m.shape[0])
    crop = _FIT_CROP.get(mode, "disabled")
    m = comfy.utils.common_upscale(
        m.unsqueeze(1), int(W), int(H), "bilinear", crop
    ).squeeze(1)                              # [Tm,H,W]
    if Tm != int(T):
        idx = torch.linspace(0, Tm - 1, steps=int(T)).round().long().clamp(0, Tm - 1)
        m = m[idx]
    return m


def _mask_to_latent(m, T_lat, lat_h, lat_w, device, dtype):
    """Reduz a mascara de pixel [Tpix,H,W] para o grid latente
    [1,1,T_lat,lat_h,lat_w]. No tempo agrupa em blocos de 4 (1 + 4k) usando o
    maximo (qualquer frame do bloco dentro => latente dentro)."""
    m = m.to(device=device, dtype=torch.float32)
    m = torch.nn.functional.interpolate(
        m.unsqueeze(1), size=(int(lat_h), int(lat_w)), mode="bilinear", align_corners=False
    ).squeeze(1)                             # [Tpix,lat_h,lat_w]
    Tpix = int(m.shape[0])
    out = torch.zeros(int(T_lat), int(lat_h), int(lat_w), device=device, dtype=torch.float32)
    out[0] = m[0]
    for li in range(1, int(T_lat)):
        a = 1 + (li - 1) * 4
        b = min(Tpix, a + 4)
        out[li] = m[a:b].amax(dim=0) if a < Tpix else m[-1]
    return out.view(1, 1, int(T_lat), int(lat_h), int(lat_w)).to(dtype=dtype)


def _mask_bbox(m, pad, stride, W, H, thr=0.02):
    """bbox (x0,y0,x1,y1) cobrindo a regiao em TODOS os frames, com folga `pad`
    e alinhada a `stride` (multiplo exigido por largura/altura)."""
    any2d = (m.amax(dim=0) > thr)
    rows = torch.where(any2d.any(dim=1))[0]
    cols = torch.where(any2d.any(dim=0))[0]
    if rows.numel() == 0 or cols.numel() == 0:
        return 0, 0, int(W), int(H)
    y0 = int(rows.min()); y1 = int(rows.max()) + 1
    x0 = int(cols.min()); x1 = int(cols.max()) + 1
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(int(W), x1 + pad); y1 = min(int(H), y1 + pad)
    x0 -= x0 % stride
    y0 -= y0 % stride
    if x1 % stride:
        x1 = min(int(W), x1 + (stride - x1 % stride))
    if y1 % stride:
        y1 = min(int(H), y1 + (stride - y1 % stride))
    if x1 - x0 < stride:
        x1 = min(int(W), x0 + stride)
    if y1 - y0 < stride:
        y1 = min(int(H), y0 + stride)
    return x0, y0, x1, y1


def _collect_reference_latents(vae, length, ref_max_size, reference_video=None, reference_images=None):
    latents = []
    if reference_video is not None:
        ref_vid = _resize_long_edge(reference_video[:length], ref_max_size)
        latents.append(_encode_video(vae, ref_vid[:, :, :, :3]))

    if reference_images:
        for name in sorted(reference_images):
            imgs = reference_images[name]
            if imgs is None:
                continue
            for i in range(imgs.shape[0]):
                img = _resize_long_edge(imgs[i:i + 1], ref_max_size)
                latents.append(_encode_video(vae, img[:, :, :, :3]))
    return latents


def _merge_linear_overlap(first, second, overlap):
    if overlap <= 0:
        return torch.cat([first, second], dim=0)
    if first.shape[0] < overlap or second.shape[0] < overlap:
        overlap = min(int(first.shape[0]), int(second.shape[0]), overlap)
    if overlap <= 0:
        return torch.cat([first, second], dim=0)

    left = first[:-overlap]
    right = second[overlap:]
    first_tail = first[-overlap:]
    second_head = second[:overlap]
    weights = torch.linspace(0.0, 1.0, overlap, dtype=first.dtype, device=first.device)
    while weights.ndim < first_tail.ndim:
        weights = weights.unsqueeze(-1)
    blended = first_tail * (1.0 - weights) + second_head * weights
    return torch.cat([left, blended, right], dim=0)


def _merge_latent_overlap(first, second, overlap):
    # Latente de vídeo no formato [B, C, T, H, W]; o eixo temporal é a dim 2.
    if overlap <= 0:
        return torch.cat([first, second], dim=2)
    overlap = min(int(first.shape[2]), int(second.shape[2]), int(overlap))
    if overlap <= 0:
        return torch.cat([first, second], dim=2)

    left = first[:, :, :-overlap]
    right = second[:, :, overlap:]
    first_tail = first[:, :, -overlap:]
    second_head = second[:, :, :overlap]
    weights = torch.linspace(0.0, 1.0, overlap, dtype=first.dtype, device=first.device)
    weights = weights.view(1, 1, overlap, 1, 1)
    blended = first_tail * (1.0 - weights) + second_head * weights
    return torch.cat([left, blended, right], dim=2)


class BerniniLongConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "source_video": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "chunk_size": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "overlap": ("INT", {"default": 5, "min": 0, "max": 512, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "tail_memory": ("BOOLEAN", {"default": True}),
                "tail_frames": ("INT", {"default": 5, "min": 1, "max": 128, "step": 1}),
                "ref_max_size": ("INT", {"default": 848, "min": 16, "max": 8192, "step": 16}),
            },
            "optional": {
                "reference_video": ("IMAGE",),
                "reference_images.reference_image_0": ("IMAGE",),
                "reference_images.reference_image_1": ("IMAGE",),
                "reference_images.reference_image_2": ("IMAGE",),
                "reference_images.reference_image_3": ("IMAGE",),
                "reference_images.reference_image_4": ("IMAGE",),
                "reference_images.reference_image_5": ("IMAGE",),
                "reference_images.reference_image_6": ("IMAGE",),
                "reference_images.reference_image_7": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "BERNINI_POSITIVE_CHUNKS",
        "BERNINI_NEGATIVE_CHUNKS",
        "BERNINI_LATENT_CHUNKS",
        "BERNINI_VIDEO_CHUNKS",
        "BERNINI_CHUNK_RANGES",
        "INT",
    )
    RETURN_NAMES = (
        "positive_chunks",
        "negative_chunks",
        "latent_chunks",
        "video_chunks",
        "chunk_ranges",
        "chunk_count",
    )
    FUNCTION = "build"
    CATEGORY = "Bruxos do VFX/Bernini"

    def build(
        self,
        positive,
        negative,
        vae,
        source_video,
        width,
        height,
        chunk_size=81,
        overlap=5,
        batch_size=1,
        tail_memory=True,
        tail_frames=5,
        ref_max_size=848,
        reference_video=None,
        **kwargs,
    ):
        chunks, ranges = _split_video(source_video, int(chunk_size), int(overlap))
        reference_images = {
            key: value
            for key, value in kwargs.items()
            if key.startswith("reference_images.reference_image_") and value is not None
        }

        positive_chunks = []
        negative_chunks = []
        latent_chunks = []
        previous_chunk = None

        for chunk in chunks:
            source_chunk = _resize_source_video(chunk, int(width), int(height))
            encoded_chunk = _encode_video(vae, source_chunk)
            context_latents = [encoded_chunk]
            context_latents.extend(
                _collect_reference_latents(
                    vae,
                    int(chunk.shape[0]),
                    int(ref_max_size),
                    reference_video=reference_video,
                    reference_images=reference_images,
                )
            )

            if tail_memory and previous_chunk is not None:

                tail_count = min(
                    int(tail_frames),
                    int(previous_chunk.shape[0])
                )

                if tail_count > 0:

                    tail = previous_chunk[-tail_count:]

                    tail_latent = _encode_video(
                        vae,
                        tail[:, :, :, :3]
                    )

                    context_latents.append(tail_latent)

                    print(
                        f"[Bernini Infinity] injected {tail_count} generated frames as tail memory",
                        flush=True,
                    )

            values = {"context_latents": context_latents}
            positive_chunks.append(_clone_conditioning_set_values(positive, values))
            negative_chunks.append(_clone_conditioning_set_values(negative, values))
            latent_chunks.append(
                {"samples": _make_empty_latent(int(chunk.shape[0]), int(width), int(height), int(batch_size))}
            )
            previous_chunk = source_chunk

        return (positive_chunks, negative_chunks, latent_chunks, chunks, ranges, len(chunks))


class BerniniLongChunkSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_chunks": ("BERNINI_POSITIVE_CHUNKS",),
                "negative_chunks": ("BERNINI_NEGATIVE_CHUNKS",),
                "latent_chunks": ("BERNINI_LATENT_CHUNKS",),
                "index": ("INT", {"default": 0, "min": 0, "max": 4095, "step": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "select"
    CATEGORY = "Bruxos do VFX/Bernini"

    def select(self, positive_chunks, negative_chunks, latent_chunks, index):
        count = len(positive_chunks)
        if count == 0:
            raise ValueError("No chunks were generated.")
        index = max(0, min(int(index), count - 1))
        return positive_chunks[index], negative_chunks[index], latent_chunks[index]


class BerniniLongVideoMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_chunks": ("BERNINI_VIDEO_CHUNKS",),
                "overlap": ("INT", {"default": 5, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {"extra_video": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    FUNCTION = "merge"
    CATEGORY = "Bruxos do VFX/Bernini"

    def merge(self, video_chunks, overlap=5, extra_video=None):
        chunks = list(video_chunks)
        if extra_video is not None:
            chunks.append(extra_video)
        if not chunks:
            raise ValueError("No video chunks to merge.")
        final = chunks[0]
        for chunk in chunks[1:]:
            final = _merge_linear_overlap(final, chunk, int(overlap))
        return (final,)


class BerniniLongAppendVideoChunk:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_chunks": ("BERNINI_VIDEO_CHUNKS",), "video": ("IMAGE",)}}

    RETURN_TYPES = ("BERNINI_VIDEO_CHUNKS",)
    RETURN_NAMES = ("video_chunks",)
    FUNCTION = "append"
    CATEGORY = "Bruxos do VFX/Bernini"

    def append(self, video_chunks, video):
        chunks = list(video_chunks)
        chunks.append(video)
        return (chunks,)


class BerniniLongEmptyVideoChunks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("BERNINI_VIDEO_CHUNKS",)
    RETURN_NAMES = ("video_chunks",)
    FUNCTION = "empty"
    CATEGORY = "Bruxos do VFX/Bernini"

    def empty(self):
        return ([],)


class BerniniLongInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"chunk_ranges": ("BERNINI_CHUNK_RANGES",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "info"
    CATEGORY = "Bruxos do VFX/Bernini"

    def info(self, chunk_ranges):
        lines = []
        for index, (start, end) in enumerate(chunk_ranges):
            lines.append(f"{index}: frames {start}..{end - 1} ({end - start} frames)")
        return ("\n".join(lines),)



def _decode_video(vae, latent_samples, tiled=False):
    if tiled:
        # Passa tile/overlap explicitos: em algumas versoes do ComfyUI o
        # decode_tiled 3D deixa overlap=None num eixo e quebra em "tile - overlap".
        try:
            images = vae.decode_tiled(
                latent_samples,
                tile_x=256, tile_y=256, overlap=64,
                tile_t=32, overlap_t=8,
            )
        except TypeError:
            # assinaturas mais antigas nao aceitam tile_t/overlap_t
            try:
                images = vae.decode_tiled(latent_samples, tile_x=256, tile_y=256, overlap=64)
            except Exception:
                images = vae.decode(latent_samples)
        except Exception:
            images = vae.decode(latent_samples)
    else:
        images = vae.decode(latent_samples)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _decode_video_chunked(vae, latent_samples, tiled=False, lat_chunk=0, lat_overlap=1):
    """Decodifica o latente em blocos temporais (eixo dim 2) para evitar pico de VRAM
    no VAE em videos longos. lat_chunk<=0 (ou >= T) decodifica tudo de uma vez."""
    total = int(latent_samples.shape[2])
    if lat_chunk is None or int(lat_chunk) <= 0 or total <= int(lat_chunk):
        return _decode_video(vae, latent_samples, tiled)
    lat_chunk = int(lat_chunk)
    stride = max(1, lat_chunk - int(lat_overlap))
    pix_overlap = max(0, int(lat_overlap)) * 4  # fator de compressao temporal ~4 do VAE Wan
    result = None
    start = 0
    while start < total:
        end = min(start + lat_chunk, total)
        sub = latent_samples[:, :, start:end]
        imgs = _decode_video(vae, sub, tiled).cpu()
        result = imgs if result is None else _merge_linear_overlap(result, imgs, pix_overlap)
        if end == total:
            break
        start += stride
    return result


def _ordered_offset(idx, stride):
    """Espalha offsets quase uniformemente em [0, stride) conforme os passos avancam."""
    if stride <= 1:
        return 0
    bits = max(1, int(math.ceil(math.log2(stride))))
    r = 0
    for b in range(bits):
        r = (r << 1) | ((idx >> b) & 1)
    return int(r) % stride


def _context_windows(total, win, overlap, offset=0):
    """Janelas (start, end) em frames LATENTES. As pontas reais (0 e total-win) ficam
    SEMPRE ancoradas (video ABERTO, sem wrap end->start); so as fronteiras internas
    deslizam com `offset` para nao travar nos mesmos frames a cada passo de denoise."""
    if win <= 0 or total <= win:
        return [(0, total)]
    stride = max(1, win - overlap)
    offset = int(offset) % stride
    starts = {0, total - win}
    s = offset
    while s < total - win:
        if s > 0:
            starts.add(s)
        s += stride
    return [(p, p + win) for p in sorted(starts)]


def _window_blend_weights(length, ramp, device, dtype):
    """Pesos de blend (rampa Hann nas bordas) ao longo do eixo temporal latente."""
    w = torch.ones(length, device=device, dtype=dtype)
    ramp = int(min(max(0, ramp), length // 2))
    if ramp > 0:
        t = torch.linspace(0.0, math.pi, steps=ramp + 2, device=device, dtype=dtype)[1:-1]
        edge = (1.0 - torch.cos(t)) * 0.5
        w[:ramp] = edge
        w[-ramp:] = torch.flip(edge, dims=[0])
    return w


def _slice_temporal(obj, s, e, total):
    """Fatia recursivamente qualquer tensor cujo eixo temporal (dim 2 em [B,C,T,H,W])
    tenha comprimento == total. O que nao for temporal passa intacto (refs, texto)."""
    if torch.is_tensor(obj):
        if obj.dim() >= 5 and obj.shape[2] == total:
            return obj[:, :, s:e]
        return obj
    if isinstance(obj, dict):
        return {k: _slice_temporal(v, s, e, total) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_slice_temporal(v, s, e, total) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_slice_temporal(v, s, e, total) for v in obj)
    return obj


def _debug_dump_shapes(obj, total, prefix="c", depth=0, acc=None):
    if acc is None:
        acc = []
    if depth > 3:
        return acc
    if torch.is_tensor(obj):
        tag = " <== T_lat" if (obj.dim() >= 5 and obj.shape[2] == total) else ""
        acc.append(f"{prefix}={tuple(obj.shape)}{tag}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _debug_dump_shapes(v, total, f"{prefix}.{k}", depth + 1, acc)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _debug_dump_shapes(v, total, f"{prefix}[{i}]", depth + 1, acc)
    return acc


def _make_context_wrapper(win_len, win_overlap, ramp, jitter=True, debug_holder=None):
    """model_function_wrapper estilo WanVideoWrapper: divide o latente completo em
    janelas sobrepostas a cada passo, roda o modelo em cada uma e compoe as predicoes
    com blend. Com jitter, as fronteiras internas deslizam por passo."""
    state = {"calls": 0}
    stride = max(1, win_len - win_overlap)

    def wrapper(model_function, params):
        x = params["input"]
        t = params["timestep"]
        c = params["c"]
        total = int(x.shape[2])

        offset = _ordered_offset(state["calls"], stride) if jitter else 0
        state["calls"] += 1
        windows = _context_windows(total, win_len, win_overlap, offset)

        if debug_holder is not None and not debug_holder.get("printed"):
            debug_holder["printed"] = True
            try:
                print(f"[Bernini Infinity][ctx] x={tuple(x.shape)} T_lat={total} offset={offset} janelas={windows}", flush=True)
                for line in _debug_dump_shapes(c, total):
                    print(f"[Bernini Infinity][ctx]   {line}", flush=True)
            except Exception:
                pass

        if len(windows) <= 1:
            return model_function(x, t, **c)

        out = torch.zeros_like(x)
        counter = torch.zeros((1, 1, total, 1, 1), device=x.device, dtype=x.dtype)
        for (s, e) in windows:
            xw = x[:, :, s:e]
            cw = _slice_temporal(c, s, e, total)
            ow = model_function(xw, t, **cw)
            wts = _window_blend_weights(e - s, ramp, x.device, x.dtype).view(1, 1, e - s, 1, 1)
            out[:, :, s:e] += ow * wts
            counter[:, :, s:e] += wts
        return out / counter.clamp(min=1e-6)

    return wrapper


class BerniniInfinity:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "Condicionamento POSITIVO (prompt do que VOCE QUER). Ligue aqui o CLIP Text Encode positivo."}),
                "negative": ("CONDITIONING", {"tooltip": "Condicionamento NEGATIVO (o que voce NAO quer). Ligue aqui o CLIP Text Encode negativo."}),
                "high_model": ("MODEL", {"tooltip": "Modelo de ALTO ruido (high noise) do Bernini/Wan 2.x. Roda nos primeiros passos (ate split_step). E o que define o movimento/estrutura."}),
                "low_model": ("MODEL", {"tooltip": "Modelo de BAIXO ruido (low noise). Roda nos passos finais (a partir de split_step). E o que refina detalhe/textura."}),
                "vae": ("VAE", {"tooltip": "VAE do Wan (ex.: wan_2.1_vae). Comprime o video para latente e decodifica de volta. E aqui que mora a compressao temporal 4x (1 latente = 4 frames)."}),
                "source_video": ("IMAGE", {"tooltip": "Video de entrada (sequencia de frames). E ele que sera editado/animado. O numero de frames daqui define o alvo de saida."}),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16, "tooltip": "Largura de saida (multiplo de 16). Se a proporcao diferir do video, veja 'resize_mode'. Maior = mais VRAM e mais lento."}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16, "tooltip": "Altura de saida (multiplo de 16). Combine com 'width' na mesma proporcao do video pra nao distorcer nem cortar."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Semente do ruido. Mesma seed + mesmas entradas = mesmo resultado. Mude pra gerar variacoes."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "Total de passos de denoise. Com LoRAs de poucos passos (lightx2v etc.) 4-8 ja basta. Mais passos = mais lento, nem sempre melhor."}),
                "split_step": ("INT", {"default": 4, "min": 0, "max": 10000, "tooltip": "Em qual passo troca do high_model para o low_model. Ex.: steps=6, split_step=4 => 4 passos no high + 2 no low. 0 = so usa o low."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "Forca do prompt (classifier-free guidance). Em modelos destilados/turbo costuma ficar em 1.0. Valores altos podem 'fritar' a imagem."}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"tooltip": "Algoritmo de amostragem (ex.: euler). Se nao tiver certeza, 'euler' e uma escolha segura."}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"tooltip": "Como os niveis de ruido sao distribuidos pelos passos (ex.: simple). Afeta sutilmente o resultado."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Quanto recria do zero. 1.0 = geracao total. Menor preserva mais o video original (edicao leve)."}),
                "chunk_size": ("INT", {"default": 81, "min": 5, "max": 8192, "step": 4, "tooltip": "Tamanho da JANELA em frames. context_window: tamanho da janela de atencao (81 e o padrao do Wan). sequential: tamanho de cada bloco. Use 4n+1 (5,9,...,81,121)."}),
                "overlap": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1, "tooltip": "Sobreposicao em frames entre janelas/blocos. Mais overlap = transicao mais suave entre janelas, porem mais lento. ~16 e um bom meio-termo."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, "tooltip": "Limita quantos frames processar (0 = todos do video). Util pra testar rapido so o comeco."}),
                "tail_memory": ("BOOLEAN", {"default": True, "tooltip": "[Modo sequential] Propaga os ultimos frames JA EDITADOS do bloco anterior pro proximo. Ajuda a manter identidade/roupa/luz entre os blocos. Nao tem efeito no context_window."}),
                "tail_frames": ("INT", {"default": 5, "min": 1, "max": 128, "step": 1, "tooltip": "[Modo sequential] Quantos frames editados do bloco anterior reaproveitar como memoria. Mais = mais coerencia, porem menos liberdade pra mudar."}),
                "decode_tiled": ("BOOLEAN", {"default": False, "tooltip": "Decodifica o VAE em ladrilhos (tiles). Liga so se estourar VRAM no decode; e mais lento e pode deixar leve costura."}),
                "decode_chunk": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, "tooltip": "Decodifica o video em blocos temporais de latente pra evitar OOM no VAE em videos longos. 0 = decodifica tudo de uma vez. Ex.: 16 decodifica de 16 em 16 latentes."}),
                "vary_seed_per_chunk": ("BOOLEAN", {"default": False, "tooltip": "[Modo sequential] Soma o indice do bloco a seed, dando uma seed diferente por bloco. Off mantem a mesma seed em todos (mais consistente)."}),
                "ref_max_size": ("INT", {"default": 848, "min": 16, "max": 8192, "step": 16, "tooltip": "Lado maior (em px) pra redimensionar as imagens/video de referencia antes de virar latente de contexto. Maior = referencia mais detalhada, mais VRAM."}),
                "mode": (["sequential", "context_window"], {"default": "sequential", "tooltip": "context_window: gera o video inteiro de uma vez com janelas deslizantes de atencao (mais coerente, recomendado). sequential: gera bloco a bloco com tail_memory (bom pra videos muito longos / pouca VRAM)."}),
                "context_jitter": ("BOOLEAN", {"default": True, "tooltip": "[Modo context_window] Desliza as fronteiras das janelas a cada passo pra nao 'marcar' sempre nos mesmos frames. Geralmente deixe ligado."}),
                "mask_mode": (["off", "inpaint", "bbox"], {"default": "off", "tooltip": "Geracao por mascara (requer region_mask). off = mascara desligada. inpaint = gera o frame todo, mas SO a area da mascara muda (resto = fonte). bbox = recorta na caixa da mascara e gera so esse retangulo em resolucao menor (mais rapido / menos VRAM) e cola de volta."}),
                "mask_grow": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1, "tooltip": "Dilata (+) ou contrai (-) a mascara, em pixels. Use + pra pegar uma margem ao redor do objeto; - pra apertar a regiao."}),
                "mask_blur": ("INT", {"default": 6, "min": 0, "max": 256, "step": 1, "tooltip": "Suaviza a borda da mascara (feather), em pixels. Evita emenda dura entre a area gerada e a fonte. 0 = borda seca."}),
                "mask_pad": ("INT", {"default": 16, "min": 0, "max": 1024, "step": 16, "tooltip": "[mask_mode=bbox] Folga em pixels ao redor da caixa da mascara antes de recortar. Mais folga = menos risco de cortar o objeto, porem recorte maior (mais lento)."}),
                "resize_mode": (["stretch", "crop"], {"default": "stretch", "tooltip": "Encaixe quando a proporcao do video difere de width/height. stretch = estica pro tamanho exato, SEM cortar (pode distorcer um pouco). crop = corta as bordas mantendo a proporcao."}),
            },
            "optional": {
                "region_mask": ("MASK,IMAGE", {"tooltip": "Mascara da regiao a gerar (branco/colorido = gera, preto = mantem a fonte). Aceita MASK ou IMAGE colorida -- pode ligar direto o pose_video_mask/reference_image_mask do 'Create SCAIL-2 Colored Mask'. Use com mask_mode = inpaint ou bbox."}),
                "reference_video": ("IMAGE", {"tooltip": "Video de referencia opcional (ex.: clean plate / identidade). Vira latente de contexto extra pra guiar a geracao. Nao precisa ter o mesmo tamanho do source."}),
                "reference_images.reference_image_0": ("IMAGE", {"tooltip": "Imagem de referencia 0 (ex.: rosto/objeto a preservar). Vira contexto extra. Pode deixar vazio."}),
                "reference_images.reference_image_1": ("IMAGE", {"tooltip": "Imagem de referencia 1 (opcional)."}),
                "reference_images.reference_image_2": ("IMAGE", {"tooltip": "Imagem de referencia 2 (opcional)."}),
                "reference_images.reference_image_3": ("IMAGE", {"tooltip": "Imagem de referencia 3 (opcional)."}),
                "reference_images.reference_image_4": ("IMAGE", {"tooltip": "Imagem de referencia 4 (opcional)."}),
                "reference_images.reference_image_5": ("IMAGE", {"tooltip": "Imagem de referencia 5 (opcional)."}),
                "reference_images.reference_image_6": ("IMAGE", {"tooltip": "Imagem de referencia 6 (opcional)."}),
                "reference_images.reference_image_7": ("IMAGE", {"tooltip": "Imagem de referencia 7 (opcional)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "INT")
    RETURN_NAMES = ("images", "latent", "total_frames")
    OUTPUT_TOOLTIPS = (
        "Video gerado (frames). Sai com EXATAMENTE o numero de frames do source (corrige a perda do 4n+1 do VAE).",
        "Latente do resultado (caso queira reaproveitar/decodificar de novo).",
        "Quantidade de frames de saida (= frames do source, ja descontado o padding interno).",
    )
    DESCRIPTION = (
        "Bernini Infinity (Bruxos do VFX): gera/edita video com Bernini/Wan acima do limite de 81 frames, "
        "sem perder frames no fim (alinhamento 4n+1 automatico com padding espelhado) e com mascara opcional "
        "pra gerar so na area selecionada. Passe o mouse sobre cada campo pra ver o que faz."
    )
    FUNCTION = "render"
    CATEGORY = "Bruxos do VFX/Bernini"

    def render(
        self,
        positive,
        negative,
        high_model,
        low_model,
        vae,
        source_video,
        width,
        height,
        seed,
        steps,
        split_step,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        mode,
        chunk_size,
        overlap,
        max_frames,
        tail_memory,
        tail_frames,
        decode_tiled,
        decode_chunk,
        context_jitter,
        vary_seed_per_chunk,
        ref_max_size,
        mask_mode="off",
        mask_grow=0,
        mask_blur=6,
        mask_pad=16,
        resize_mode="stretch",
        region_mask=None,
        reference_video=None,
        **kwargs,
    ):
        frame_count = int(source_video.shape[0])
        target = frame_count if int(max_frames) <= 0 else min(frame_count, int(max_frames))
        chunk_size = int(chunk_size)
        overlap = int(overlap)
        if overlap >= chunk_size:
            raise ValueError("overlap deve ser menor que chunk_size")

        sampler = KSamplerSelect.execute(sampler_name).args[0]
        sigmas = BasicScheduler.execute(low_model, scheduler, int(steps), float(denoise)).args[0]
        high_sigmas, low_sigmas = SplitSigmas.execute(sigmas, int(split_step)).args

        reference_images = {
            key: value
            for key, value in kwargs.items()
            if key.startswith("reference_images.reference_image_") and value is not None
        }

        if mode == "context_window":
            return self._render_context_window(
                positive, negative, high_model, low_model, vae, source_video,
                width, height, seed, sampler, high_sigmas, low_sigmas, cfg,
                chunk_size, overlap, target, decode_tiled, decode_chunk,
                ref_max_size, reference_video, reference_images, context_jitter,
                region_mask=region_mask, mask_mode=mask_mode,
                mask_grow=mask_grow, mask_blur=mask_blur, mask_pad=mask_pad,
                resize_mode=resize_mode,
            )
        return self._render_sequential(
            positive, negative, high_model, low_model, vae, source_video,
            width, height, seed, sampler, high_sigmas, low_sigmas, cfg,
            chunk_size, overlap, target, tail_memory, tail_frames,
            decode_tiled, decode_chunk, vary_seed_per_chunk, ref_max_size,
            reference_video, reference_images,
            region_mask=region_mask, mask_mode=mask_mode,
            mask_grow=mask_grow, mask_blur=mask_blur,
            resize_mode=resize_mode,
        )

    # ------------------------------------------------------------------
    # MODO SEQUENTIAL: edicao por PROPAGACAO
    #   - cada chunk e condicionado em (fonte local) + (referencias/clean plate)
    #     + (tail_memory: frames JA EDITADOS do chunk anterior).
    #   - juncao por CROSSFADE de latente (sem corte seco).
    # ------------------------------------------------------------------
    def _render_sequential(
        self, positive, negative, high_model, low_model, vae, source_video,
        width, height, seed, sampler, high_sigmas, low_sigmas, cfg,
        chunk_size, overlap, target, tail_memory, tail_frames,
        decode_tiled, decode_chunk, vary_seed_per_chunk, ref_max_size,
        reference_video, reference_images,
        region_mask=None, mask_mode="off", mask_grow=0, mask_blur=0,
        resize_mode="stretch",
    ):
        step = max(1, chunk_size - overlap)
        lat_drop = ((overlap - 1) // 4) + 1 if overlap > 0 else 0
        total_chunks = 1 + max(0, target - chunk_size + step - 1) // step
        print(
            f"[Bernini Infinity][seq] target={target} chunk={chunk_size} overlap={overlap} "
            f"step={step} -> {total_chunks} chunk(s) | tail_memory={tail_memory}",
            flush=True,
        )

        # mascara global (em pixels, no tamanho de saida); fatiada por chunk depois.
        use_mask = (mask_mode != "off") and (region_mask is not None)
        seq_mask = None
        if use_mask:
            seq_mask = _normalize_mask(region_mask)
            seq_mask = _resize_mask_spatial_temporal(seq_mask, int(target), int(height), int(width), resize_mode)
            seq_mask = _grow_blur_mask(seq_mask, int(mask_grow), int(mask_blur))
            if mask_mode == "bbox":
                print("[Bernini Infinity][seq] bbox nao se aplica ao modo sequential; "
                      "usando inpaint.", flush=True)

        stitched_imgs = None
        stitched_latents = None
        previous_generated_frames = None
        chunk_index = 0

        for start in range(0, target, step):
            end = min(start + chunk_size, target)
            raw_chunk = source_video[start:end]
            true_len = int(raw_chunk.shape[0])
            if true_len == 0:
                break
            # alinha o chunk ao grid 4n+1 (o ultimo chunk costuma cair fora)
            aligned_len = _align_up_4n1(true_len)
            if aligned_len != true_len:
                raw_chunk = _mirror_pad_frames(raw_chunk, aligned_len)

            source_chunk = _resize_source_video(raw_chunk, int(width), int(height), resize_mode)
            encoded_chunk = _encode_video(vae, source_chunk)
            context_latents = [encoded_chunk]
            context_latents.extend(
                _collect_reference_latents(
                    vae, true_len, int(ref_max_size),
                    reference_video=reference_video, reference_images=reference_images,
                )
            )

            if tail_memory and previous_generated_frames is not None:
                tail_count = min(int(tail_frames), int(previous_generated_frames.shape[0]))
                if tail_count > 0:
                    tail = previous_generated_frames[-tail_count:]
                    tail_latent = _encode_video(vae, tail[:, :, :, :3])
                    context_latents.append(tail_latent)
                    print(f"[Bernini Infinity][seq] tail_memory: +{tail_count} frames editados", flush=True)

            values = {"context_latents": context_latents}
            pos = _clone_conditioning_set_values(positive, values)
            neg = _clone_conditioning_set_values(negative, values)

            latent = {"samples": _make_empty_latent(aligned_len, int(width), int(height), 1)}

            chunk_seed = int(seed) + (chunk_index if vary_seed_per_chunk else 0)
            print(
                f"[Bernini Infinity][seq] chunk {chunk_index + 1}/{total_chunks}: "
                f"frames {start}..{end - 1} seed={chunk_seed} ctx={len(context_latents)}",
                flush=True,
            )

            high = SamplerCustom.execute(
                high_model, True, chunk_seed, float(cfg), pos, neg, sampler, high_sigmas, latent
            ).args[0]
            low = SamplerCustom.execute(
                low_model, False, 0, float(cfg), pos, neg, sampler, low_sigmas, high
            ).args[0]
            chunk_latent = low["samples"]
            imgs = _decode_video(vae, chunk_latent, bool(decode_tiled))
            # corta o padding de alinhamento: volta ao tamanho real do chunk
            if imgs.shape[0] > true_len:
                imgs = imgs[:true_len]
                chunk_latent = chunk_latent[:, :, :_lat_len(true_len)]

            # mantem so a area selecionada: fora da mascara volta a ser a fonte
            if use_mask:
                cm = seq_mask[start:end][:true_len].unsqueeze(-1).to(imgs.device, imgs.dtype)
                src_px = source_chunk[:true_len].to(imgs.device, imgs.dtype)
                imgs = src_px * (1.0 - cm) + imgs * cm

            # estado EDITADO -> alimenta tail_memory do proximo chunk (a propagacao)
            previous_generated_frames = imgs.detach().cpu()
            full_imgs = previous_generated_frames
            full_lat = chunk_latent.detach().cpu()

            if stitched_imgs is None:
                stitched_imgs = full_imgs
                stitched_latents = full_lat
            else:
                stitched_imgs = _merge_linear_overlap(stitched_imgs, full_imgs, overlap)
                stitched_latents = _merge_latent_overlap(stitched_latents, full_lat, lat_drop)

            del high, low, latent, chunk_latent, imgs, pos, neg, context_latents
            comfy.model_management.soft_empty_cache()
            chunk_index += 1
            if end >= target:
                break

        if stitched_imgs.shape[0] > target:
            stitched_imgs = stitched_imgs[:target]
            stitched_latents = stitched_latents[:, :, :(((target - 1) // 4) + 1)]

        print(f"[Bernini Infinity][seq] done: {int(stitched_imgs.shape[0])} frames, {chunk_index} chunk(s).", flush=True)
        return (stitched_imgs.cpu(), {"samples": stitched_latents.cpu()}, int(stitched_imgs.shape[0]))

    # ------------------------------------------------------------------
    # MODO CONTEXT_WINDOW: latente unico + janelas paralelas (geracao livre)
    # ------------------------------------------------------------------
    def _render_context_window(
        self, positive, negative, high_model, low_model, vae, source_video,
        width, height, seed, sampler, high_sigmas, low_sigmas, cfg,
        chunk_size, overlap, target, decode_tiled, decode_chunk,
        ref_max_size, reference_video, reference_images, context_jitter,
        region_mask=None, mask_mode="off", mask_grow=0, mask_blur=0, mask_pad=16,
        resize_mode="stretch",
    ):
        target = int(target)
        # --- alinhamento temporal: trabalhamos no proximo 4n+1 e cortamos depois ---
        aligned = _align_up_4n1(target)
        T_lat = _lat_len(aligned)
        win_lat = _lat_len(chunk_size)
        ovl_lat = _lat_len(overlap) if overlap > 0 else 0
        # IMPORTANTE: decide janelas pelo tamanho REAL (target), nao pelo padded.
        # Assim o padding de alinhamento (que so conserta a cauda do decode) nao
        # liga windowing a toa -- windowing so quando o video de fato passa da janela.
        use_ctx = win_lat < _lat_len(target)

        raw = source_video[:target]
        if aligned != target:
            raw = _mirror_pad_frames(raw, aligned)
            print(
                f"[Bernini Infinity][ctx] padding temporal {target}->{aligned} "
                f"(4n+1, espelhado) p/ nao perder frames", flush=True,
            )

        # --- preparo opcional da mascara (gerar so na area selecionada) ---
        use_mask = (mask_mode != "off") and (region_mask is not None)
        full_source = _resize_source_video(raw, int(width), int(height), resize_mode)
        mask_full = None
        if use_mask:
            mpix = _normalize_mask(region_mask)
            mpix = _resize_mask_spatial_temporal(mpix, aligned, int(height), int(width), resize_mode)
            mpix = _grow_blur_mask(mpix, int(mask_grow), int(mask_blur))
            mask_full = mpix

        # ============ MODO BBOX: recorta na bbox da mascara e gera menor ============
        if use_mask and mask_mode == "bbox" and not use_ctx:
            return self._render_ctx_bbox(
                positive, negative, high_model, low_model, vae,
                full_source, mask_full, aligned, target, int(width), int(height),
                seed, sampler, high_sigmas, low_sigmas, cfg,
                decode_tiled, decode_chunk, ref_max_size,
                reference_video, reference_images, int(mask_pad),
            )
        if use_mask and mask_mode == "bbox" and use_ctx:
            print("[Bernini Infinity][ctx] bbox indisponivel com janelas; "
                  "usando inpaint full-res.", flush=True)
            mask_mode = "inpaint"

        source = full_source
        encoded_source = _encode_video(vae, source)
        context_latents = [encoded_source]
        context_latents.extend(
            _collect_reference_latents(
                vae, int(aligned), int(ref_max_size),
                reference_video=reference_video, reference_images=reference_images,
            )
        )
        values = {"context_latents": context_latents}
        pos = _clone_conditioning_set_values(positive, values)
        neg = _clone_conditioning_set_values(negative, values)

        # ============ MODO INPAINT: gera normal e mantem so a area da mascara ======
        latent = {"samples": _make_empty_latent(int(aligned), int(width), int(height), 1)}
        if use_mask:
            print("[Bernini Infinity][ctx] mascara inpaint ativa "
                  "(composite em pixels: fora da mascara = fonte).", flush=True)

        print(
            f"[Bernini Infinity][ctx] target={target} T_lat={T_lat} win_lat={win_lat} "
            f"ovl_lat={ovl_lat} windowing={use_ctx} jitter={bool(context_jitter)} ctx={len(context_latents)}",
            flush=True,
        )
        try:
            print(f"[Bernini Infinity][ctx] source_latent={tuple(encoded_source.shape)}", flush=True)
        except Exception:
            pass

        hi = high_model.clone()
        lo = low_model.clone()
        if use_ctx:
            dbg = {}
            wrapper = _make_context_wrapper(win_lat, ovl_lat, ovl_lat, jitter=bool(context_jitter), debug_holder=dbg)
            hi.set_model_unet_function_wrapper(wrapper)
            lo.set_model_unet_function_wrapper(wrapper)

        high = SamplerCustom.execute(hi, True, int(seed), float(cfg), pos, neg, sampler, high_sigmas, latent).args[0]
        low = SamplerCustom.execute(lo, False, 0, float(cfg), pos, neg, sampler, low_sigmas, high).args[0]
        result_latent = low["samples"]

        if int(decode_chunk) > 0:
            result_imgs = _decode_video_chunked(vae, result_latent, bool(decode_tiled), int(decode_chunk), 1)
        else:
            result_imgs = _decode_video(vae, result_latent, bool(decode_tiled))

        # mantem so a area selecionada: fora da mascara volta a ser a fonte
        if use_mask:
            gen = result_imgs.cpu()
            src = full_source.cpu()
            n = min(gen.shape[0], src.shape[0], mask_full.shape[0])
            m = mask_full[:n].unsqueeze(-1).cpu()
            result_imgs = src[:n] * (1.0 - m) + gen[:n] * m

        if result_imgs.shape[0] > target:
            result_imgs = result_imgs[:target]
            result_latent = result_latent[:, :, :_lat_len(target)]

        comfy.model_management.soft_empty_cache()
        print(f"[Bernini Infinity][ctx] done: {int(result_imgs.shape[0])} frames "
              f"(alvo do usuario={target}).", flush=True)
        return (result_imgs.cpu(), {"samples": result_latent.cpu()}, int(result_imgs.shape[0]))

    # ------------------------------------------------------------------
    # MODO BBOX: recorta na bounding box da mascara, gera so esse retangulo
    #   (resolucao menor => mais rapido e menos VRAM) e cola de volta no
    #   video original com a mascara (com feather). "Gerar so na area".
    # ------------------------------------------------------------------
    def _render_ctx_bbox(
        self, positive, negative, high_model, low_model, vae,
        full_source, mask_full, aligned, target, width, height,
        seed, sampler, high_sigmas, low_sigmas, cfg,
        decode_tiled, decode_chunk, ref_max_size,
        reference_video, reference_images, mask_pad,
    ):
        x0, y0, x1, y1 = _mask_bbox(mask_full, int(mask_pad), 16, int(width), int(height))
        cw, ch = x1 - x0, y1 - y0
        print(
            f"[Bernini Infinity][bbox] regiao=({x0},{y0})-({x1},{y1}) "
            f"{cw}x{ch} de {int(width)}x{int(height)} "
            f"(~{100.0 * (cw * ch) / (int(width) * int(height)):.0f}% da area)",
            flush=True,
        )

        src_crop = full_source[:, y0:y1, x0:x1, :].contiguous()
        mask_crop = mask_full[:, y0:y1, x0:x1].contiguous()

        encoded_crop = _encode_video(vae, src_crop)
        context_latents = [encoded_crop]
        context_latents.extend(
            _collect_reference_latents(
                vae, int(aligned), int(ref_max_size),
                reference_video=reference_video, reference_images=reference_images,
            )
        )
        values = {"context_latents": context_latents}
        pos = _clone_conditioning_set_values(positive, values)
        neg = _clone_conditioning_set_values(negative, values)

        latent = {"samples": _make_empty_latent(int(aligned), cw, ch, 1)}

        high = SamplerCustom.execute(
            high_model.clone(), True, int(seed), float(cfg), pos, neg, sampler, high_sigmas, latent
        ).args[0]
        low = SamplerCustom.execute(
            low_model.clone(), False, 0, float(cfg), pos, neg, sampler, low_sigmas, high
        ).args[0]
        crop_latent = low["samples"]

        if int(decode_chunk) > 0:
            crop_imgs = _decode_video_chunked(vae, crop_latent, bool(decode_tiled), int(decode_chunk), 1)
        else:
            crop_imgs = _decode_video(vae, crop_latent, bool(decode_tiled))
        crop_imgs = crop_imgs.cpu()

        # cola de volta: fora da bbox = fonte original; dentro = blend pela mascara
        out = full_source.clone().cpu()
        n = min(out.shape[0], crop_imgs.shape[0])
        blend = mask_crop[:n].unsqueeze(-1).cpu()                 # [n,ch,cw,1]
        region = out[:n, y0:y1, x0:x1, :]
        out[:n, y0:y1, x0:x1, :] = region * (1.0 - blend) + crop_imgs[:n] * blend

        if out.shape[0] > target:
            out = out[:target]
        comfy.model_management.soft_empty_cache()
        print(f"[Bernini Infinity][bbox] done: {int(out.shape[0])} frames "
              f"(alvo do usuario={target}).", flush=True)
        return (out, {"samples": crop_latent.cpu()}, int(out.shape[0]))


class BerniniRegionMask:
    """Prepara uma MASK para o Bernini Infinity: aceita MASK ou IMAGE colorida
    (estilo SCAIL2ColoredMask), permite dilatar/contrair, suavizar a borda,
    binarizar por threshold e inverter. Saida pronta p/ o input region_mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grow": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1, "tooltip": "Dilata (+) ou contrai (-) a mascara, em pixels. + pega uma margem ao redor; - aperta a regiao."}),
                "blur": ("INT", {"default": 4, "min": 0, "max": 256, "step": 1, "tooltip": "Suaviza a borda (feather), em pixels. Deixa a emenda com a fonte mais natural. 0 = borda seca."}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Binariza a mascara: pixels >= threshold viram 1, o resto 0. 0.0 = nao binariza. Aplicado ANTES de grow/blur."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Inverte a mascara (troca o que e gerado pelo que e mantido). Aplicado por ultimo."}),
            },
            "optional": {
                "mask": ("MASK,IMAGE", {"tooltip": "Mascara de entrada (branco = regiao a gerar, preto = manter). Aceita MASK ou IMAGE colorida (ex.: SCAIL-2). Use esta OU 'image_mask'."}),
                "image_mask": ("IMAGE", {"tooltip": "Alternativa a 'mask': uma mascara COLORIDA como IMAGE (estilo SCAIL2ColoredMask). Qualquer pixel nao-preto vira regiao. Se ligado, tem prioridade sobre 'mask'."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ("Mascara tratada (0..1), pronta pra ligar no 'region_mask' do Bernini Infinity.",)
    DESCRIPTION = (
        "Prepara uma mascara pro Bernini Infinity: aceita MASK ou IMAGE colorida (SCAIL2ColoredMask), "
        "com dilatar/contrair (grow), suavizar borda (blur), binarizar (threshold) e inverter."
    )
    FUNCTION = "build"
    CATEGORY = "Bruxos do VFX/Bernini"

    def build(self, mask=None, grow=0, blur=4, threshold=0.0, invert=False, image_mask=None):
        src = image_mask if image_mask is not None else mask
        m = _normalize_mask(src)
        if m is None:
            raise ValueError("Conecte uma MASK ou um IMAGE em image_mask.")
        if float(threshold) > 0.0:
            m = (m >= float(threshold)).float()
        m = _grow_blur_mask(m, int(grow), int(blur))
        if invert:
            m = 1.0 - m
        return (m.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "BruxosBerniniInfinity": BerniniInfinity,
    "BruxosBerniniRegionMask": BerniniRegionMask,
    "BruxosBerniniLongConditioning": BerniniLongConditioning,
    "BruxosBerniniLongChunkSelect": BerniniLongChunkSelect,
    "BruxosBerniniLongVideoMerge": BerniniLongVideoMerge,
    "BruxosBerniniLongAppendVideoChunk": BerniniLongAppendVideoChunk,
    "BruxosBerniniLongEmptyVideoChunks": BerniniLongEmptyVideoChunks,
    "BruxosBerniniLongInfo": BerniniLongInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosBerniniInfinity": "Bernini Infinity",
    "BruxosBerniniRegionMask": "Bernini Region Mask",
    "BruxosBerniniLongConditioning": "Bernini Long Condition",
    "BruxosBerniniLongChunkSelect": "Bernini Long Chunk Select",
    "BruxosBerniniLongVideoMerge": "Bernini Long Video Merge",
    "BruxosBerniniLongAppendVideoChunk": "Bernini Long Append Video Chunk",
    "BruxosBerniniLongEmptyVideoChunks": "Bernini Long Empty Video Chunks",
    "BruxosBerniniLongInfo": "Bernini Long Info",
}


# =============================================================================
# FaceStitchUpscale (Bruxos do VFX)
# -----------------------------------------------------------------------------
# Cola o rosto de volta no video depois de um upscale, usando os face_bboxes
# do node "Pose and Face Detection" (ComfyUI-WanAnimatePreprocess / kijai).
#
# O crop original do WanAnimate, por frame, e:
#     face = frame[y1:y2, x1:x2]
#     face = cv2.resize(face, (512, 512))
# e face_bboxes guarda exatamente (x1, y1, x2, y2) em pixels do frame original.
# Este node faz o inverso: redimensiona o rosto upscalado de volta para
# (x2-x1, y2-y1) e compoe na posicao [y1:y2, x1:x2], com borda suave (feather).
# E 100% reversivel.
# =============================================================================

import logging as _fsu_logging

import numpy as _fsu_np

try:
    import cv2 as _fsu_cv2
    _FSU_HAS_CV2 = True
except Exception:  # pragma: no cover
    _FSU_HAS_CV2 = False


class _FsuAnyType(str):
    """Tipo coringa: o socket face_bboxes do WanAnimate tem tipo "BBOX," (typo
    com virgula no codigo do kijai). Um coringa garante a conexao direta."""

    def __ne__(self, other):
        return False


_FSU_ANY = _FsuAnyType("*")


def _fsu_normalize_image(t):
    """float32 0..1, shape (B,H,W,C)."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    t = t.float()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.shape[-1] not in (1, 3, 4) and t.shape[1] in (1, 3, 4):
        t = t.permute(0, 2, 3, 1).contiguous()
    if t.numel() and t.max() > 1.5:
        t = t / 255.0
    return t.clamp(0.0, 1.0)


def _fsu_parse_bboxes(face_bboxes):
    """Normaliza para lista de (x1,y1,x2,y2) floats."""
    boxes = face_bboxes
    if isinstance(boxes, dict):
        for k in ("face_bboxes", "bboxes", "boxes"):
            if k in boxes:
                boxes = boxes[k]
                break
    if isinstance(boxes, (tuple, list)) and len(boxes) == 4 and all(
        isinstance(v, (int, float, _fsu_np.integer, _fsu_np.floating)) for v in boxes
    ):
        boxes = [boxes]
    out = []
    for b in boxes:
        arr = _fsu_np.array(b).flatten().astype(float).tolist()
        if len(arr) >= 4:
            out.append((arr[0], arr[1], arr[2], arr[3]))
    return out


def _fsu_resize(img, w, h):
    """Redimensiona array HWC (0..1). cv2 se disponivel, senao torch."""
    if _FSU_HAS_CV2:
        cur_h, cur_w = img.shape[:2]
        shrink = (cur_h > h) or (cur_w > w)
        interp = _fsu_cv2.INTER_AREA if shrink else _fsu_cv2.INTER_CUBIC
        res = _fsu_cv2.resize(img, (w, h), interpolation=interp)
        if res.ndim == 2:
            res = res[..., None]
        return res
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()


def _fsu_blur(m, sigma):
    if sigma <= 0:
        return m
    if _FSU_HAS_CV2:
        return _fsu_cv2.GaussianBlur(m, (0, 0), sigmaX=sigma, sigmaY=sigma)
    k = max(1, int(sigma) * 2 + 1)
    ker = _fsu_np.ones(k, dtype=_fsu_np.float32) / k
    tmp = _fsu_np.apply_along_axis(lambda r: _fsu_np.convolve(r, ker, mode="same"), 1, m)
    return _fsu_np.apply_along_axis(lambda c: _fsu_np.convolve(c, ker, mode="same"), 0, tmp)


def _fsu_blend_mask(h, w, feather, shape="ellipse", inset=0):
    """Mascara de blend HxW.
    shape: 'ellipse' (recomendado p/ rosto) ou 'rectangle'.
    inset: encolhe (+) a area solida pra dentro, em px (NAO mexe no crop, so na
           mascara — mantem o rosto alinhado). inset negativo cresce.
    feather: suavizacao gaussiana da borda em px.
    """
    m = _fsu_np.zeros((h, w), dtype=_fsu_np.float32)
    ins = float(inset)
    if shape == "ellipse":
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        ry = max(1.0, h / 2.0 - max(0.0, ins))
        rx = max(1.0, w / 2.0 - max(0.0, ins))
        if ins < 0:  # crescer alem da caixa
            ry = h / 2.0 - ins
            rx = w / 2.0 - ins
        yy, xx = _fsu_np.ogrid[:h, :w]
        ell = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        m[ell <= 1.0] = 1.0
    else:  # rectangle
        x0 = int(max(0, ins)); y0 = int(max(0, ins))
        x1 = int(w - max(0, ins)); y1 = int(h - max(0, ins))
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, w, h
        m[y0:y1, x0:x1] = 1.0
    if feather and feather > 0:
        m = _fsu_blur(m, max(1.0, feather / 2.0))
    return _fsu_np.clip(m, 0.0, 1.0)


class FaceStitchUpscale:
    """Cola rostos upscalados de volta no video usando os face_bboxes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_frames": ("IMAGE", {"tooltip": "Video onde o rosto sera colado (original OU ja upscalado)."}),
                "upscaled_faces": ("IMAGE", {"tooltip": "Batch de rostos depois do upscale. Mesma contagem de frames."}),
                "face_bboxes": (_FSU_ANY, {"tooltip": "Saida 'face_bboxes' do Pose and Face Detection."}),
            },
            "optional": {
                "bbox_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01,
                                         "tooltip": "Se target_frames foi upscalado Nx, coloque N (ex: 2.0). Original = 1.0."}),
                "mask_shape": (["ellipse", "rectangle"], {"default": "ellipse",
                                "tooltip": "ellipse esconde o 'box' do bbox 1.3x do WanAnimate. rectangle = patch reto."}),
                "feather": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1,
                                    "tooltip": "Suavizacao da borda (px). Suba se ver costura."}),
                "mask_expand": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1,
                                        "tooltip": "Encolhe (+) a mascara pra dentro sem desalinhar o rosto. Use +20..+60 se o box sobra alem do rosto."}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "1.0 substitui o rosto inteiro; <1.0 mistura com o original."}),
                "external_mask": ("MASK", {"tooltip": "Opcional: mascara full-frame (ex: SAM2) para limitar a colagem so ao rosto/sujeito."}),
                "color_match": (["off", "mean", "mean_std"], {"default": "mean",
                                "tooltip": "Casa a cor/exposicao do rosto novo com a regiao original. 'mean' corrige tom; 'mean_std' tambem o contraste."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "face_mask")
    FUNCTION = "stitch"
    CATEGORY = "Bruxos do VFX/Face"

    def stitch(self, target_frames, upscaled_faces, face_bboxes,
               bbox_scale=1.0, mask_shape="ellipse", feather=24, mask_expand=0,
               blend=1.0, external_mask=None, color_match="mean"):
        frames = _fsu_normalize_image(target_frames)
        faces = _fsu_normalize_image(upscaled_faces)

        B, H, W, C = frames.shape
        Bf = faces.shape[0]
        boxes = _fsu_parse_bboxes(face_bboxes)

        if len(boxes) == 0:
            _fsu_logging.warning("[FaceStitchUpscale] Nenhum bbox recebido; retornando frames sem alteracao.")
            return (frames, torch.zeros((B, H, W), dtype=torch.float32))

        n = min(B, Bf, len(boxes))
        if not (B == Bf == len(boxes)):
            _fsu_logging.warning(
                f"[FaceStitchUpscale] Contagens diferentes (frames={B}, faces={Bf}, "
                f"bboxes={len(boxes)}). Processando {n} frames."
            )

        # mascara externa opcional (B,H,W) -> numpy
        ext = None
        if external_mask is not None:
            em = external_mask
            if not torch.is_tensor(em):
                em = torch.as_tensor(em)
            em = em.float()
            if em.ndim == 2:
                em = em.unsqueeze(0)
            ext = em.cpu().numpy()

        frames_np = frames.cpu().numpy()
        faces_np = faces.cpu().numpy()
        out = frames_np.copy()
        mask_out = _fsu_np.zeros((B, H, W), dtype=_fsu_np.float32)

        for i in range(n):
            x1, y1, x2, y2 = boxes[i]
            # crop = bbox exato (mantem alinhamento). NAO mexer aqui com mask_expand.
            x1 = int(round(x1 * bbox_scale)); y1 = int(round(y1 * bbox_scale))
            x2 = int(round(x2 * bbox_scale)); y2 = int(round(y2 * bbox_scale))
            x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W, x2))
            y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))
            cw, ch = x2 - x1, y2 - y1
            if cw <= 1 or ch <= 1:
                continue

            face = faces_np[i]
            if face.shape[-1] != C:
                if face.shape[-1] == 1:
                    face = _fsu_np.repeat(face, C, axis=-1)
                else:
                    face = face[..., :C]
            face_resized = _fsu_resize(face, cw, ch)[..., :C]

            # mascara de blend: elipse/retangulo com inset (mask_expand) + feather
            m = _fsu_blend_mask(ch, cw, feather, shape=mask_shape, inset=mask_expand)

            # color match: casa tom/exposicao do rosto novo com a regiao original
            if color_match != "off":
                region0 = out[i, y1:y2, x1:x2, :]
                w = m[..., None]
                wsum = float(w.sum()) + 1e-6
                src_mean = (face_resized * w).reshape(-1, C).sum(0) / wsum
                dst_mean = (region0 * w).reshape(-1, C).sum(0) / wsum
                if color_match == "mean_std":
                    src_std = _fsu_np.sqrt(((face_resized - src_mean) ** 2 * w).reshape(-1, C).sum(0) / wsum) + 1e-6
                    dst_std = _fsu_np.sqrt(((region0 - dst_mean) ** 2 * w).reshape(-1, C).sum(0) / wsum) + 1e-6
                    face_resized = (face_resized - src_mean) * (dst_std / src_std) + dst_mean
                else:
                    face_resized = face_resized + (dst_mean - src_mean)
                face_resized = _fsu_np.clip(face_resized, 0.0, 1.0)
            # intersecta com mascara externa (SAM2) se houver
            if ext is not None:
                idx = min(i, ext.shape[0] - 1)
                em_frame = ext[idx]
                if em_frame.shape[:2] != (H, W):
                    em_frame = _fsu_resize(em_frame[..., None], W, H)[..., 0]
                em_crop = em_frame[y1:y2, x1:x2]
                if em_crop.shape != (ch, cw):
                    em_crop = _fsu_resize(em_crop[..., None], cw, ch)[..., 0]
                m = m * _fsu_np.clip(em_crop, 0.0, 1.0)
            m = m * float(blend)

            m3 = m[..., None]
            region = out[i, y1:y2, x1:x2, :]
            out[i, y1:y2, x1:x2, :] = face_resized * m3 + region * (1.0 - m3)
            mask_out[i, y1:y2, x1:x2] = _fsu_np.maximum(mask_out[i, y1:y2, x1:x2], m)

        out_t = torch.from_numpy(_fsu_np.clip(out, 0.0, 1.0)).float()
        mask_t = torch.from_numpy(mask_out).float()
        return (out_t, mask_t)


NODE_CLASS_MAPPINGS["BruxosFaceStitchUpscale"] = FaceStitchUpscale
NODE_DISPLAY_NAME_MAPPINGS["BruxosFaceStitchUpscale"] = "FaceStitchUpscale"


# =============================================================================
# Bruxos do VFX — utilidades extra (v0.4.0)
# -----------------------------------------------------------------------------
#  * BruxosPad4n1            : padding temporal espelhado para 4n+1 frames
#  * BruxosTrim4n1           : corte de volta ao numero original de frames
#  * BruxosQwenVLCaption     : caption de video/imagem com Qwen2.5-VL
#                              (substituto direto do Florence2Run)
# =============================================================================

import os as _bx_os
import logging as _bx_logging


# -------------------------------------------------------------------------
# Alinhamento 4n+1 (Wan VAE comprime tempo ~4x)
# Mesmo principio do Bernini Infinity: N frames -> ((N-1)//4)+1 latentes ->
# decode devolve (T_lat-1)*4 + 1 frames. So sobrevivem comprimentos 4n+1.
# Em vez de embutir o fix em cada sampler, esses dois nodes envolvem
# *qualquer* etapa que rode no Wan (incl. UltimateSDUpscaleNoUpscale com Wan).
# -------------------------------------------------------------------------

def _bx_next_4n1(n: int) -> int:
    """Proximo comprimento 4n+1 >= n."""
    if n <= 1:
        return 1
    return ((n - 1 + 3) // 4) * 4 + 1


def _bx_mirror_pad_frames(frames: "torch.Tensor", target_len: int) -> "torch.Tensor":
    """Pad temporal por reflexao ping-pong (sem frame congelado, como o
    truque do Kijai no Wan Animate). frames: (B,H,W,C)."""
    cur = frames.shape[0]
    if target_len <= cur:
        return frames[:target_len]
    need = target_len - cur
    pieces = [frames]
    # ping-pong: ...3,2,1,0, 0,1,2,3, 3,2,1,0, ...
    direction = -1  # primeiro espelha pra tras
    while need > 0:
        if direction == -1:
            chunk = frames.flip(0)[1:1 + need] if cur > 1 else frames[:need]
        else:
            chunk = frames[1:1 + need] if cur > 1 else frames[:need]
        pieces.append(chunk)
        need -= chunk.shape[0]
        direction = -direction
    out = torch.cat(pieces, dim=0)[:target_len]
    return out


class BruxosPad4n1:
    """Pad de video pro proximo comprimento 4n+1 (compatibilidade Wan VAE).
    Use ANTES de qualquer node que rode no Wan. Use BruxosTrim4n1 depois
    pra cortar de volta ao numero original de frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video de entrada que sera ajustado para 4n+1 frames."}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "Desligue para passar reto sem padding (debug)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "original_count", "padded_count")
    FUNCTION = "run"
    CATEGORY = "Bruxos do VFX/Video"

    def run(self, images, enabled=True):
        n = int(images.shape[0])
        if not enabled:
            return (images, n, n)
        target = _bx_next_4n1(n)
        if target == n:
            return (images, n, n)
        padded = _bx_mirror_pad_frames(images, target)
        _bx_logging.info(f"[BruxosPad4n1] padding temporal {n}->{target} (4n+1, espelhado)")
        return (padded, n, target)


class BruxosTrim4n1:
    """Corta o video ao numero original de frames (par com BruxosPad4n1).
    Sobrevive a perdas de frames no meio do caminho — se o pipeline devolveu
    menos do que o pad pediu, ele toma min(disponivel, original)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video processado, possivelmente com frames a mais do padding 4n+1."}),
                "original_count": ("INT", {"default": 0, "min": 0, "max": 10_000_000,
                    "tooltip": "Numero de frames original (ligue no original_count do Pad ou no frame_count do loader)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Bruxos do VFX/Video"

    def run(self, images, original_count):
        cur = int(images.shape[0])
        if original_count <= 0:
            return (images,)
        if cur < original_count:
            _bx_logging.warning(
                f"[BruxosTrim4n1] pipeline devolveu {cur} frames, esperava >={original_count}. "
                f"Devolvendo {cur} (sem completar)."
            )
            return (images,)
        return (images[:original_count],)


# -------------------------------------------------------------------------
# Qwen2.5-VL caption — substituto do Florence2Run
# Output: STRING (mesmo shape do Florence) -> drop-in no StringReplace/JoinStrings
# Funciona em duas modalidades:
#   - "single_frame"     : extrai 1 frame (index configuravel) e captiona
#   - "keyframes_merge"  : amostra N frames espacados e pede 1 caption unica
# O modelo e baixado por transformers/Hugging Face na primeira execucao.
# -------------------------------------------------------------------------

_BX_QWEN_DEFAULT_INSTRUCTION = (
    "Describe this video in one rich paragraph for a text-to-video upscale "
    "prompt: subjects, clothing, materials, environment, lighting, camera, "
    "color palette, and motion. Be concrete and visual. No meta commentary."
)

_BX_QWEN_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]

_BX_QWEN_CACHE = {"name": None, "model": None, "processor": None}


def _bx_qwen_load(model_name: str, dtype_str: str, device: str):
    """Carrega Qwen-VL via transformers, com cache em memoria."""
    if _BX_QWEN_CACHE["name"] == (model_name, dtype_str, device) and _BX_QWEN_CACHE["model"] is not None:
        return _BX_QWEN_CACHE["model"], _BX_QWEN_CACHE["processor"]
    try:
        from transformers import AutoProcessor
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers nao esta instalado. Rode: pip install -U transformers accelerate"
        ) from e

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
        dtype_str, torch.float16
    )

    # Qwen2.5-VL e Qwen2-VL usam classes diferentes; tentamos os dois.
    Model = None
    last_err = None
    for cls_name in ("Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"):
        try:
            import transformers
            Model = getattr(transformers, cls_name, None)
            if Model is None:
                continue
            model = Model.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
            processor = AutoProcessor.from_pretrained(model_name)
            _BX_QWEN_CACHE.update(
                {"name": (model_name, dtype_str, device), "model": model, "processor": processor}
            )
            return model, processor
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Nao consegui carregar {model_name}. Verifique o nome do modelo e a versao "
        f"do transformers (precisa de >=4.45 para Qwen2.5-VL). Ultimo erro: {last_err}"
    )


def _bx_tensor_to_pil(frame):
    """(H,W,C) torch 0..1 -> PIL.Image RGB."""
    from PIL import Image
    arr = (frame.detach().cpu().clamp(0, 1).numpy() * 255.0).astype("uint8")
    if arr.shape[-1] == 1:
        arr = arr.repeat(3, axis=-1)
    return Image.fromarray(arr[..., :3])


class BruxosQwenVLCaption:
    """Caption de imagem/video com Qwen2.5-VL.
    Saida 'caption' (STRING) e drop-in pro lugar do Florence2Run.caption."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frames de entrada que serao descritos."}),
                "model_name": (_BX_QWEN_MODELS, {"default": _BX_QWEN_MODELS[0],
                    "tooltip": "Qual modelo Qwen-VL usar (3B leve; 7B mais forte). Baixa do HuggingFace na 1a vez."}),
                "mode": (["single_frame", "keyframes_merge"], {"default": "keyframes_merge",
                    "tooltip": "single_frame: 1 frame (como o Florence). keyframes_merge: varios frames -> UM prompt unico."}),
            },
            "optional": {
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": _BX_QWEN_DEFAULT_INSTRUCTION,
                    "tooltip": "O que o modelo deve descrever. Ja vem ajustada p/ prompt de upscale.",
                }),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 10_000_000,
                    "tooltip": "Indice do frame em single_frame."}),
                "num_keyframes": ("INT", {"default": 6, "min": 2, "max": 32,
                    "tooltip": "Quantos keyframes amostrar em keyframes_merge."}),
                "max_new_tokens": ("INT", {"default": 220, "min": 16, "max": 2048,
                    "tooltip": "Tamanho maximo do texto gerado (mais tokens = descricao mais longa)."}),
                "dtype": (["fp16", "bf16", "fp32"], {"default": "fp16",
                    "tooltip": "Precisao do modelo. fp16 economiza VRAM; bf16 se a GPU suportar."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto",
                    "tooltip": "Onde rodar. auto escolhe GPU se houver."}),
                "keep_loaded": ("BOOLEAN", {"default": True,
                    "tooltip": "Mantem o modelo na memoria entre execucoes (mais rapido, usa VRAM)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Semente da geracao de texto (0 = livre)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "run"
    CATEGORY = "Bruxos do VFX/Caption"

    def _sample_frames(self, images, mode, frame_index, num_keyframes):
        n = int(images.shape[0])
        if mode == "single_frame" or n == 1:
            idx = max(0, min(n - 1, frame_index))
            return [images[idx]]
        k = max(2, min(num_keyframes, n))
        # amostragem uniforme cobrindo o video
        step = (n - 1) / float(k - 1) if k > 1 else 0
        idxs = sorted({int(round(i * step)) for i in range(k)})
        return [images[i] for i in idxs]

    def run(self, images, model_name, mode,
            instruction=_BX_QWEN_DEFAULT_INSTRUCTION,
            frame_index=0, num_keyframes=6, max_new_tokens=220,
            dtype="fp16", device="auto", keep_loaded=True, seed=0):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model, processor = _bx_qwen_load(model_name, dtype, device)

        # Monta a mensagem multimodal (formato Qwen2-VL/2.5-VL)
        pil_frames = [_bx_tensor_to_pil(f) for f in self._sample_frames(
            images, mode, frame_index, num_keyframes
        )]
        content = [{"type": "image", "image": img} for img in pil_frames]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]

        # Os processors do Qwen-VL aceitam tanto apply_chat_template quanto
        # a API "imagens + texto" direta. Tentamos chat_template (oficial).
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=pil_frames, return_tensors="pt", padding=True,
            ).to(device)
        except Exception:
            # fallback simples
            inputs = processor(
                text=[instruction], images=pil_frames, return_tensors="pt", padding=True,
            ).to(device)

        if seed:
            try:
                torch.manual_seed(int(seed))
            except Exception:
                pass

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        # remove o prompt do output
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        out_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        if not keep_loaded:
            _BX_QWEN_CACHE.update({"name": None, "model": None, "processor": None})
            try:
                del model, processor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception:
                pass

        return (out_text,)


NODE_CLASS_MAPPINGS["BruxosPad4n1"] = BruxosPad4n1
NODE_CLASS_MAPPINGS["BruxosTrim4n1"] = BruxosTrim4n1
NODE_CLASS_MAPPINGS["BruxosQwenVLCaption"] = BruxosQwenVLCaption

NODE_DISPLAY_NAME_MAPPINGS["BruxosPad4n1"] = "Pad to 4n+1 (Bruxos)"
NODE_DISPLAY_NAME_MAPPINGS["BruxosTrim4n1"] = "Trim 4n+1 back to N (Bruxos)"
NODE_DISPLAY_NAME_MAPPINGS["BruxosQwenVLCaption"] = "Qwen-VL Caption (Bruxos)"


# =============================================================================
# BruxosFaceCropExpand
# -----------------------------------------------------------------------------
# Re-crop dos rostos a partir dos face_bboxes do WanAnimate, com expansao
# controlavel (zoom-out + extra no topo p/ cabelo) e crop QUADRADO real
# (sem o estica-desestica do 512x512). Gera novos face_images E novos
# face_bboxes — use ambos: crops -> upscale, bboxes -> FaceStitchUpscale.
#
#   Trim/video (mesma res da deteccao) -> images
#   PoseAndFaceDetection.face_bboxes   -> face_bboxes
#       -> face_images  (mais contexto) -> [upscale Wan] -> FaceStitchUpscale.upscaled_faces
#       -> face_bboxes  (expandidos)    ------------------> FaceStitchUpscale.face_bboxes
# =============================================================================

class BruxosFaceCropExpand:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frames na MESMA resolucao em que os bboxes foram medidos (ex: saida do Trim que entrou na deteccao)."}),
                "face_bboxes": (_FSU_ANY, {"tooltip": "Saida 'face_bboxes' do Pose and Face Detection."}),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 1.8, "min": 1.0, "max": 4.0, "step": 0.05,
                            "tooltip": "Zoom-out em torno do rosto. 1.0 = bbox original (fechado). 1.8 pega cabelo/ombros."}),
                "top_extra": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05,
                            "tooltip": "Extra so no TOPO (fracao da altura do bbox) pra incluir cabelo/testa."}),
                "square": ("BOOLEAN", {"default": True,
                            "tooltip": "Crop quadrado (recomendado p/ o upscaler nao distorcer)."}),
                "output_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16,
                            "tooltip": "Tamanho dos crops de saida (lado, em px)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BBOX")
    RETURN_NAMES = ("face_images", "face_bboxes")
    FUNCTION = "run"
    CATEGORY = "Bruxos do VFX/Face"

    def run(self, images, face_bboxes, scale=1.8, top_extra=0.35, square=True, output_size=512):
        frames = _fsu_normalize_image(images)
        B, H, W, C = frames.shape
        boxes = _fsu_parse_bboxes(face_bboxes)
        if len(boxes) == 0:
            _fsu_logging.warning("[BruxosFaceCropExpand] Nenhum bbox; passando frames reduzidos.")
            empty = _fsu_resize(frames[0].cpu().numpy(), output_size, output_size)[None]
            return (torch.from_numpy(empty).float(), [])

        n = min(B, len(boxes))
        frames_np = frames.cpu().numpy()
        crops = []
        new_boxes = []
        for i in range(n):
            x1, y1, x2, y2 = boxes[i]
            bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0

            nw = bw * scale
            nh = bh * scale
            # extra no topo: empurra o centro pra cima e aumenta a altura
            top = top_extra * bh
            ny1 = cy - nh / 2.0 - top
            ny2 = cy + nh / 2.0
            nx1 = cx - nw / 2.0
            nx2 = cx + nw / 2.0

            if square:
                cur_w = nx2 - nx1; cur_h = ny2 - ny1
                side = max(cur_w, cur_h)
                ccx = (nx1 + nx2) / 2.0; ccy = (ny1 + ny2) / 2.0
                nx1 = ccx - side / 2.0; nx2 = ccx + side / 2.0
                ny1 = ccy - side / 2.0; ny2 = ccy + side / 2.0
                # shift-to-fit (mantem quadrado dentro da imagem sem distorcer)
                if nx1 < 0: nx2 -= nx1; nx1 = 0
                if ny1 < 0: ny2 -= ny1; ny1 = 0
                if nx2 > W: nx1 -= (nx2 - W); nx2 = W
                if ny2 > H: ny1 -= (ny2 - H); ny2 = H

            ix1 = max(0, int(round(nx1))); iy1 = max(0, int(round(ny1)))
            ix2 = min(W, int(round(nx2))); iy2 = min(H, int(round(ny2)))
            if ix2 - ix1 < 2 or iy2 - iy1 < 2:
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

            crop = frames_np[i][iy1:iy2, ix1:ix2]
            crop = _fsu_resize(crop, output_size, output_size)[..., :C]
            crops.append(crop)
            new_boxes.append((ix1, iy1, ix2, iy2))

        face_images = torch.from_numpy(_fsu_np.stack(crops, 0)).float()
        return (face_images, new_boxes)


NODE_CLASS_MAPPINGS["BruxosFaceCropExpand"] = BruxosFaceCropExpand
NODE_DISPLAY_NAME_MAPPINGS["BruxosFaceCropExpand"] = "Face Crop Expand (Bruxos)"


# =============================================================================
# Descricoes em portugues (DESCRIPTION) — aparecem ao passar o mouse no node.
# Anexadas aqui pra cobrir cada node sem alterar as classes.
# =============================================================================

BerniniLongConditioning.DESCRIPTION = (
    "Bernini Long Condition — quebra um video longo em chunks e gera os "
    "condicionamentos certos pra cada pedaco, em vez de um Bernini Conditioning "
    "unico (assim passa do limite de ~81 frames sem novo sampler).\n"
    "ENTRADAS: positive/negative (condicionamentos); vae; source_video (video "
    "de origem); width/height (resolucao de geracao); chunk_size (frames por "
    "chunk, padrao 81); overlap (frames de sobreposicao entre chunks, p/ "
    "transicao suave, padrao 5); batch_size; tail_memory (liga memoria do fim "
    "do chunk anterior como context_latents, padrao True); tail_frames (quantos "
    "frames de cauda usar, padrao 5).\n"
    "SAIDAS: positive_chunks, negative_chunks, latent_chunks (listas por chunk), "
    "video_chunks, chunk_ranges (intervalos de frames), chunk_count (quantidade)."
)

BerniniLongChunkSelect.DESCRIPTION = (
    "Bernini Long Chunk Select — escolhe um indice de chunk e devolve o "
    "positive, negative e latent daquele pedaco, pra voce renderizar cada chunk "
    "com o sampler Bernini que ja usa (indice 0, depois 1, etc.)."
)

BerniniLongVideoMerge.DESCRIPTION = (
    "Bernini Long Video Merge — recebe a lista de videos ja renderizados (um por "
    "chunk) e junta tudo, fazendo blend linear na regiao de overlap pra nao "
    "aparecer emenda entre os pedacos."
)

BerniniLongAppendVideoChunk.DESCRIPTION = (
    "Bernini Long Append Video Chunk — utilitario: adiciona um video renderizado "
    "a uma lista de chunks (use junto com o Empty Video Chunks pra montar a lista "
    "manualmente antes do Merge)."
)

BerniniLongEmptyVideoChunks.DESCRIPTION = (
    "Bernini Long Empty Video Chunks — cria uma lista vazia de chunks de video, "
    "ponto de partida pra ir adicionando os renders com o Append Video Chunk."
)

BerniniLongInfo.DESCRIPTION = (
    "Bernini Long Info — mostra os intervalos de frames (ranges) de cada chunk, "
    "pra voce conferir como o video foi dividido."
)

FaceStitchUpscale.DESCRIPTION = (
    "FaceStitchUpscale — cola o rosto upscalado de volta no video usando os "
    "face_bboxes da deteccao (WanAnimate). Resolve o encaixe frame a frame.\n"
    "PARAMETROS:\n"
    "- target_frames: video onde o rosto sera colado (original OU ja upscalado).\n"
    "- upscaled_faces: os rostos depois do upscale (mesma contagem de frames).\n"
    "- face_bboxes: saida de bboxes da deteccao (ou do Face Crop Expand).\n"
    "- bbox_scale: 1.0 se cola no video original; se o alvo foi upscalado Nx, use N.\n"
    "- mask_shape: ellipse (esconde o 'box' do bbox 1.3x) ou rectangle.\n"
    "- feather: suavizacao da borda em px (suba se ver costura).\n"
    "- mask_expand: encolhe (+) a mascara pra dentro SEM desalinhar o rosto.\n"
    "- blend: 1.0 substitui o rosto; <1.0 mistura com o original.\n"
    "- external_mask (opcional): mascara full-frame (ex: SAM2) p/ limitar a colagem.\n"
    "- color_match: off/mean/mean_std — casa cor e exposicao com a regiao original.\n"
    "SAIDAS: images (video com rosto colado) e face_mask (onde colou)."
)

BruxosPad4n1.DESCRIPTION = (
    "Pad to 4n+1 — faz padding temporal do video pro proximo comprimento 4n+1 "
    "(o Wan VAE comprime o tempo ~4x e so sobrevivem comprimentos 4n+1). Use "
    "ANTES de qualquer etapa que rode no Wan; depois use o Trim 4n+1 pra cortar "
    "de volta. O padding e por espelhamento ping-pong (sem frame congelado).\n"
    "- images: video de entrada.\n"
    "- enabled: desligue pra passar reto sem padding (debug).\n"
    "SAIDAS: images (padded), original_count (N original), padded_count (4n+1)."
)

BruxosTrim4n1.DESCRIPTION = (
    "Trim 4n+1 back to N — corta o video de volta ao numero original de frames "
    "(par do Pad to 4n+1). Tolerante: se o pipeline devolveu menos frames que o "
    "esperado, devolve o que tem sem quebrar.\n"
    "- images: video processado.\n"
    "- original_count: numero de frames original (ligue no original_count do Pad, "
    "ou no frame_count do loader)."
)

BruxosQwenVLCaption.DESCRIPTION = (
    "Qwen-VL Caption — gera o prompt a partir do video usando Qwen2.5-VL "
    "(substituto direto do Florence2Run; a saida 'caption' e STRING).\n"
    "- images: frames de entrada.\n"
    "- model_name: qual modelo Qwen-VL usar (3B e leve; 7B mais forte).\n"
    "- mode: single_frame (1 frame, como o Florence) ou keyframes_merge "
    "(amostra varios frames e gera UM prompt unico — pega mudancas de cena).\n"
    "- instruction: instrucao do que descrever (ja vem afiada p/ upscale).\n"
    "- frame_index: qual frame em single_frame.\n"
    "- num_keyframes: quantos frames amostrar em keyframes_merge.\n"
    "- max_new_tokens: tamanho maximo do texto gerado.\n"
    "- dtype/device: precisao e GPU/CPU.\n"
    "- keep_loaded: mantem o modelo na memoria entre execucoes.\n"
    "Requer: pip install -U transformers accelerate pillow."
)

BruxosFaceCropExpand.DESCRIPTION = (
    "Face Crop Expand — re-crops o rosto a partir dos face_bboxes com expansao "
    "controlavel, pra dar mais contexto (cabelo, testa, ombros) ao upscaler e "
    "evitar que ele erre o sujeito. Faz crop QUADRADO real (sem estica-desestica).\n"
    "- images: frames na MESMA resolucao em que os bboxes foram medidos.\n"
    "- face_bboxes: saida da deteccao (Pose and Face Detection).\n"
    "- scale: zoom-out em torno do rosto (1.0=fechado, 1.8 pega cabelo/ombros).\n"
    "- top_extra: margem extra SO no topo (fracao da altura) p/ incluir cabelo.\n"
    "- square: crop quadrado (recomendado).\n"
    "- output_size: tamanho (lado) dos crops de saida.\n"
    "SAIDAS: face_images (crops com mais contexto) e face_bboxes (expandidos — "
    "ligue no FaceStitchUpscale p/ manter o alinhamento)."
)
