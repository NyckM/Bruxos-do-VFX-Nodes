"""MiniMax H3: ferramentas AV e memoria de tail latente no SSD.

Inspirado pelas conclusoes tecnicas do ComfyUI-MMH3Tools (MIT), mas escrito
para a API classica de custom nodes usada pelo pacote Bruxos. O formato de
disco guarda tensores simples, nao a classe NestedTensor, para sobreviver a
atualizacoes do ComfyUI.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time

import torch
import torch.nn.functional as F

try:
    import folder_paths
except Exception:  # testes fora do ComfyUI
    folder_paths = None


CAT = "Bruxos do VFX/MiniMax H3/Latent SSD"
FPS = 24
AUDIO_FPS = 40
VIDEO_T_DIM = 2
AUDIO_T_DIM = 3
LATENT_BASE = 2
LATENTS_PER_GROUP = 5
FRAME_BASE = 5
FRAMES_PER_GROUP = 17
MARCA = "BRUXOS_H3_LATENT_SSD_V1"


def _nested_cls():
    from comfy.nested_tensor import NestedTensor
    return NestedTensor


def _unpack(latent, nome="latent", allow_plain=True):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError(f"[Bruxos H3] {nome} nao e um LATENT valido.")
    s = latent["samples"]
    if torch.is_tensor(s):
        if allow_plain and s.ndim == 5:
            return s, None
        raise ValueError(f"[Bruxos H3] {nome} e tensor simples, nao AV H3.")
    parts = None
    if hasattr(s, "unbind"):
        try:
            parts = list(s.unbind())
        except Exception:
            parts = None
    if not parts:
        t = getattr(s, "tensors", None)
        if isinstance(t, (list, tuple)):
            parts = list(t)
    if not parts:
        try:
            parts = list(s)
        except Exception:
            parts = []
    if len(parts) != 2 or not all(torch.is_tensor(x) for x in parts):
        raise ValueError(
            f"[Bruxos H3] {nome}: esperado AV NestedTensor (video+audio); recebido {type(s).__name__}."
        )
    return parts[0], parts[1]


def _pack(base, video, audio):
    out = dict(base or {})
    out["samples"] = _nested_cls()((video.contiguous(), audio.contiguous()))
    out.pop("noise_mask", None)  # mascara de uma geracao terminada nao deve atravessar o cache
    return out


def _snap_latents(n):
    n = max(LATENT_BASE, int(n))
    return LATENTS_PER_GROUP * ((n - LATENT_BASE) // LATENTS_PER_GROUP) + LATENT_BASE


def _latents_to_frames(t):
    t = max(LATENT_BASE, int(t))
    return FRAMES_PER_GROUP * ((t - LATENT_BASE) // LATENTS_PER_GROUP) + FRAME_BASE


def _frames_to_audio(frames):
    return int(round(int(frames) / FPS * AUDIO_FPS))


def _on_grid(t):
    return int(t) >= 2 and int(t) % 5 == 2


def _supported_factors(h, w):
    g = math.gcd(max(1, int(h) // 2), max(1, int(w) // 2))
    return [f for f in range(1, g + 1) if g % f == 0]


def _snap_factor(requested, h, w):
    valid = _supported_factors(h, w)
    return min(valid, key=lambda f: (abs(f - int(requested)), f))


def _downscale(v, requested):
    h, w = int(v.shape[3]), int(v.shape[4])
    factor = _snap_factor(max(1, int(requested)), h, w)
    if factor == 1:
        return v.contiguous(), h, w, 1
    b, c, t = v.shape[:3]
    x = v.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    x = F.interpolate(x, size=(h // factor, w // factor), mode="bilinear", align_corners=False)
    x = x.reshape(b, t, c, h // factor, w // factor).permute(0, 2, 1, 3, 4)
    return x.to(dtype=v.dtype).contiguous(), h // factor, w // factor, factor


def _tail(video, audio, carry_latents, audio_latents=0):
    want = _snap_latents(min(int(carry_latents), int(video.shape[VIDEO_T_DIM])))
    v = video[:, :, -want:, :, :].contiguous()
    frames = _latents_to_frames(want)
    at = int(audio_latents) if int(audio_latents) > 0 else _frames_to_audio(frames)
    at = min(at, int(audio.shape[AUDIO_T_DIM]))
    a = audio[:, :, :, -at:].contiguous()
    return v, a, frames, at


def _append_cond(conditioning, block):
    out = []
    for entry in conditioning:
        meta = entry[1].copy()
        meta["minimax_refs"] = list(meta.get("minimax_refs") or []) + [block]
        out.append([entry[0], meta] + list(entry[2:]))
    return out


def _safe_name(name):
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name or "h3_latent")).strip("._")
    return value[:96] or "h3_latent"


def _root():
    base = folder_paths.get_temp_directory() if folder_paths else os.path.join(os.getcwd(), ".bruxos_temp")
    path = os.path.abspath(os.path.join(base, "bruxos_h3_latent_ssd"))
    os.makedirs(path, exist_ok=True)
    return path


def _job(name):
    return os.path.join(_root(), _safe_name(name))


def _atomic_torch_save(data, path):
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


def _atomic_json(data, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


class BruxosH3LatentInfoSSD:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("frames", "tokens_ref_frame", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    OUTPUT_NODE = True
    DESCRIPTION = "Inspeciona os eixos AV, a grade 5j+2 e fatores espaciais validos."

    def run(self, latent):
        v, a = _unpack(latent, allow_plain=True)
        vt = int(v.shape[2])
        frames = _latents_to_frames(vt)
        factors = _supported_factors(int(v.shape[3]), int(v.shape[4]))
        at = int(a.shape[3]) if a is not None else 0
        tokens = (int(v.shape[3]) // 2) * (int(v.shape[4]) // 2)
        info = (
            f"video={tuple(v.shape)} | audio={tuple(a.shape) if a is not None else 'ausente'}\n"
            f"{vt} latentes -> {frames} frames ({frames/FPS:.3f}s) | grade 5j+2={'sim' if _on_grid(vt) else 'NAO'}\n"
            f"audio T40={at}, esperado={_frames_to_audio(frames)} | tokens/ref/frame={tokens}\n"
            f"downscale validos: {', '.join(map(str, factors))}"
        )
        print("[Bruxos H3 Latent Info]\n" + info, flush=True)
        return frames, tokens, info


class BruxosH3SplitAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent_av": ("LATENT",)}}

    RETURN_TYPES = ("LATENT", "LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("video_latent", "audio_latent", "video_t", "audio_t", "info")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, latent_av):
        v, a = _unpack(latent_av, "latent_av", allow_plain=True)
        if a is None:
            frames = _latents_to_frames(v.shape[2])
            a = torch.zeros((v.shape[0], 32, 2, _frames_to_audio(frames)), dtype=v.dtype, device=v.device)
            nota = " | audio ausente: silencio criado"
        else:
            nota = ""
        info = f"video T={v.shape[2]} (dim 2) | audio T40={a.shape[3]} (dim 3; dim 2=estereo){nota}"
        return {"samples": v.contiguous()}, {"samples": a.contiguous()}, int(v.shape[2]), int(a.shape[3]), info


class BruxosH3PackAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_latent": ("LATENT",)}, "optional": {"audio_latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent_av", "info")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, video_latent, audio_latent=None):
        v, _ = _unpack(video_latent, "video_latent", allow_plain=True)
        frames = _latents_to_frames(v.shape[2])
        want = _frames_to_audio(frames)
        if audio_latent is None:
            a = torch.zeros((v.shape[0], 32, 2, want), dtype=v.dtype, device=v.device)
            nota = "silencio criado"
        else:
            raw = audio_latent["samples"]
            if torch.is_tensor(raw):
                a = raw
            else:
                _, a = _unpack(audio_latent, "audio_latent", allow_plain=False)
            if a.ndim != 4 or int(a.shape[1]) != 32 or int(a.shape[2]) != 2:
                raise ValueError(f"[Bruxos H3 Pack AV] audio esperado [B,32,2,T40], recebido {tuple(a.shape)}")
            have = int(a.shape[3])
            if have > want:
                a, nota = a[:, :, :, :want], f"audio cortado {have}->{want}"
            elif have < want:
                pad = torch.zeros((*a.shape[:3], want - have), dtype=a.dtype, device=a.device)
                a, nota = torch.cat((a, pad), 3), f"audio completado {have}->{want}"
            else:
                nota = "audio ja sincronizado"
        out = _pack(video_latent, v, a.to(device=v.device, dtype=v.dtype))
        return out, f"AV: {v.shape[2]} video latents/{frames} frames + {want} audio latents | {nota}"


class BruxosH3LatentParaReferencia:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "latent_av": ("LATENT",),
            "ativar": ("BOOLEAN", {"default": True}),
            "carry_latents": ("INT", {"default": 12, "min": 2, "max": 512, "step": 5}),
            "incluir_audio": ("BOOLEAN", {"default": True}),
            "carry_video": ("BOOLEAN", {"default": True}),
            "audio_latents": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 40}),
            "ref_downscale": (["none", "2x", "3x", "4x", "6x"], {"default": "2x"}),
        }}

    RETURN_TYPES = ("CONDITIONING", "INT", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "carried_frames", "factor_usado", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = "Anexa o tail latente em minimax_refs sem decode/re-encode pelo VAE."

    def run(self, conditioning, latent_av, ativar, carry_latents, incluir_audio,
            carry_video, audio_latents, ref_downscale):
        v0, a0 = _unpack(latent_av, "latent_av", allow_plain=False)
        v, a, frames, at = _tail(v0, a0, carry_latents, audio_latents)
        factor_req = {"none": 1, "2x": 2, "3x": 3, "4x": 4, "6x": 6}[ref_downscale]
        vv, lh, lw, used = _downscale(v, factor_req)
        if not incluir_audio:
            a, at = None, 0
        if not carry_video:
            if a is None:
                raise ValueError("carry_video e incluir_audio estao desligados: nao existe referencia para anexar.")
            block = {"kind": "audio", "ref_audio_t": at, "audio_latent": a}
        else:
            block = {
                "kind": "video_audio" if a is not None else "video",
                "latent_t": int(vv.shape[2]), "latent_h": lh, "latent_w": lw,
                "ref_audio_t": at, "latent": vv, "audio_latent": a,
            }
        out = _append_cond(conditioning, block) if ativar else conditioning
        tokens = (lh // 2) * (lw // 2) if carry_video else 0
        info = (
            f"{'ATIVA' if ativar else 'bypass'} | tail {v.shape[2]} latents/{frames} frames | "
            f"audio={at if a is not None else 0} | downscale pedido {factor_req}x, usado {used}x | "
            f"{tokens} tokens de referencia/frame"
        )
        print("[Bruxos H3 Latent->Ref] " + info, flush=True)
        return out, frames, used, info


class BruxosH3TailLatenteSalvarSSD:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent_av": ("LATENT",),
            "nome_job": ("STRING", {"default": "h3_latent_124_01"}),
            "indice_bloco": ("INT", {"default": 0, "min": 0, "max": 100000, "forceInput": True}),
            "carry_latents": ("INT", {"default": 12, "min": 2, "max": 512, "step": 5}),
            "audio_latents": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 40}),
            "sobrescrever": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pasta_job", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    OUTPUT_NODE = True

    def run(self, latent_av, nome_job, indice_bloco, carry_latents, audio_latents, sobrescrever):
        v0, a0 = _unpack(latent_av, allow_plain=False)
        v, a, frames, at = _tail(v0, a0, carry_latents, audio_latents)
        pasta = _job(nome_job)
        os.makedirs(pasta, exist_ok=True)
        indice = max(0, int(indice_bloco))
        path = os.path.join(pasta, f"tail_{indice:05d}.pt")
        if os.path.exists(path) and not sobrescrever:
            raise FileExistsError(f"[Bruxos H3 Latent SSD] tail ja existe: {path}")
        start = time.perf_counter()
        payload = {
            "marca": MARCA,
            "video": v.detach().to(device="cpu", dtype=torch.float16).contiguous(),
            "audio": a.detach().to(device="cpu", dtype=torch.float16).contiguous(),
            "frames": frames,
        }
        _atomic_torch_save(payload, path)
        manifest = {
            "marca": MARCA, "job": _safe_name(nome_job), "ultimo_bloco": indice,
            "video_shape": list(v.shape), "audio_shape": list(a.shape),
            "carry_latents": int(v.shape[2]), "carry_frames": frames,
            "audio_latents": at, "dtype_disco": "float16", "atualizado": time.time(),
        }
        _atomic_json(manifest, os.path.join(pasta, "manifesto.json"))
        info = f"tail bloco {indice:05d} salvo: {v.shape[2]}V/{at}A ({frames} frames) em {time.perf_counter()-start:.2f}s | {path}"
        print("[Bruxos H3 Latent SSD] " + info, flush=True)
        return pasta, info


class BruxosH3TailLatenteLerSSD:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "fallback_latent": ("LATENT",),
            "nome_job": ("STRING", {"default": "h3_latent_124_01"}),
            "indice_bloco": ("INT", {"default": 0, "min": 0, "max": 100000, "forceInput": True}),
            "carry_latents_fallback": ("INT", {"default": 12, "min": 2, "max": 512, "step": 5}),
            "exigir_anterior": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("LATENT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("tail_av", "tem_memoria", "info")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, fallback_latent, nome_job, indice_bloco, carry_latents_fallback, exigir_anterior):
        indice = max(0, int(indice_bloco))
        path = os.path.join(_job(nome_job), f"tail_{indice - 1:05d}.pt")
        exists = indice > 0 and os.path.isfile(path)
        if indice > 0 and not exists and exigir_anterior:
            raise FileNotFoundError(f"[Bruxos H3 Latent SSD] rode/salve primeiro o bloco {indice-1}: {path}")
        if exists:
            try:
                data = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                data = torch.load(path, map_location="cpu")
            if data.get("marca") != MARCA:
                raise ValueError(f"cache H3 desconhecido ou antigo: {path}")
            v, a = data["video"].float(), data["audio"].float()
            tail = _pack(fallback_latent, v, a)
            info = f"memoria do bloco {indice-1}: {v.shape[2]}V/{a.shape[3]}A | CPU/SSD | {path}"
        else:
            v0, a0 = _unpack(fallback_latent, allow_plain=False)
            v, a, frames, at = _tail(v0, a0, carry_latents_fallback)
            tail = _pack(fallback_latent, v, a)
            info = f"bloco {indice}: sem memoria anterior; fallback {v.shape[2]}V/{at}A ({frames} frames), referencia deve ficar em bypass"
        print("[Bruxos H3 Latent SSD] " + info, flush=True)
        return tail, bool(exists), info

    @classmethod
    def IS_CHANGED(cls, fallback_latent, nome_job, indice_bloco, carry_latents_fallback, exigir_anterior):
        path = os.path.join(_job(nome_job), f"tail_{max(0, int(indice_bloco))-1:05d}.pt")
        try:
            s = os.stat(path)
            return s.st_mtime_ns, s.st_size, int(indice_bloco)
        except OSError:
            return float("nan") if int(indice_bloco) > 0 else (0, carry_latents_fallback)


class BruxosH3EncontrarDivergencia:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "anterior": ("IMAGE",), "continuacao": ("IMAGE",),
            "buscar_frames": ("INT", {"default": 96, "min": 1, "max": 2048}),
            "tail_anterior": ("INT", {"default": 96, "min": 1, "max": 2048}),
            "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 2.0, "step": 0.001}),
            "downsample": ("INT", {"default": 48, "min": 8, "max": 256, "step": 8}),
            "comparar": (["estrutura", "raw"], {"default": "estrutura"}),
        }}

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("cortar_inicio_B", "erro", "info")
    FUNCTION = "run"
    CATEGORY = CAT

    @staticmethod
    def _prep(img, size, structure):
        x = img[..., :3].mean(-1, keepdim=True).movedim(-1, 1).float()
        x = F.interpolate(x, size=(size, size), mode="area")
        if structure:
            x = (x - x.mean((1, 2, 3), keepdim=True)) / x.std((1, 2, 3), keepdim=True).clamp_min(1e-5)
        return x

    def run(self, anterior, continuacao, buscar_frames, tail_anterior, threshold, downsample, comparar):
        tail = min(int(tail_anterior), int(anterior.shape[0]))
        search = min(int(buscar_frames), int(continuacao.shape[0]))
        a = self._prep(anterior[-tail:], int(downsample), comparar == "estrutura")
        b = self._prep(continuacao[:search], int(downsample), comparar == "estrutura")
        d = (b.unsqueeze(1) - a.unsqueeze(0)).abs().mean((2, 3, 4))
        limit = min(search, tail)
        errs = []
        for k in range(1, limit + 1):
            i = torch.arange(k, device=d.device)
            errs.append(float(d[i, tail - k + i].mean()))
        et = torch.tensor(errs)
        best = int(et.argmin()) + 1
        error = float(et.min())
        median = float(et.median())
        trim = best if error <= float(threshold) else 0
        sep = median / max(error, 1e-8)
        info = f"melhor sobreposicao={best} frames | corte aceito={trim} | erro={error:.5f} | mediana={median:.5f} | separacao={sep:.1f}x"
        if sep < 3.0:
            info += " | AVISO: minimo fraco; valide visualmente"
        print("[Bruxos H3 Divergencia] " + info, flush=True)
        return trim, error, info


class BruxosH3JuntarAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "imagens_a": ("IMAGE",), "imagens_b": ("IMAGE",),
                "cortar_inicio_b": ("INT", {"default": 0, "min": 0, "max": 4096, "forceInput": True}),
                "crossfade_frames": ("INT", {"default": 4, "min": 0, "max": 240}),
            },
            "optional": {"audio_a": ("AUDIO",), "audio_b": ("AUDIO",)},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("imagens", "audio", "info")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, imagens_a, imagens_b, cortar_inicio_b, crossfade_frames, audio_a=None, audio_b=None):
        cut_frames = max(0, int(cortar_inicio_b))
        b = imagens_b[cut_frames:]
        if not len(b):
            raise ValueError("cortar_inicio_b removeu todas as imagens de B")
        if tuple(imagens_a.shape[1:]) != tuple(b.shape[1:]):
            raise ValueError(f"resolucoes diferentes: A {tuple(imagens_a.shape[1:3])}, B {tuple(b.shape[1:3])}")
        n = min(max(0, int(crossfade_frames)), int(imagens_a.shape[0]), int(b.shape[0]))
        if n:
            w = torch.linspace(0, 1, n + 2, device=imagens_a.device)[1:-1].view(-1, 1, 1, 1)
            mid = imagens_a[-n:] * (1 - w) + b[:n].to(imagens_a) * w
            video = torch.cat((imagens_a[:-n], mid, b[n:].to(imagens_a)), 0)
        else:
            video = torch.cat((imagens_a, b.to(imagens_a)), 0)
        audio = audio_a or audio_b
        if audio_a is not None and audio_b is not None:
            sr = int(audio_a["sample_rate"])
            if int(audio_b["sample_rate"]) != sr:
                raise ValueError("sample rates de A e B sao diferentes")
            wa, wb = audio_a["waveform"], audio_b["waveform"].to(audio_a["waveform"])
            wb = wb[:, :, int(round(cut_frames / FPS * sr)):]
            m = min(int(round(n / FPS * sr)), int(wa.shape[-1]), int(wb.shape[-1]))
            if m:
                w = torch.linspace(0, 1, m + 2, device=wa.device)[1:-1].view(1, 1, -1)
                mid = wa[:, :, -m:] * (1 - w) + wb[:, :, :m] * w
                wav = torch.cat((wa[:, :, :-m], mid, wb[:, :, m:]), -1)
            else:
                wav = torch.cat((wa, wb), -1)
            audio = {"waveform": wav, "sample_rate": sr}
        info = f"A={imagens_a.shape[0]} + B={imagens_b.shape[0]} - corte={cut_frames} - fusao={n} -> {video.shape[0]} frames"
        print("[Bruxos H3 Join AV] " + info, flush=True)
        return video, audio, info


class BruxosH3PlanejarFramesRef:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "segundos": ("FLOAT", {"default": 5.0, "min": 0.2, "max": 150.0, "step": 0.1}),
            "largura": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
            "altura": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
            "downscale_desejado": ("INT", {"default": 2, "min": 1, "max": 32}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("frames", "video_latents", "audio_latents", "ref_width", "ref_height", "info")
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, segundos, largura, altura, downscale_desejado):
        wanted = max(5, int(round(float(segundos) * FPS)))
        low = max(5, 5 + 17 * max(0, (wanted - 5) // 17))
        high = min(3600, low + 17)
        frames = min((low, high), key=lambda n: (abs(n - wanted), n))
        vt = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
        at = _frames_to_audio(frames)
        lh, lw = int(altura) // 16, int(largura) // 16
        f = _snap_factor(downscale_desejado, lh, lw)
        rw, rh = (lw // f) * 16, (lh // f) * 16
        info = f"{segundos:.2f}s pedidos -> {frames} frames/{frames/FPS:.3f}s | {vt}V + {at}A | ref {rw}x{rh} ({f}x)"
        return frames, vt, at, rw, rh, info


NODE_CLASS_MAPPINGS = {
    "BruxosH3LatentInfoSSD": BruxosH3LatentInfoSSD,
    "BruxosH3SplitAV": BruxosH3SplitAV,
    "BruxosH3PackAV": BruxosH3PackAV,
    "BruxosH3LatentParaReferencia": BruxosH3LatentParaReferencia,
    "BruxosH3TailLatenteSalvarSSD": BruxosH3TailLatenteSalvarSSD,
    "BruxosH3TailLatenteLerSSD": BruxosH3TailLatenteLerSSD,
    "BruxosH3EncontrarDivergencia": BruxosH3EncontrarDivergencia,
    "BruxosH3JuntarAV": BruxosH3JuntarAV,
    "BruxosH3PlanejarFramesRef": BruxosH3PlanejarFramesRef,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3LatentInfoSSD": "H3 Latent Info AV (Bruxos)",
    "BruxosH3SplitAV": "H3 Split AV Correto (Bruxos)",
    "BruxosH3PackAV": "H3 Pack AV Correto (Bruxos)",
    "BruxosH3LatentParaReferencia": "H3 Tail Latente -> Referencia (Bruxos)",
    "BruxosH3TailLatenteSalvarSSD": "H3 Tail Latente SSD - Salvar (Bruxos)",
    "BruxosH3TailLatenteLerSSD": "H3 Tail Latente SSD - Ler (Bruxos)",
    "BruxosH3EncontrarDivergencia": "H3 Encontrar Divergencia (Bruxos)",
    "BruxosH3JuntarAV": "H3 Juntar AV Decodificado (Bruxos)",
    "BruxosH3PlanejarFramesRef": "H3 Planejar Frames + Ref (Bruxos)",
}
