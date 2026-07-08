"""
ComfyUI-Bruxos-do-VFX — Video I/O
=================================
Dois nodes no estilo do VideoHelperSuite, porem compativeis com o tipo VIDEO
nativo dos nodes "2.0" do ComfyUI:

  * Bruxos Load Video   -> images, video (VIDEO nativo), audio, fps, frame_count, video_info
  * Bruxos Save Video   -> exporta MP4 e/ou sequencia de PNG, criando pastas.

Backends de decode/encode em camadas: PyAV (vem com o ComfyUI) -> imageio-ffmpeg
-> OpenCV. O codigo degrada com elegancia se algum nao estiver disponivel.
"""

import os
import json
import logging
import datetime
from fractions import Fraction

import numpy as np
import torch

# ---- ComfyUI helpers (guardados p/ rodar fora do Comfy em teste) ----------
try:
    import folder_paths
    _HAS_FP = True
except Exception:
    folder_paths = None
    _HAS_FP = False

# tipo VIDEO nativo (nodes 2.0). Import guardado: varia por versao do ComfyUI.
_VIDEO_API = None
try:
    from comfy_api.latest import InputImpl as _InputImpl, Types as _Types
    _VIDEO_API = "latest"
except Exception:
    try:
        from comfy_api.input_impl import VideoFromComponents as _VFC  # type: ignore
        from comfy_api.util import VideoComponents as _VComp           # type: ignore
        _VIDEO_API = "legacy"
    except Exception:
        _VIDEO_API = None

# backends de midia
try:
    import av  # PyAV (vem com o ComfyUI)
    _HAS_AV = True
except Exception:
    _HAS_AV = False
try:
    import imageio
    import imageio_ffmpeg  # noqa: F401
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".gif", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv")


# ===========================================================================
# Helpers de decode
# ===========================================================================
def _list_input_videos():
    if not _HAS_FP:
        return []
    d = folder_paths.get_input_directory()
    out = []
    try:
        for f in os.listdir(d):
            if f.lower().endswith(VIDEO_EXTS) and os.path.isfile(os.path.join(d, f)):
                out.append(f)
    except Exception:
        pass
    return sorted(out)


def _resolve_path(video, video_path):
    if video_path and str(video_path).strip():
        p = str(video_path).strip().strip('"')
        if os.path.isfile(p):
            return p
    if video and _HAS_FP:
        cand = os.path.join(folder_paths.get_input_directory(), video)
        if os.path.isfile(cand):
            return cand
    if video and os.path.isfile(video):
        return video
    raise FileNotFoundError(f"[Bruxos Load Video] video nao encontrado: video={video!r} video_path={video_path!r}")


def _input_preview_ref(video, video_path):
    """Devolve {filename, subfolder, type, format} se o video estiver no diretorio
    de input do ComfyUI (pro preview via /view). Retorna None caso contrario
    (ex.: caminho absoluto fora do input, que o /view nao serve)."""
    if video_path and str(video_path).strip():
        return None
    if not (video and _HAS_FP):
        return None
    try:
        in_dir = folder_paths.get_input_directory()
        cand = os.path.join(in_dir, video)
        if not os.path.isfile(cand):
            return None
        rel = os.path.relpath(cand, in_dir).replace("\\", "/")
        subfolder = os.path.dirname(rel)
        filename = os.path.basename(rel)
        ext = os.path.splitext(filename)[1].lower().lstrip(".") or "mp4"
        return {"filename": filename, "subfolder": subfolder,
                "type": "input", "format": "video/" + ext}
    except Exception:
        return None


def _iter_frames_imageio(path):
    rdr = imageio.get_reader(path, "ffmpeg")
    meta = rdr.get_meta_data()
    fps = float(meta.get("fps", 0) or 0)
    for frame in rdr:
        yield np.asarray(frame)[..., :3], fps
    rdr.close()


def _iter_frames_cv2(path):
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), fps
    cap.release()


def _iter_frames_av(path):
    container = av.open(path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    for frame in container.decode(stream):
        yield frame.to_ndarray(format="rgb24"), fps
    container.close()


def _frame_iterator(path):
    """Itera (frame_rgb_uint8, source_fps). Tenta av -> imageio -> cv2."""
    if _HAS_AV:
        try:
            yield from _iter_frames_av(path); return
        except Exception as e:
            logging.warning(f"[Bruxos] av falhou ({e}); tentando imageio")
    if _HAS_IMAGEIO:
        try:
            yield from _iter_frames_imageio(path); return
        except Exception as e:
            logging.warning(f"[Bruxos] imageio falhou ({e}); tentando cv2")
    if _HAS_CV2:
        yield from _iter_frames_cv2(path); return
    raise RuntimeError("[Bruxos] Nenhum backend de video disponivel (av / imageio-ffmpeg / opencv).")


def _resize_frame(f, cw, ch):
    H, W = f.shape[:2]
    if cw <= 0 and ch <= 0:
        return f
    if cw > 0 and ch > 0:
        tw, th = cw, ch
    elif cw > 0:
        tw = cw; th = max(1, round(H * cw / W))
    else:
        th = ch; tw = max(1, round(W * ch / H))
    if (tw, th) == (W, H):
        return f
    if _HAS_CV2:
        interp = cv2.INTER_AREA if (tw < W or th < H) else cv2.INTER_CUBIC
        return cv2.resize(f, (tw, th), interpolation=interp)
    # fallback torch
    t = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(th, tw), mode="bicubic", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()


def decode_video(path, skip_first_frames=0, frame_load_cap=0, select_every_nth=1,
                 force_rate=0.0, custom_width=0, custom_height=0):
    """Retorna (images_tensor[B,H,W,3] float 0..1, source_fps, out_fps)."""
    select_every_nth = max(1, int(select_every_nth))
    frames = []
    src_fps = 0.0
    kept = 0
    next_tick = 0.0
    step = None
    idx = -1
    for raw, sfps in _frame_iterator(path):
        if sfps:
            src_fps = sfps
        idx += 1
        if idx < skip_first_frames:
            continue
        j = idx - skip_first_frames
        if force_rate and src_fps:
            if step is None:
                step = src_fps / float(force_rate)
            if j < next_tick - 1e-9:
                continue
            next_tick += step
        else:
            if j % select_every_nth != 0:
                continue
        frames.append(_resize_frame(raw, custom_width, custom_height))
        kept += 1
        if frame_load_cap and kept >= frame_load_cap:
            break

    if not frames:
        raise RuntimeError("[Bruxos Load Video] nenhum frame decodificado (cheque skip/cap/path).")

    # garante shape uniforme (caso custom resize off e video tenha mudanca rara)
    h0, w0 = frames[0].shape[:2]
    arr = np.stack([f if f.shape[:2] == (h0, w0) else _resize_frame(f, w0, h0) for f in frames], 0)
    images = torch.from_numpy(arr.astype(np.float32) / 255.0)

    if force_rate:
        out_fps = float(force_rate)
    elif src_fps:
        out_fps = src_fps / select_every_nth
    else:
        out_fps = 0.0
    return images, src_fps, out_fps


def _extract_audio_av(path):
    """Best-effort: AUDIO dict {'waveform':[1,C,N], 'sample_rate'} via PyAV."""
    if not _HAS_AV:
        return None
    try:
        container = av.open(path)
        if not container.streams.audio:
            container.close(); return None
        astream = container.streams.audio[0]
        sr = astream.rate
        chunks = []
        for frame in container.decode(astream):
            chunks.append(frame.to_ndarray())
        container.close()
        if not chunks:
            return None
        data = np.concatenate(chunks, axis=1) if chunks[0].ndim == 2 else np.concatenate(chunks)[None, :]
        wav = torch.from_numpy(np.ascontiguousarray(data)).float()
        if wav.abs().max() > 1.5:  # int PCM
            wav = wav / 32768.0
        return {"waveform": wav.unsqueeze(0), "sample_rate": int(sr)}
    except Exception as e:
        logging.warning(f"[Bruxos] extracao de audio falhou: {e}")
        return None


def _make_video_obj(images, audio, fps):
    """Constroi o objeto VIDEO nativo, se a API existir; senao None."""
    if _VIDEO_API == "latest":
        try:
            comps = _Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps).limit_denominator(100000))
            return _InputImpl.VideoFromComponents(comps)
        except Exception as e:
            logging.warning(f"[Bruxos] VideoFromComponents (latest) falhou: {e}")
    elif _VIDEO_API == "legacy":
        try:
            comps = _VComp(images=images, audio=audio, frame_rate=Fraction(fps).limit_denominator(100000))
            return _VFC(comps)
        except Exception as e:
            logging.warning(f"[Bruxos] VideoFromComponents (legacy) falhou: {e}")
    return None


# ===========================================================================
# NODE: Load Video
# ===========================================================================
class BruxosLoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_input_videos()
        return {
            "required": {
                "video": (files if files else ["(coloque videos em ComfyUI/input)"],
                          {"tooltip": "Seletor de videos da pasta ComfyUI/input. Use o botao de upload pra enviar um novo."}),
            },
            "optional": {
                "video_path": ("STRING", {"default": "", "tooltip": "Caminho absoluto (ex: C:\\\\...\\\\clip.mp4). Tem prioridade sobre o seletor."}),
                "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                                         "tooltip": "Reamostra para esse fps. 0 = mantem o original."}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                                         "tooltip": "0 = mantem. Se so um lado for >0, mantem proporcao."}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                                          "tooltip": "Altura de saida em px. 0 = mantem. Se so um lado for >0, mantem a proporcao."}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1_000_000,
                                           "tooltip": "Maximo de frames a carregar. 0 = todos."}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 1_000_000,
                                              "tooltip": "Pula os N primeiros frames do video."}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000,
                                             "tooltip": "Pega 1 a cada N frames."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "AUDIO", "FLOAT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "video", "audio", "fps", "frame_count", "width", "height", "video_info")
    FUNCTION = "load"
    CATEGORY = "Bruxos do VFX/Video"

    def load(self, video, video_path="", force_rate=0.0, custom_width=0, custom_height=0,
             frame_load_cap=0, skip_first_frames=0, select_every_nth=1):
        path = _resolve_path(video, video_path)
        images, src_fps, out_fps = decode_video(
            path, skip_first_frames, frame_load_cap, select_every_nth,
            force_rate, custom_width, custom_height,
        )
        audio = _extract_audio_av(path)
        video_obj = _make_video_obj(images, audio, out_fps if out_fps > 0 else (src_fps or 24.0))

        B, H, W, _ = images.shape
        info = {
            "source_path": path,
            "source_fps": round(src_fps, 4),
            "output_fps": round(out_fps, 4),
            "frame_count": int(B),
            "width": int(W),
            "height": int(H),
            "has_audio": audio is not None,
        }
        info_json = json.dumps(info, ensure_ascii=False)
        result = (images, video_obj, audio,
                  float(out_fps if out_fps > 0 else src_fps),
                  int(B), int(W), int(H), info_json)

        # UI: infos pro node + ponteiro de preview (so quando vem do diretorio input)
        ui = {"bruxos_info": [info_json]}
        prev = _input_preview_ref(video, video_path)
        if prev is not None:
            ui["bruxos_video"] = [prev]
        return {"ui": ui, "result": result}


# ===========================================================================
# Helpers de encode
# ===========================================================================
def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _write_png_sequence(frames_uint8, folder, prefix="frame", start=1):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, f in enumerate(frames_uint8):
        fp = os.path.join(folder, f"{prefix}_{start + i:05d}.png")
        if _HAS_CV2:
            cv2.imwrite(fp, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        else:
            from PIL import Image
            Image.fromarray(f).save(fp)
        paths.append(fp)
    return paths


_CODEC_MAP = {"h264": "libx264", "h265": "libx265", "vp9": "libvpx-vp9", "prores": "prores_ks"}


def _encode_mp4_imageio(frames_uint8, out_path, fps, codec="h264", crf=19, pix_fmt="yuv420p"):
    lib = _CODEC_MAP.get(codec, "libx264")
    params = []
    if codec in ("h264", "h265"):
        params += ["-crf", str(crf)]
    w = imageio.get_writer(out_path, fps=max(1.0, fps), codec=lib, pixelformat=pix_fmt,
                           macro_block_size=None, ffmpeg_params=params if params else None)
    for f in frames_uint8:
        w.append_data(f)
    w.close()
    return out_path


def _mux_audio(video_path, audio, fps):
    """Anexa audio ao mp4 ja escrito, via ffmpeg CLI. Best-effort."""
    if audio is None:
        return video_path
    try:
        import subprocess, tempfile, wave
        wav = audio["waveform"]
        sr = int(audio["sample_rate"])
        a = wav[0].cpu().numpy()  # [C, N]
        a = np.clip(a, -1, 1)
        pcm = (a.T * 32767.0).astype(np.int16)  # [N, C]
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with wave.open(tmp_wav, "wb") as wf:
            wf.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
            wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        out2 = video_path.rsplit(".", 1)[0] + "_a." + video_path.rsplit(".", 1)[1]
        subprocess.run([_ffmpeg_exe(), "-y", "-i", video_path, "-i", tmp_wav,
                        "-c:v", "copy", "-c:a", "aac", "-shortest", out2],
                       check=True, capture_output=True)
        os.replace(out2, video_path)
        os.remove(tmp_wav)
    except Exception as e:
        logging.warning(f"[Bruxos] mux de audio falhou (video salvo sem audio): {e}")
    return video_path


# ===========================================================================
# NODE: Save Video
# ===========================================================================
class BruxosSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frames a salvar."}),
                "filename_prefix": ("STRING", {"default": "Bruxos/video",
                    "tooltip": "Prefixo/caminho relativo dentro de ComfyUI/output. Subpastas sao criadas."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Quadros por segundo do arquivo final."}),
            },
            "optional": {
                "save_mp4": ("BOOLEAN", {"default": True,
                    "tooltip": "Liga/desliga a exportacao do video."}),
                "codec": (["h264", "h265", "vp9", "prores"], {"default": "h264",
                    "tooltip": "h264/h265 -> .mp4; vp9 -> .webm; prores -> .mov."}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51,
                                "tooltip": "Menor = mais qualidade/maior arquivo (h264/h265)."}),
                "pix_fmt": (["yuv420p", "yuv444p", "yuv422p"], {"default": "yuv420p",
                    "tooltip": "Formato de pixel. yuv420p e o mais compativel com players."}),
                "save_png_sequence": ("BOOLEAN", {"default": False,
                    "tooltip": "Salva tambem a sequencia de PNG (1 arquivo por frame)."}),
                "png_in_subfolder": ("BOOLEAN", {"default": True,
                                "tooltip": "Cria uma pasta dedicada pra sequencia de PNG."}),
                "png_prefix": ("STRING", {"default": "frame",
                    "tooltip": "Prefixo dos arquivos PNG (ex: frame_00001.png)."}),
                "date_subfolder": ("BOOLEAN", {"default": False,
                                "tooltip": "Cria subpasta com a data (YYYY-MM-DD)."}),
                "pingpong": ("BOOLEAN", {"default": False,
                    "tooltip": "Anexa o video invertido no fim (efeito ida-e-volta)."}),
                "audio": ("AUDIO", {"tooltip": "Opcional: trilha de audio pra embutir no MP4 (best-effort)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("mp4_path", "png_folder")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Bruxos do VFX/Video"

    def _frames_uint8(self, images, pingpong):
        arr = (images.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        frames = [arr[i] for i in range(arr.shape[0])]
        if pingpong and len(frames) > 2:
            frames = frames + frames[-2:0:-1]
        return frames

    def _resolve_outdir(self, filename_prefix, date_subfolder):
        base = folder_paths.get_output_directory() if _HAS_FP else os.path.abspath("output")
        sub = os.path.dirname(filename_prefix)
        name = os.path.basename(filename_prefix) or "video"
        out_dir = os.path.join(base, sub)
        if date_subfolder:
            out_dir = os.path.join(out_dir, datetime.date.today().isoformat())
        os.makedirs(out_dir, exist_ok=True)
        return out_dir, name

    def _next_counter(self, out_dir, name):
        n = 1
        for f in os.listdir(out_dir):
            if f.startswith(name + "_") and f.endswith(".mp4"):
                try:
                    n = max(n, int(f[len(name) + 1:].split("_")[0].split(".")[0]) + 1)
                except Exception:
                    pass
        return n

    def save(self, images, filename_prefix, fps, save_mp4=True, codec="h264", crf=19,
             pix_fmt="yuv420p", save_png_sequence=False, png_in_subfolder=True,
             png_prefix="frame", date_subfolder=False, pingpong=False, audio=None):
        out_dir, name = self._resolve_outdir(filename_prefix, date_subfolder)
        counter = self._next_counter(out_dir, name)
        frames = self._frames_uint8(images, pingpong)

        mp4_path = ""
        png_folder = ""
        ui_files = []

        if save_png_sequence:
            if png_in_subfolder:
                png_folder = os.path.join(out_dir, f"{name}_{counter:05d}_pngs")
            else:
                png_folder = out_dir
            _write_png_sequence(frames, png_folder, prefix=png_prefix, start=1)

        if save_mp4:
            ext = "webm" if codec == "vp9" else ("mov" if codec == "prores" else "mp4")
            mp4_path = os.path.join(out_dir, f"{name}_{counter:05d}.{ext}")
            if _HAS_IMAGEIO:
                _encode_mp4_imageio(frames, mp4_path, fps, codec, crf, pix_fmt)
            elif _HAS_AV:
                _encode_mp4_av(frames, mp4_path, fps, codec, crf, pix_fmt)
            else:
                raise RuntimeError("[Bruxos Save Video] sem backend de encode (imageio-ffmpeg ou av).")
            _mux_audio(mp4_path, audio, fps)
            try:
                base = folder_paths.get_output_directory() if _HAS_FP else os.path.abspath("output")
                rel_sub = os.path.relpath(os.path.dirname(mp4_path), base)
                ui_files.append({"filename": os.path.basename(mp4_path),
                                 "subfolder": "" if rel_sub == "." else rel_sub,
                                 "type": "output", "format": f"video/{ext if ext!='mov' else 'quicktime'}"})
            except Exception:
                pass

        logging.info(f"[Bruxos Save Video] mp4={mp4_path or '-'} png={png_folder or '-'} frames={len(frames)}")
        return {"ui": {"gifs": ui_files}, "result": (mp4_path, png_folder)}


def _encode_mp4_av(frames_uint8, out_path, fps, codec="h264", crf=19, pix_fmt="yuv420p"):
    lib = _CODEC_MAP.get(codec, "libx264")
    container = av.open(out_path, mode="w")
    stream = container.add_stream(lib, rate=Fraction(fps).limit_denominator(100000))
    stream.width = frames_uint8[0].shape[1]
    stream.height = frames_uint8[0].shape[0]
    stream.pix_fmt = pix_fmt
    if codec in ("h264", "h265"):
        stream.options = {"crf": str(crf)}
    for f in frames_uint8:
        frame = av.VideoFrame.from_ndarray(f, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return out_path


NODE_CLASS_MAPPINGS = {
    "BruxosLoadVideo": BruxosLoadVideo,
    "BruxosSaveVideo": BruxosSaveVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosLoadVideo": "Load Video (Bruxos)",
    "BruxosSaveVideo": "Save Video (Bruxos)",
}


# --- Descricoes em portugues ---------------------------------------------
BruxosLoadVideo.DESCRIPTION = (
    "Load Video (Bruxos) — importa um video e ja entrega o tipo VIDEO nativo dos "
    "nodes 2.0, alem dos frames.\n"
    "- video: seletor de arquivos da pasta ComfyUI/input (use o botao de upload "
    "pra enviar um arquivo novo).\n"
    "- video_path: caminho absoluto (ex: C:\\\\...\\\\clip.mp4). Tem prioridade "
    "sobre o seletor.\n"
    "- force_rate: reamostra pra esse fps (0 = mantem o original).\n"
    "- custom_width / custom_height: redimensiona (0 = mantem; se so um lado for "
    ">0, mantem a proporcao).\n"
    "- frame_load_cap: maximo de frames a carregar (0 = todos).\n"
    "- skip_first_frames: pula os N primeiros frames.\n"
    "- select_every_nth: pega 1 a cada N frames.\n"
    "SAIDAS: images, video (VIDEO nativo), audio, fps, frame_count, video_info (JSON)."
)

BruxosSaveVideo.DESCRIPTION = (
    "Save Video (Bruxos) — exporta com mais opcoes que o VideoHelperSuite, "
    "criando pastas.\n"
    "- images: frames a salvar.\n"
    "- filename_prefix: prefixo/caminho relativo dentro de ComfyUI/output "
    "(subpastas sao criadas).\n"
    "- fps: quadros por segundo do arquivo.\n"
    "- save_mp4: liga/desliga a exportacao de video.\n"
    "- codec: h264 (.mp4), h265 (.mp4), vp9 (.webm) ou prores (.mov).\n"
    "- crf: qualidade (menor = melhor/maior arquivo) p/ h264/h265.\n"
    "- pix_fmt: formato de pixel (yuv420p e o mais compativel).\n"
    "- save_png_sequence: salva tambem a sequencia de PNG.\n"
    "- png_in_subfolder: cria uma pasta dedicada pra sequencia.\n"
    "- png_prefix: prefixo dos arquivos PNG.\n"
    "- date_subfolder: cria uma subpasta com a data (YYYY-MM-DD).\n"
    "- pingpong: anexa o video invertido no fim (ida e volta).\n"
    "- audio (opcional): trilha pra embutir no MP4 (best-effort).\n"
    "SAIDAS: mp4_path e png_folder (caminhos do que foi salvo)."
)
