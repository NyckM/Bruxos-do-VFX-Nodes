from .nodes import (
    NODE_CLASS_MAPPINGS as _N1,
    NODE_DISPLAY_NAME_MAPPINGS as _D1,
)

NODE_CLASS_MAPPINGS = dict(_N1)
NODE_DISPLAY_NAME_MAPPINGS = dict(_D1)


def _merge(modname):
    try:
        mod = __import__(f"{__name__}.{modname}", fromlist=["*"])
        NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception as e:  # pragma: no cover
        import logging
        logging.warning(f"[Bruxos do VFX] modulo '{modname}' nao carregou: {e}")


_merge("video_nodes")
_merge("prompt_guide")
_merge("loaders")
_merge("utility_nodes")
_merge("video_compare")
_merge("mask_bbox_preview")
_merge("simplifier_nodes")
_merge("points_editor")
_merge("timing_nodes")  # cronometro passthrough (fallback ao selo automatico)
_merge("tracking_nodes")  # camera/objeto/pontos — puxa os trackers instalados (estilo Bernini)
_merge("tracking_export")  # exporta trajetoria de camera pro VFX (JSON + Nuke .chan)
_merge("tracking_visualizer")  # desenha os tracks sobre o video (destino visual dos pontos)
_merge("facefusion_nodes")  # face swap local (ONNX) — so carrega se onnxruntime/opencv presentes
_merge("mocha_nodes")  # MoCha (Wan/WanVideoWrapper): embeds com fix 4n+1 + info de custo
_merge("tile_nodes")  # Tiles: corta/costura por contagem (substitui o subgraph Tile Settings)
_merge("wan_tiled")   # Wan Tiled Sampler: ladrilho FUNDIDO a cada passo (1 node, sem For Loop)
_merge("bernini_tiled")  # Bernini Infinity Tiled: ladrilho em PIXELS c/ costura viva (qualquer funcao, resolucao maior)

# ---- rota HTTP que serve os presets do Prompt Guide p/ a extensao JS ----
try:
    from server import PromptServer
    from aiohttp import web
    from .prompt_guide import presets_payload

    @PromptServer.instance.routes.get("/bruxos/prompt_presets")
    async def _bruxos_prompt_presets(request):  # pragma: no cover
        return web.json_response(presets_payload())

    # ---- preview JA CORTADO (estilo VHS advanced): re-renderiza o trecho ----
    @PromptServer.instance.routes.get("/bruxos/video_preview")
    async def _bruxos_video_preview(request):  # pragma: no cover
        import os, asyncio, hashlib
        import folder_paths
        q = request.query
        filename = q.get("filename", "")
        ftype = q.get("type", "input")
        subfolder = q.get("subfolder", "")
        def _i(k, d=0):
            try: return int(float(q.get(k, d)))
            except Exception: return d
        def _f(k, d=0.0):
            try: return float(q.get(k, d))
            except Exception: return d
        skip = max(0, _i("skip_first_frames", 0))
        cap = max(0, _i("frame_load_cap", 0))
        nth = max(1, _i("select_every_nth", 1))
        rate = max(0.0, _f("force_rate", 0.0))
        maxside = max(64, _i("maxside", 360))
        try:
            if ftype == "output":
                base = folder_paths.get_output_directory()
            elif ftype == "temp":
                base = folder_paths.get_temp_directory()
            else:
                base = folder_paths.get_input_directory()
            path = os.path.abspath(os.path.join(base, subfolder, filename))
            if not path.startswith(os.path.abspath(base)) or not os.path.isfile(path):
                return web.json_response({"error": "arquivo invalido"}, status=400)

            mtime = os.path.getmtime(path)
            key = hashlib.md5(f"{path}|{mtime}|{skip}|{cap}|{nth}|{rate}|{maxside}".encode()).hexdigest()[:16]
            tmp = folder_paths.get_temp_directory()
            os.makedirs(tmp, exist_ok=True)
            out = os.path.join(tmp, f"bruxos_prev_{key}.mp4")

            if not os.path.isfile(out):
                def _render():
                    import cv2, numpy as np
                    cap_cv = cv2.VideoCapture(path)
                    src_fps = float(cap_cv.get(cv2.CAP_PROP_FPS)) or 24.0
                    frames = []
                    idx = -1; kept = 0; next_tick = 0.0; step = None
                    HARD = 900  # limite de frames do preview
                    while True:
                        ok, raw = cap_cv.read()
                        if not ok: break
                        idx += 1
                        if idx < skip: continue
                        j = idx - skip
                        if rate and src_fps:
                            if step is None: step = src_fps / rate
                            if j < next_tick - 1e-9: continue
                            next_tick += step
                        else:
                            if j % nth != 0: continue
                        h, w = raw.shape[:2]
                        sc = maxside / max(h, w)
                        if sc < 1.0:
                            raw = cv2.resize(raw, (max(2, int(w*sc)), max(2, int(h*sc))), interpolation=cv2.INTER_AREA)
                        # par (yuv420p exige dimensoes pares)
                        hh, ww = raw.shape[:2]
                        if hh % 2 or ww % 2:
                            raw = raw[:hh - (hh % 2), :ww - (ww % 2)]
                        frames.append(raw)
                        kept += 1
                        if cap and kept >= cap: break
                        if kept >= HARD: break
                    cap_cv.release()
                    if not frames: return False
                    out_fps = rate if rate else (src_fps / nth if src_fps else 24.0)
                    out_fps = max(1.0, out_fps)
                    hh, ww = frames[0].shape[:2]
                    try:
                        import imageio
                        with imageio.get_writer(out, fps=out_fps, codec="libx264",
                                                quality=6, macro_block_size=None,
                                                ffmpeg_params=["-pix_fmt", "yuv420p"]) as wr:
                            for fr in frames:
                                wr.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                        return True
                    except Exception:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        vw = cv2.VideoWriter(out, fourcc, out_fps, (ww, hh))
                        for fr in frames: vw.write(fr)
                        vw.release()
                        return os.path.isfile(out)
                ok = await asyncio.get_event_loop().run_in_executor(None, _render)
                if not ok or not os.path.isfile(out):
                    return web.json_response({"error": "sem frames apos corte"}, status=400)

            return web.FileResponse(out, headers={"Content-Type": "video/mp4",
                                                   "Cache-Control": "no-cache"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @PromptServer.instance.routes.get("/bruxos/video_probe")
    async def _bruxos_video_probe(request):  # pragma: no cover
        """Le frames/resolucao/fps/duracao de um video do diretorio input/output/temp,
        pra preencher as infos no node Load Video assim que o video e escolhido."""
        import os
        import folder_paths
        filename = request.query.get("filename", "")
        ftype = request.query.get("type", "input")
        subfolder = request.query.get("subfolder", "")
        try:
            if ftype == "output":
                base = folder_paths.get_output_directory()
            elif ftype == "temp":
                base = folder_paths.get_temp_directory()
            else:
                base = folder_paths.get_input_directory()
            path = os.path.abspath(os.path.join(base, subfolder, filename))
            if not path.startswith(os.path.abspath(base)) or not os.path.isfile(path):
                return web.json_response({"error": "arquivo invalido"}, status=400)
            w = h = fc = 0
            fps = 0.0
            try:
                import cv2
                cap = cv2.VideoCapture(path)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception:
                pass
            dur = (fc / fps) if fps else 0.0

            # --- calcula o recorte (skip / nth / cap / force_rate) igual ao decode_video ---
            def _int(q, d=0):
                try:
                    return int(float(request.query.get(q, d)))
                except Exception:
                    return d
            def _float(q, d=0.0):
                try:
                    return float(request.query.get(q, d))
                except Exception:
                    return d
            skip = max(0, _int("skip_first_frames", 0))
            nth = max(1, _int("select_every_nth", 1))
            cap = max(0, _int("frame_load_cap", 0))
            frate = max(0.0, _float("force_rate", 0.0))

            avail = max(0, fc - skip)
            if frate and fps:
                avail = int(round(avail * (frate / fps)))
            kept = 0 if avail <= 0 else ((avail - 1) // nth + 1)
            if cap:
                kept = min(kept, cap)
            out_fps = frate if frate else (fps / nth if fps else 0.0)
            trim_dur = (kept / out_fps) if out_fps else 0.0

            return web.json_response({
                "width": w, "height": h, "fps": round(fps, 4),
                "frame_count": fc, "duration": round(dur, 4),
                "trim_frames": int(kept), "trim_fps": round(out_fps, 4),
                "trim_duration": round(trim_dur, 4),
                "skip_first_frames": skip, "select_every_nth": nth,
                "frame_load_cap": cap,
                "start_time": round((skip / fps) if fps else 0.0, 4),
                # fracoes (0..1) do video: robustas a divergencia de fps no navegador
                "start_frac": round((skip / fc) if fc else 0.0, 6),
                "end_frac": round(min(1.0, (skip + kept * nth) / fc) if fc else 1.0, 6),
                # span de tempo do arquivo original consumido (p/ o preview parar no cap)
                "end_time": round(((skip + (kept * nth if not frate else avail * nth)) / fps) if fps else 0.0, 4),
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
except Exception as e:  # pragma: no cover
    import logging
    logging.info(f"[Bruxos do VFX] rota de presets nao registrada (ok fora do server): {e}")

# Banner de inicializacao (logo Bruxos em ASCII verde/roxo)
try:
    from .banner import print_banner
    print_banner(node_count=len(NODE_CLASS_MAPPINGS), version="0.19.1")
except Exception:
    pass

# Extensoes JS (botao de upload, prompt guide, etc.)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
