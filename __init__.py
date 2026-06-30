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

# ---- rota HTTP que serve os presets do Prompt Guide p/ a extensao JS ----
try:
    from server import PromptServer
    from aiohttp import web
    from .prompt_guide import presets_payload

    @PromptServer.instance.routes.get("/bruxos/prompt_presets")
    async def _bruxos_prompt_presets(request):  # pragma: no cover
        return web.json_response(presets_payload())

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
            return web.json_response({"width": w, "height": h, "fps": round(fps, 4),
                                      "frame_count": fc, "duration": round(dur, 4)})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
except Exception as e:  # pragma: no cover
    import logging
    logging.info(f"[Bruxos do VFX] rota de presets nao registrada (ok fora do server): {e}")

# Extensoes JS (botao de upload, prompt guide, etc.)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
