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

# ---- rota HTTP que serve os presets do Prompt Guide p/ a extensao JS ----
try:
    from server import PromptServer
    from aiohttp import web
    from .prompt_guide import presets_payload

    @PromptServer.instance.routes.get("/bruxos/prompt_presets")
    async def _bruxos_prompt_presets(request):  # pragma: no cover
        return web.json_response(presets_payload())
except Exception as e:  # pragma: no cover
    import logging
    logging.info(f"[Bruxos do VFX] rota de presets nao registrada (ok fora do server): {e}")

# Extensoes JS (botao de upload, prompt guide, etc.)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
