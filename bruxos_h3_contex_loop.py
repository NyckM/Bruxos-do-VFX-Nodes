"""Pontes tile-aware entre Bruxos Video Tiler e MiniMax H3 Contex Loop.

O pacote GPL Contex Loop continua instalado separadamente e conserva toda a
logica de patch/conditioning. Este modulo prepara janelas e recorta o tail
completo nas mesmas regioes espaciais usadas pelo render em ladrilhos.
"""

from __future__ import annotations

from .bruxos_disk_stream import BruxosDiscoLerJanela, BruxosVideoParaDisco
from .bruxos_video_tiler import ler_config


CAT = "Bruxos do VFX/MiniMax H3/Contex Loop"


def _state(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict) or "plan" not in value or "index" not in value:
        raise ValueError("[Bruxos H3 Contex] state invalido; ligue Current Shot -> state.")
    return value


class BruxosH3ChainSourceWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "state": ("H3_CHAIN_STATE", {"forceInput": True}),
            "modo": (["SSD", "RAM"], {"default": "SSD", "tooltip":
                     "SSD le somente a janela atual. RAM recorta um IMAGE com o video inteiro."}),
            "prefetch": ("BOOLEAN", {"default": True}),
            "pin_memory": ("BOOLEAN", {"default": False}),
            "completar_ultimo": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "cache": ("BRUXOS_FRAME_CACHE", {"forceInput": True, "lazy": True}),
            "imagens_ram": ("IMAGE", {"lazy": True}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("frames", "inicio_fonte", "length", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = ("Seleciona SSD ou RAM e entrega somente a janela fonte exata "
                   "calculada pelo plano recursivo H3.")

    def check_lazy_status(self, state, modo, prefetch=True, pin_memory=False,
                          completar_ultimo=True, cache=None, imagens_ram=None):
        if str(modo).upper() == "RAM":
            return ["imagens_ram"] if imagens_ram is None else []
        return ["cache"] if cache is None else []

    def run(self, state, modo, prefetch=True, pin_memory=False, completar_ultimo=True,
            cache=None, imagens_ram=None):
        st = _state(state)
        index = int(st["index"])
        shot = st["plan"]["shots"][index - 1]
        inicio = int(shot["generation_start_frame"])
        length = int(shot["raw_frames"])
        overlap = int(st["plan"]["compatibility"]["context_length"])
        if str(modo).upper() == "RAM":
            if imagens_ram is None:
                raise ValueError("[Bruxos H3 Contex] modo RAM requer imagens_ram.")
            if getattr(imagens_ram, "ndim", 0) != 4 or int(imagens_ram.shape[0]) < 1:
                raise ValueError("[Bruxos H3 Contex] imagens_ram precisa ser IMAGE [T,H,W,C].")
            total = int(imagens_ram.shape[0])
            start = min(max(0, inicio), max(0, total - 1))
            end = min(total, start + length)
            images = imagens_ram[start:end]
            reais = int(images.shape[0])
            preenchidos = 0
            if completar_ultimo and reais < length:
                preenchidos = length - reais
                images = __import__("torch").cat(
                    [images, images[-1:].repeat(preenchidos, 1, 1, 1)], dim=0)
            ended = end >= total
            fps = 24.0
            read_info = (f"RAM {start}:{end} ({reais} reais) de {total}" +
                         (f" | +{preenchidos} repetidos" if preenchidos else ""))
        else:
            if cache is None:
                raise ValueError("[Bruxos H3 Contex] modo SSD requer cache.")
            images, _next, total, ended, fps, read_info = BruxosDiscoLerJanela().run(
                cache, inicio, length, overlap, bool(prefetch), bool(pin_memory),
                bool(completar_ultimo))
        info = (f"cena {index} | fonte {inicio}:{inicio + length} | length={length} | "
                f"modo={str(modo).upper()} | context={overlap} | total={total} | "
                f"fps={fps:.3f} | terminou={ended}\n{read_info}")
        return images, inicio, length, info


class BruxosVideoParaDiscoLazy(BruxosVideoParaDisco):
    """Mesma cache SSD, mas so executa quando uma entrada lazy realmente pede."""
    OUTPUT_NODE = False
    DESCRIPTION = (BruxosVideoParaDisco.DESCRIPTION +
                   " Variante para switches lazy: nao executa como output independente.")


class BruxosH3ChainTileContextPrep:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "state": ("H3_CHAIN_STATE", {"forceInput": True}),
            "fallback_frames": ("IMAGE",),
            "tile_config": ("TILE_CONFIG", {"forceInput": True}),
        }}

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("context_tiles", "continuacao", "info")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = "Recorta o tail completo anterior em um tail local para cada tile."

    def run(self, state, fallback_frames, tile_config):
        st = _state(state)
        index = int(st["index"])
        previous = st.get("previous_frames")
        continuation = index > 1
        if continuation and previous is None:
            raise ValueError("[Bruxos H3 Contex] continuacao sem previous_frames no checkpoint.")
        source = previous if continuation else fallback_frames
        W, H, _lw, _lh, _ox, _oy, _m, specs = ler_config(tile_config)
        if int(source.shape[1]) != H or int(source.shape[2]) != W:
            import torch.nn.functional as F
            source = F.interpolate(
                source.permute(0, 3, 1, 2).float(), size=(H, W),
                mode="bicubic", align_corners=False,
            ).permute(0, 2, 3, 1).clamp(0, 1).to(source.dtype)
        tiles = [source[:, t["y"]:t["y"] + t["h"],
                        t["x"]:t["x"] + t["w"], :]
                 for t in specs]
        origin = "tail costurado anterior" if continuation else "fallback da cena 1 (bypass)"
        info = f"cena {index} | {origin} -> {len(tiles)} tails locais"
        print(f"[Bruxos H3 Contex] {info}", flush=True)
        return tiles, continuation, info


class BruxosH3ChainTileContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "state": ("H3_CHAIN_STATE", {"forceInput": True}),
            "conditioning": ("CONDITIONING",),
            "vae": ("VAE",),
            "latent": ("LATENT",),
            "context_frames": ("IMAGE",),
        }}

    RETURN_TYPES = ("CONDITIONING", "INT", "BOOLEAN")
    RETURN_NAMES = ("conditioning", "trim_frames", "continuacao")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = "Aplica o Motion Context GPL ao tail do mesmo tile; cena 1 passa em bypass."

    def run(self, state, conditioning, vae, latent, context_frames):
        st = _state(state)
        index = int(st["index"])
        if index == 1:
            return conditioning, 0, False
        try:
            import nodes as comfy_nodes
            engine = comfy_nodes.NODE_CLASS_MAPPINGS["MiniMaxH3MotionContextEngine"]
        except Exception as exc:
            raise RuntimeError(
                "[Bruxos H3 Contex] reinicie o ComfyUI e confira a instalacao de "
                "ComfyUI-MiniMaxH3-Contex-Loop."
            ) from exc
        cfg = st["plan"]["compatibility"]
        use_latent_audio = cfg["audio_mode"] in ("generated_audio", "source_plus_timeline")
        previous_latent = st.get("previous_latent") if use_latent_audio else None
        if use_latent_audio and previous_latent is None:
            raise ValueError("[Bruxos H3 Contex] continuacao de audio sem previous_latent.")
        out, trim = engine().apply(
            conditioning=conditioning,
            vae=vae,
            latent=latent,
            context_frames=context_frames,
            context_length=int(cfg["context_length"]),
            encode_mode=str(cfg["encode_mode"]),
            anchor_mode=str(cfg["anchor_mode"]),
            crop=str(cfg["crop"]),
            audio_context_length=int(cfg["audio_context_length"]),
            audio_mode="timeline",
            context_latent=previous_latent,
        )
        return out, int(trim), True


class _PrimeiroDaLista:
    INPUT_IS_LIST = (True,)

    @staticmethod
    def _first(values):
        if not isinstance(values, (list, tuple)):
            return values
        if not values:
            raise ValueError("[Bruxos H3 Contex] lista vazia.")
        return values[0]


class BruxosPrimeiroLatent(_PrimeiroDaLista):
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"lista": ("LATENT",)}}
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("primeiro",)
    FUNCTION = "run"
    CATEGORY = CAT
    def run(self, lista): return (self._first(lista),)


class BruxosPrimeiraImagem(_PrimeiroDaLista):
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"lista": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("primeira",)
    FUNCTION = "run"
    CATEGORY = CAT
    def run(self, lista): return (self._first(lista),)


class BruxosPrimeiroAudio(_PrimeiroDaLista):
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"lista": ("AUDIO",)}}
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("primeiro",)
    FUNCTION = "run"
    CATEGORY = CAT
    def run(self, lista): return (self._first(lista),)


class BruxosPrimeiroInt(_PrimeiroDaLista):
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"lista": ("INT",)}}
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("primeiro",)
    FUNCTION = "run"
    CATEGORY = CAT
    def run(self, lista): return (int(self._first(lista)),)


NODE_CLASS_MAPPINGS = {
    "BruxosVideoParaDiscoLazy": BruxosVideoParaDiscoLazy,
    "BruxosH3ChainSourceWindow": BruxosH3ChainSourceWindow,
    "BruxosH3ChainTileContextPrep": BruxosH3ChainTileContextPrep,
    "BruxosH3ChainTileContext": BruxosH3ChainTileContext,
    "BruxosPrimeiroLatent": BruxosPrimeiroLatent,
    "BruxosPrimeiraImagem": BruxosPrimeiraImagem,
    "BruxosPrimeiroAudio": BruxosPrimeiroAudio,
    "BruxosPrimeiroInt": BruxosPrimeiroInt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosVideoParaDiscoLazy": "Video -> Disk Cache Lazy (Bruxos)",
    "BruxosH3ChainSourceWindow": "H3 Loop - Janela Fonte SSD (Bruxos)",
    "BruxosH3ChainTileContextPrep": "H3 Loop - Tail por Tile (Bruxos)",
    "BruxosH3ChainTileContext": "H3 Loop - Motion Context por Tile (Bruxos)",
    "BruxosPrimeiroLatent": "Lista - Primeiro Latent (Bruxos)",
    "BruxosPrimeiraImagem": "Lista - Primeira Imagem (Bruxos)",
    "BruxosPrimeiroAudio": "Lista - Primeiro Audio (Bruxos)",
    "BruxosPrimeiroInt": "Lista - Primeiro INT (Bruxos)",
}
