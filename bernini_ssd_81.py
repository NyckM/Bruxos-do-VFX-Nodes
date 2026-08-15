"""
Bernini Infinity SSD 81.

Executa uma geracao longa como uma sequencia de filas independentes. Cada fila:
  * le somente 81 frames do cache-fonte no SSD;
  * le do SSD o tail GERADO pelo bloco anterior;
  * usa esse tail como contexto temporal do Bernini;
  * grava imediatamente o bloco e o novo tail no SSD.

Nao e KV-cache. E a mesma memoria visual explicita usada pelo modo sequential
do Bernini Infinity, mas persistida entre execucoes e sem acumular o video
inteiro em RAM. O cache de saida usa o formato BRUXOS_FRAME_CACHE e pode ser
lido pelo node "Disk Cache -> Window (Bruxos)".
"""

from __future__ import annotations

import copy
import json
import os
import re
import time

import torch

try:
    import folder_paths
except Exception:
    folder_paths = None

from .nodes import BerniniInfinity
from .bruxos_disk_stream import (
    MARCA as FRAME_CACHE_MARCA,
    BruxosDiscoLerJanela,
    _ler_manifesto,
)


CAT = "Bruxos do VFX/Bernini/SSD 81"
JOB_MARCA = "BRUXOS_BERNINI_SSD_81_V1"
BLOCK = 81


def _safe_name(value):
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "bernini_ssd_81")).strip("._")
    return name[:96] or "bernini_ssd_81"


def _root():
    if folder_paths is not None:
        base = folder_paths.get_temp_directory()
    else:
        base = os.path.join(os.getcwd(), ".bruxos_temp")
    path = os.path.abspath(os.path.join(base, "bruxos_bernini_ssd_81"))
    os.makedirs(path, exist_ok=True)
    return path


def _job(name):
    return os.path.join(_root(), _safe_name(name))


def _save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _save_tensor(tensor, path):
    tmp = path + ".tmp"
    torch.save(tensor.contiguous(), tmp)
    os.replace(tmp, path)


def _as_u8(images):
    return (
        images[..., :3].detach().to("cpu", dtype=torch.float32)
        .mul(255.0).round_().clamp_(0, 255).to(torch.uint8).contiguous()
    )


def _load_images(path):
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not torch.is_tensor(value) or value.ndim != 4 or int(value.shape[0]) < 1:
        raise ValueError(f"[Bernini SSD 81] tensor invalido: {path}")
    if value.dtype == torch.uint8:
        return value.to(torch.float32).div_(255.0)
    return value.to(torch.float32)


def _output_manifest(source_manifest, entries, finished, height, width):
    entries = sorted(entries, key=lambda item: int(item["inicio"]))
    frames = max((int(item["fim"]) for item in entries), default=0)
    return {
        "marca": FRAME_CACHE_MARCA,
        "bernini_job_marca": JOB_MARCA,
        "frames": frames,
        "fps": float(source_manifest.get("fps", 0.0)),
        "source_fps": float(source_manifest.get("fps", 0.0)),
        "altura": int(height),
        "largura": int(width),
        "canais": 3,
        "dtype": "uint8",
        "frames_por_bloco": BLOCK,
        "blocos": [
            {"arquivo": item["arquivo"], "inicio": int(item["inicio"]), "fim": int(item["fim"])}
            for item in entries
        ],
        "origem": {"tipo": "BERNINI_SSD_81", "source_cache": source_manifest.get("origem", {})},
        "completo": bool(finished),
        "atualizado": time.time(),
    }


class BruxosBerniniInfinitySSD81(BerniniInfinity):
    @classmethod
    def INPUT_TYPES(cls):
        base = copy.deepcopy(BerniniInfinity.INPUT_TYPES())
        hidden = {
            "source_video", "mode", "chunk_size", "overlap", "max_frames",
            "tail_memory", "tail_frames", "context_jitter", "vary_seed_per_chunk",
        }
        required = {}
        for name, spec in base.get("required", {}).items():
            if name == "source_video":
                required["source_cache"] = (
                    "BRUXOS_FRAME_CACHE",
                    {"forceInput": True, "tooltip": "Cache do video-fonte. O node le somente o bloco atual de 81 frames."},
                )
                required["indice_bloco"] = (
                    "INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                            "tooltip": "0, 1, 2... Cada indice avanca exatamente 81 frames no cache-fonte."},
                )
                required["nome_job"] = (
                    "STRING", {"default": "bernini_long_01",
                               "tooltip": "Identifica no SSD os blocos e tails desta sequencia."},
                )
                required["tail_frames_ssd"] = (
                    "INT", {"default": 9, "min": 1, "max": 81, "step": 4,
                            "tooltip": "Frames finais gerados usados como memoria temporal do proximo bloco. 5/9 = leve; 17 = mais continuidade e menos liberdade."},
                )
                required["exigir_tail_anterior"] = (
                    "BOOLEAN", {"default": True,
                                "tooltip": "No bloco > 0, para com erro se o tail anterior nao existir."},
                )
                required["sobrescrever_bloco"] = (
                    "BOOLEAN", {"default": False,
                                "tooltip": "Permite substituir um bloco ja renderizado deste job."},
                )
                required["variar_seed_por_bloco"] = (
                    "BOOLEAN", {"default": False,
                                "tooltip": "Soma indice_bloco a seed. Off tende a preservar melhor a identidade."},
                )
            elif name not in hidden:
                required[name] = spec
        base["required"] = required

        # O tail SSD ocupa o stream temporal reference_video. Uma referencia
        # inicial ainda pode ser usada no bloco zero; nos demais, o tail manda.
        optional = base.get("optional", {})
        optional.pop("reference_video", None)
        optional["reference_video_inicial"] = (
            "IMAGE", {"tooltip": "Referencia temporal opcional somente para o bloco 0. A partir do bloco 1, o tail SSD assume este papel."},
        )
        base["optional"] = optional
        return base

    RETURN_TYPES = ("IMAGE", "LATENT", "INT", "BRUXOS_FRAME_CACHE", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("bloco", "latent", "frames_reais", "cache_saida", "proximo_bloco", "terminou", "info")
    FUNCTION = "render_ssd"
    CATEGORY = CAT
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Bernini em blocos nativos de 81 frames. Le fonte e tail do SSD, renderiza somente um bloco "
        "por fila e salva o resultado imediatamente. Nao acumula o video longo em RAM/VRAM."
    )

    def render_ssd(
        self,
        source_cache,
        indice_bloco,
        nome_job,
        tail_frames_ssd,
        exigir_tail_anterior,
        sobrescrever_bloco,
        variar_seed_por_bloco,
        reference_video_inicial=None,
        **kwargs,
    ):
        source_manifest = _ler_manifesto(source_cache)
        source_total = int(source_manifest["frames"])
        index = max(0, int(indice_bloco))
        start = index * BLOCK
        if start >= source_total:
            raise IndexError(
                f"[Bernini SSD 81] bloco {index} comeca em {start}, mas a fonte possui "
                f"somente {source_total} frames."
            )

        job_dir = _job(nome_job)
        os.makedirs(job_dir, exist_ok=True)
        block_path = os.path.join(job_dir, f"bloco_{index:06d}.pt")
        tail_path = os.path.join(job_dir, f"tail_{index:06d}.pt")
        previous_block_path = os.path.join(job_dir, f"bloco_{index - 1:06d}.pt")
        previous_tail_path = os.path.join(job_dir, f"tail_{index - 1:06d}.pt")
        manifest_path = os.path.join(job_dir, "manifesto.json")

        if index > 0 and not os.path.isfile(previous_block_path):
            raise FileNotFoundError(
                f"[Bernini SSD 81] falta o bloco {index - 1}: {previous_block_path}. "
                "Os blocos precisam ser renderizados em ordem para o cache de saida nao ter buracos."
            )
        if os.path.isfile(block_path) and not bool(sobrescrever_bloco):
            raise FileExistsError(
                f"[Bernini SSD 81] bloco {index} ja existe: {block_path}. "
                "Ligue sobrescrever_bloco somente se quiser substitui-lo."
            )
        if os.path.isfile(block_path) and bool(sobrescrever_bloco):
            later = os.path.join(job_dir, f"bloco_{index + 1:06d}.pt")
            if os.path.isfile(later):
                raise RuntimeError(
                    f"[Bernini SSD 81] nao e seguro sobrescrever o bloco {index} porque o bloco "
                    f"{index + 1} ja depende do tail antigo. Use outro nome_job ou remova primeiro "
                    "os blocos posteriores."
                )

        tail_context = None
        if index > 0:
            if os.path.isfile(previous_tail_path):
                tail_context = _load_images(previous_tail_path)
            elif bool(exigir_tail_anterior):
                raise FileNotFoundError(
                    f"[Bernini SSD 81] falta o tail do bloco {index - 1}: {previous_tail_path}. "
                    "Rode os blocos em ordem e use o mesmo nome_job."
                )
        elif reference_video_inicial is not None:
            tail_context = reference_video_inicial

        # O leitor preenche o ultimo bloco ate 81 para manter a grade nativa.
        source_images, _, _, ended, fps, read_info = BruxosDiscoLerJanela().run(
            source_cache, start, BLOCK, 0, True, False, True
        )
        real_frames = min(BLOCK, source_total - start)

        call = dict(kwargs)
        call.update({
            "source_video": source_images,
            "mode": "context_window",
            "chunk_size": BLOCK,
            "overlap": 16,
            "max_frames": BLOCK,
            "tail_memory": False,
            "tail_frames": max(1, int(tail_frames_ssd)),
            "context_jitter": False,
            "vary_seed_per_chunk": False,
            "reference_video": tail_context,
        })
        if bool(variar_seed_por_bloco):
            call["seed"] = int(call.get("seed", 0)) + index
        # Mascara longa precisa acompanhar o mesmo recorte temporal da fonte.
        # Sem isto, o Infinity reamostraria a mascara inteira para 81 frames.
        region_mask = call.get("region_mask")
        if torch.is_tensor(region_mask) and region_mask.ndim >= 3 and int(region_mask.shape[0]) > BLOCK:
            if start < int(region_mask.shape[0]):
                mask_block = region_mask[start:start + real_frames]
            else:
                mask_block = region_mask[-1:]
            if int(mask_block.shape[0]) < BLOCK:
                repeats = BLOCK - int(mask_block.shape[0])
                mask_block = torch.cat([mask_block, mask_block[-1:].repeat(repeats, *([1] * (mask_block.ndim - 1)))], 0)
            call["region_mask"] = mask_block

        print(
            f"[Bernini SSD 81] bloco {index} | fonte {start}:{start + real_frames} | "
            f"tail={'SSD' if index > 0 and tail_context is not None else ('inicial' if tail_context is not None else 'nenhum')}",
            flush=True,
        )
        images, latent, _ = super().render(**call)
        images = images[:real_frames].detach().cpu()
        latent_samples = latent.get("samples") if isinstance(latent, dict) else None
        if torch.is_tensor(latent_samples):
            latent_len = ((real_frames - 1) // 4) + 1
            latent = dict(latent)
            latent["samples"] = latent_samples[:, :, :latent_len].detach().cpu()

        u8 = _as_u8(images)
        tail_n = min(int(u8.shape[0]), max(1, int(tail_frames_ssd)))
        _save_tensor(u8, block_path)
        _save_tensor(u8[-tail_n:], tail_path)

        entries = []
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as fp:
                    old = json.load(fp)
                if old.get("marca") == FRAME_CACHE_MARCA and old.get("bernini_job_marca") == JOB_MARCA:
                    entries = list(old.get("blocos", []))
            except Exception:
                entries = []
        entries = [item for item in entries if int(item.get("inicio", -1)) != start]
        entries.append({
            "arquivo": os.path.basename(block_path), "inicio": start,
            "fim": start + real_frames, "altura": int(u8.shape[1]), "largura": int(u8.shape[2]),
        })
        out_manifest = _output_manifest(
            source_manifest, entries, bool(ended), int(u8.shape[1]), int(u8.shape[2])
        )
        _save_json(manifest_path, out_manifest)

        next_index = index + 1
        info = (
            f"bloco {index:05d} | {real_frames}/81 frames | tail {tail_n} no SSD | "
            f"fps {float(fps):.3f} | terminou={'sim' if ended else 'nao'} | "
            f"proximo={next_index} | {job_dir} | {read_info}"
        )
        print(f"[Bernini SSD 81] DONE: {info}", flush=True)
        return images, latent, real_frames, job_dir, next_index, bool(ended), info


NODE_CLASS_MAPPINGS = {
    "BruxosBerniniInfinitySSD81": BruxosBerniniInfinitySSD81,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosBerniniInfinitySSD81": "Bernini Infinity SSD 81 (Bruxos)",
}
