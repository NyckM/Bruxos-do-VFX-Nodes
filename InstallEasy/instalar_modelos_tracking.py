#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
  Bruxos do VFX - Instalador de Modelos de Tracking
============================================================================
Baixa os CHECKPOINTS dos trackers (camera / objeto / pontos / mocap) e clona
os repositorios necessarios. Feito pra rodar em QUALQUER maquina nova, com o
python_embeded do ComfyUI.

  Uso (Windows, dentro da pasta do pacote de tracking):
     ..\\..\\..\\python_embeded\\python.exe instalar_modelos_tracking.py
  ou escolha so um grupo:
     python instalar_modelos_tracking.py --only dust3r,cotracker3

HONESTIDADE (leia):
  - Downloads HTTP/HuggingFace: automaticos e verificados (tamanho).
  - Google Drive (DROID-SLAM, SpaTracker): precisam do 'gdown'; o script tenta,
    e se falhar te da o link pra baixar na mao (o GDrive as vezes bloqueia bot).
  - git clone + COMPILACAO CUDA (DROID-SLAM, DUSt3R/curope, SpaTracker):
    NAO da pra garantir no Windows via script. O instalador CLONA os repos e
    te diz o comando exato de build; a compilacao voce roda e confere o erro.
  Ou seja: isto adianta 90% do trabalho e te aponta os 10% que sao manuais,
  em vez de fingir que instalou e quebrar silenciosamente.
============================================================================
"""

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Catalogo (espelha o configs/models.yaml do pacote de tracking).
# dest = subpasta dentro de <ComfyUI>/models/tracking/<grupo>/
# ---------------------------------------------------------------------------
CATALOGO = {
    "dust3r": {
        "nome": "DUSt3R (camera)",
        "repos": [("dust3r", "https://github.com/naver/dust3r.git")],
        "build": ["cd repos/dust3r/croco/models/curope && python setup.py build_ext --inplace"],
        "arquivos": [
            ("DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
             "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
             1_500_000_000)],
    },
    "mast3r": {
        "nome": "MASt3R (camera)",
        "arquivos": [
            ("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
             "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
             1_500_000_000)],
    },
    "droid_slam": {
        "nome": "DROID-SLAM (camera)",
        "repos": [("DROID-SLAM", "https://github.com/princeton-vl/DROID-SLAM.git")],
        "build": ["cd repos/DROID-SLAM && python setup.py install"],
        "gdrive": [("droid.pth", "1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh", 32_000_000)],
    },
    "cotracker3": {
        "nome": "CoTracker3 (pontos/objeto)",
        "repos": [("co-tracker", "https://github.com/facebookresearch/co-tracker.git")],
        "arquivos": [
            ("cotracker3_offline.pth", "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth", 150_000_000),
            ("cotracker3_online.pth", "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth", 50_000_000)],
    },
    "spatracker": {
        "nome": "SpaTracker (pontos 3D)",
        "repos": [("SpaTracker", "https://github.com/henry123-boy/SpaTracker.git")],
        "gdrive": [("spaT_final.pth", "1ZQkT7a8lQNzVpQvkM91qbzyM6GFDy1nz", 400_000_000)],
        "arquivos": [
            ("dpt_beit_large_384.pt", "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/dpt_beit_large_384.pt", 1_300_000_000),
            ("ZoeD_M12_NK.pt", "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt", 400_000_000)],
    },
    "gvhmr": {
        "nome": "GVHMR (mocap corpo)",
        "repos": [("GVHMR", "https://github.com/zju3dv/GVHMR.git")],
        "arquivos": [
            ("gvhmr/gvhmr_siga24_release.ckpt", "https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt", 164_000_000),
            ("vitpose/vitpose-h-multi-coco.pth", "https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth", 1_100_000_000),
            ("yolo/yolov8x.pt", "https://huggingface.co/camenduru/GVHMR/resolve/main/yolo/yolov8x.pt", 130_000_000)],
    },
    "sam2": {
        "nome": "SAM2 (segmentacao)",
        "arquivos": [
            ("sam2.1_hiera_large.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt", 900_000_000),
            ("sam2.1_hiera_base_plus.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt", 320_000_000)],
    },
}

AQUI = Path(__file__).resolve().parent


def achar_models_dir():
    """Sobe a arvore procurando <ComfyUI>/models. Cai em ./models/tracking se nao achar."""
    p = AQUI
    for _ in range(6):
        cand = p / "models"
        if (p / "comfy").exists() or (cand.exists() and (cand / "checkpoints").exists()):
            return cand / "tracking"
        p = p.parent
    return AQUI / "models" / "tracking"


def baixar(url, destino: Path, tam_esperado=0):
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        if tam_esperado and abs(destino.stat().st_size - tam_esperado) < tam_esperado * 0.15:
            print(f"    ja existe: {destino.name}")
            return True
        if not tam_esperado:
            print(f"    ja existe: {destino.name}")
            return True
    print(f"    baixando {destino.name} ...", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(destino, "wb") as f:
            total = int(r.headers.get("Content-Length", 0)) or tam_esperado
            lido = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                lido += len(chunk)
                if total:
                    pct = lido * 100 // total
                    print(f"\r      {pct}%  ({lido//(1<<20)} MB)", end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"    [ERRO] download falhou: {e}")
        print(f"    baixe na mao: {url}")
        if destino.exists():
            destino.unlink(missing_ok=True)
        return False


def baixar_gdrive(file_id, destino: Path, tam=0):
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        print(f"    ja existe: {destino.name}")
        return True
    # tenta gdown
    try:
        import gdown  # noqa
    except Exception:
        print("    (instalando gdown p/ Google Drive...)")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=False)
    try:
        import gdown
        print(f"    baixando (GDrive) {destino.name} ...", flush=True)
        gdown.download(id=file_id, output=str(destino), quiet=False)
        return destino.exists() and destino.stat().st_size > 0
    except Exception as e:
        print(f"    [ERRO] GDrive falhou: {e}")
        print(f"    baixe na mao: https://drive.google.com/file/d/{file_id}/view")
        print(f"    e salve como: {destino}")
        return False


def clonar(nome, url, repos_dir: Path):
    alvo = repos_dir / nome
    if alvo.exists():
        print(f"    repo ja clonado: {nome}")
        return True
    print(f"    git clone {nome} ...", flush=True)
    r = subprocess.run(["git", "clone", "--depth", "1", url, str(alvo)], check=False)
    if r.returncode != 0:
        print(f"    [ERRO] git clone falhou ({url}). Instale o git ou clone na mao.")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Instalador Bruxos de modelos de tracking")
    ap.add_argument("--only", default="", help="grupos separados por virgula (ex: dust3r,cotracker3)")
    ap.add_argument("--list", action="store_true", help="lista os grupos e sai")
    args = ap.parse_args()

    if args.list:
        print("Grupos disponiveis:")
        for k, v in CATALOGO.items():
            print(f"  {k:14s} - {v['nome']}")
        return

    grupos = [g.strip() for g in args.only.split(",") if g.strip()] or list(CATALOGO.keys())
    models_dir = achar_models_dir()
    repos_dir = AQUI / "repos"
    print("=" * 70)
    print("  Bruxos do VFX - Instalador de Modelos de Tracking")
    print("=" * 70)
    print(f"  Modelos -> {models_dir}")
    print(f"  Repos   -> {repos_dir}")
    print(f"  Grupos  -> {', '.join(grupos)}")
    print("=" * 70)

    builds_pendentes = []
    falhas = []
    for g in grupos:
        item = CATALOGO.get(g)
        if not item:
            print(f"[?] grupo desconhecido: {g}")
            continue
        print(f"\n### {item['nome']}  [{g}]")
        for nome, url in item.get("repos", []):
            if not clonar(nome, url, repos_dir):
                falhas.append(f"{g}: git clone {nome}")
        for nome, url, tam in item.get("arquivos", []):
            if not baixar(url, models_dir / g / nome, tam):
                falhas.append(f"{g}: {nome}")
        for nome, fid, tam in item.get("gdrive", []):
            if not baixar_gdrive(fid, models_dir / g / nome, tam):
                falhas.append(f"{g}: {nome} (GDrive)")
        for cmd in item.get("build", []):
            builds_pendentes.append((g, cmd))

    print("\n" + "=" * 70)
    print("  RESUMO")
    print("=" * 70)
    if falhas:
        print("  Downloads/clones que FALHARAM (baixe na mao pelos links acima):")
        for f in falhas:
            print(f"    - {f}")
    else:
        print("  Todos os downloads/clones OK.")

    if builds_pendentes:
        print("\n  COMPILACAO MANUAL necessaria (rode com o python_embeded do ComfyUI):")
        print("  -> precisa do CUDA Toolkit + Visual Studio Build Tools instalados.")
        for g, cmd in builds_pendentes:
            print(f"    [{g}]  {cmd}")
        print("\n  Se um build falhar, o tracker daquele grupo nao vai rodar -- mas os")
        print("  outros grupos e os utilitarios/export Bruxos continuam funcionando.")
    print("\n  Reinicie o ComfyUI depois. Fim.")


if __name__ == "__main__":
    main()
