# -*- coding: utf-8 -*-
"""
Bruxos do VFX - baixador de modelos
===================================
Chamado pelo install.bat / install.sh. Faz o que o huggingface-cli NAO faz
direito: coloca cada arquivo NA PASTA CERTA e PLANO (o CLI preservaria a
subpasta do repo, ex. 'diffusion_models/...', e o ComfyUI nao acharia).

- Idempotente: se o arquivo ja existe com tamanho > 0, PULA (pode rodar de novo).
- Retoma download interrompido (o hf_hub_download ja faz cache/resume).
- --list  : so mostra o que baixaria (nao baixa nada)
- --skip-optional : nao baixa os opcionais (GGUF)
"""

import argparse
import os
import shutil
import sys

# (repo_id, arquivo_no_repo, pasta_destino, nome_final, obrigatorio, descricao)
MODELS = [
    # --- Bernini-R INT8 ConvRot (RECOMENDADO: rapido, cabe na VRAM) ------------
    ("Comfy-Org/Bernini-R",
     "diffusion_models/wan2.2_bernini_r_high_noise_int8_convrot.safetensors",
     "diffusion_models", "wan2.2_bernini_r_high_noise_int8_convrot.safetensors",
     True, "Bernini-R HIGH noise (INT8 ConvRot)"),
    ("Comfy-Org/Bernini-R",
     "diffusion_models/wan2.2_bernini_r_low_noise_int8_convrot.safetensors",
     "diffusion_models", "wan2.2_bernini_r_low_noise_int8_convrot.safetensors",
     True, "Bernini-R LOW noise (INT8 ConvRot)"),

    # --- LoRAs de aceleracao LightX2V 4 steps (par high/low do Bernini-R) ------
    ("rzgar/Bernini-R-LightX2V-4step-loras",
     "Bernini-R_LightX2V_high_noise.safetensors",
     "loras", "Bernini-R_LightX2V_high_noise.safetensors",
     True, "LoRA LightX2V 4-step HIGH"),
    ("rzgar/Bernini-R-LightX2V-4step-loras",
     "Bernini-R_LightX2V_low_noise.safetensors",
     "loras", "Bernini-R_LightX2V_low_noise.safetensors",
     True, "LoRA LightX2V 4-step LOW"),

    # --- Text encoder ---------------------------------------------------------
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
     "text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
     True, "Text encoder umt5-xxl (fp8)"),

    # --- VAE de VIDEO do Wan (ATENCAO: NAO use um VAE 'imageonly'/'upscale2x',
    #     ele devolve o tensor em outro layout e o video sai preto/quebrado) ----
    ("Kijai/WanVideo_comfy",
     "Wan2_1_VAE_bf16.safetensors",
     "vae", "Wan2_1_VAE_bf16.safetensors",
     True, "VAE de video do Wan 2.1 (bf16)"),
]


def _fmt(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f}{u}"
        n /= 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True, help="pasta ComfyUI/models")
    ap.add_argument("--list", action="store_true", help="so lista, nao baixa")
    ap.add_argument("--force", action="store_true", help="rebaixa mesmo se existir")
    args = ap.parse_args()

    models_dir = os.path.abspath(args.models_dir)
    print(f"\n== Modelos Bruxos do VFX ==\ndestino: {models_dir}\n")

    if args.list:
        for _, _, folder, name, req, desc in MODELS:
            tag = "" if req else " (opcional)"
            print(f"  models/{folder}/{name}{tag}\n      {desc}")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERRO] huggingface_hub nao instalado. Rode o install.bat/install.sh "
              "(ele instala as dependencias antes de baixar).")
        return 1

    ok = skipped = failed = 0
    for repo, remote, folder, name, req, desc in MODELS:
        dest_dir = os.path.join(models_dir, folder)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name)

        if os.path.isfile(dest) and os.path.getsize(dest) > 0 and not args.force:
            print(f"[ja existe] {folder}/{name}  ({_fmt(os.path.getsize(dest))})")
            skipped += 1
            continue

        print(f"[baixando ] {desc}\n            {repo} -> models/{folder}/{name}")
        try:
            cached = hf_hub_download(repo_id=repo, filename=remote)
            # copia PLANO pro destino (o cache do HF guarda a subpasta do repo)
            shutil.copyfile(cached, dest)
            print(f"[ok       ] {folder}/{name}  ({_fmt(os.path.getsize(dest))})\n")
            ok += 1
        except Exception as e:
            print(f"[FALHOU   ] {folder}/{name}: {e}\n")
            failed += 1

    print(f"\n== Resumo: {ok} baixado(s), {skipped} ja existia(m), {failed} falhou(ram) ==")
    if failed:
        print("Rode o instalador de novo — ele retoma de onde parou.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
