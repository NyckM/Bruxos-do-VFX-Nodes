#!/usr/bin/env bash
# ============================================================================
#  ComfyUI Bruxos do VFX - instalador (Linux / RunPod)
#  Rode de dentro da pasta do node:
#     cd ComfyUI/custom_nodes/ComfyUI-Bruxos-do-VFX && bash install.sh
#  Ele: 1) instala as dependencias no python do ambiente do ComfyUI
#       2) baixa os modelos do Bernini/Wan pras pastas certas
# ============================================================================
set -euo pipefail

echo
echo "=== Bruxos do VFX - instalador ==="
echo

# ---- localizar a raiz do ComfyUI (este script esta em <ComfyUI>/custom_nodes/ComfyUI-Bruxos-do-VFX/) ----
NODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY_DIR="$(cd "$NODE_DIR/../.." && pwd)"

# ---- escolher o python: prioriza venv do ComfyUI, senao PYTHON env, senao python3 ----
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "$COMFY_DIR/venv/bin/python" ]; then
  PY="$COMFY_DIR/venv/bin/python"
elif [ -x "$COMFY_DIR/../venv/bin/python" ]; then
  PY="$COMFY_DIR/../venv/bin/python"
else
  PY="python3"
fi
echo "Python: $PY"
echo "ComfyUI: $COMFY_DIR"
echo

# ---- 1) dependencias ----
echo "[1/2] Instalando dependencias (onnxruntime-gpu, opencv, onnx, requests, tqdm, huggingface_hub)..."
"$PY" -m pip install --upgrade onnxruntime-gpu opencv-python-headless onnx requests tqdm "huggingface_hub[cli]"
# deps LEVES p/ os nodes de tracking Bruxos (trajetoria/export/utils). Modelos pesados
# (DROID-SLAM, DUSt3R, CoTracker...) NAO entram aqui: vem do pacote de tracking separado.
"$PY" -m pip install --upgrade roma kornia trimesh einops scipy pyyaml
# (opcional) Qwen-VL Caption:
# "$PY" -m pip install --upgrade transformers accelerate pillow

echo
echo "[2/2] Baixando modelos (pode demorar - varios GB)..."
dl() { "$PY" -m huggingface_hub.commands.huggingface_cli download "$@"; }

# UNET GGUF high/low -> models/unet
dl neuregex/Bernini-R-GGUF bernini_r_high_noise_14B-Q4_K_M.gguf --local-dir "$COMFY_DIR/models/unet"
dl neuregex/Bernini-R-GGUF bernini_r_low_noise_14B-Q4_K_M.gguf  --local-dir "$COMFY_DIR/models/unet"

# LoRA distill -> models/loras
dl Cyph3r/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16 --local-dir "$COMFY_DIR/models/loras"

# Text encoder umt5 -> models/text_encoders
dl Osrivers/umt5_xxl_fp8_e4m3fn_scaled.safetensors umt5_xxl_fp8_e4m3fn_scaled.safetensors --local-dir "$COMFY_DIR/models/text_encoders"

# VAE -> models/vae
dl Kijai/WanVideo_comfy Wan2_1_VAE_bf16.safetensors --local-dir "$COMFY_DIR/models/vae"

echo
echo "=== Concluido. Reinicie o ComfyUI. ==="
echo " - Se um download falhar, rode o script de novo (ele retoma)."
echo " - Os modelos ONNX de face swap baixam sozinhos no 1o uso, em models/facefusion/."
