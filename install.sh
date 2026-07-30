#!/usr/bin/env bash
# ============================================================================
#  ComfyUI Bruxos do VFX - instalador (Linux / macOS / RunPod)
#
#  Rode de DENTRO da pasta do node:
#      cd ComfyUI/custom_nodes/ComfyUI-Bruxos-do-VFX && bash install.sh
#
#  REGRAS DE OURO (nao mude):
#    - NUNCA instala/atualiza torch, numpy, triton, xformers ou flash-attn.
#      Isso quebraria seu ambiente. Este instalador NAO toca neles.
# ============================================================================
set -u

NODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY_DIR="$(cd "$NODE_DIR/../.." && pwd)"

echo
echo "=== Bruxos do VFX - instalador ==="
echo

# ---- achar o python certo (venv > embedded > sistema) ----------------------
PY=""
for cand in "$COMFY_DIR/venv/bin/python" "$COMFY_DIR/../venv/bin/python" \
            "$COMFY_DIR/../python_embeded/python" "$(command -v python3 || true)"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "[ERRO] Nao achei um python. Ative seu venv e rode de novo."
  exit 1
fi
echo "Python:  $PY"
echo "ComfyUI: $COMFY_DIR"
echo

# ---- [1/5] dependencias base ------------------------------------------------
echo "[1/5] Dependencias base (opencv, onnx, requests, tqdm, huggingface_hub, psutil)..."
"$PY" -m pip install --upgrade opencv-python onnx requests tqdm huggingface_hub psutil || {
  echo "[ERRO] Falha nas dependencias base."; exit 1; }
echo

# ---- [2/5] onnxruntime-gpu que CASA com a CUDA do torch ---------------------
#  torch cu12x -> build CUDA 12 ; torch cu13x -> build CUDA 13 (PyPI padrao)
echo "[2/5] Detectando a CUDA do seu torch..."
CUDA_MAJOR="$("$PY" -c "import torch,sys; v=torch.version.cuda or ''; sys.stdout.write(v.split('.')[0] if v else 'none')" 2>/dev/null || echo none)"

case "$CUDA_MAJOR" in
  12)
    echo "      torch com CUDA 12 -> onnxruntime-gpu build CUDA 12"
    "$PY" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-openvino >/dev/null 2>&1 || true
    "$PY" -m pip install "onnxruntime-gpu<1.23" \
      --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
      || echo "[AVISO] onnxruntime-gpu CUDA 12 falhou; face swap pode cair na CPU."
    ;;
  13)
    echo "      torch com CUDA 13 -> onnxruntime-gpu padrao do PyPI"
    "$PY" -m pip install --upgrade onnxruntime-gpu \
      || echo "[AVISO] onnxruntime-gpu falhou; face swap pode cair na CPU."
    ;;
  *)
    echo "      [AVISO] CUDA do torch nao detectada (valor: '$CUDA_MAJOR')."
    echo "              Instalando onnxruntime-gpu padrao."
    "$PY" -m pip install --upgrade onnxruntime-gpu || true
    ;;
esac
echo

# ---- [3/5] tracking + Qwen ---------------------------------------------------
echo "[3/5] Deps de tracking (roma, kornia, trimesh, einops, pyyaml)..."
"$PY" -m pip install --upgrade roma kornia trimesh einops pyyaml \
  || echo "[AVISO] Alguma dep de tracking falhou."

echo "      Deps do Qwen-VL (Prompt Enhancer / Caption)..."
"$PY" -m pip install --upgrade transformers accelerate pillow \
  || echo "[AVISO] transformers/accelerate falharam."
echo

# ---- [4/5] deps do DepthMask DVD (BruxosDepthMask) ---------------------------
echo "[4/5] Deps do DepthMask DVD (omegaconf, peft, accelerate, safetensors, einops,"
echo "      sentencepiece, matplotlib, modelscope, pandas, imageio[ffmpeg])..."
"$PY" -m pip install --upgrade omegaconf peft accelerate safetensors einops sentencepiece matplotlib modelscope pandas "imageio[ffmpeg]" \
  || echo "[AVISO] Alguma dep do DepthMask DVD falhou; o node pode nao carregar."

echo "      Verificando FlashAttention (flash_attn)..."
if "$PY" -c "import flash_attn" >/dev/null 2>&1; then
  echo "      flash_attn OK."
else
  echo "      [AVISO] flash_attn NAO esta instalado. O DepthMask DVD roda bem mais"
  echo "              lento (ou pode falhar) sem ele. A wheel e especifica de"
  echo "              CUDA/torch/python, entao este instalador nao a instala"
  echo "              sozinho. Use o instalador dedicado da sua distro (ex.:"
  echo "              Add-ons/FlashAttention.bat no ComfyUI-Easy-Install, ou"
  echo "              'pip install flash-attn --no-build-isolation' no Linux)."
fi
echo

# ---- [5/5] modelos -----------------------------------------------------------
echo "[5/5] Baixando modelos (varios GB - pode demorar)..."
echo "      Ja existentes sao PULADOS; pode rodar de novo pra retomar."
echo
"$PY" "$NODE_DIR/download_models.py" --models-dir "$COMFY_DIR/models" \
  || echo "[AVISO] Algum modelo falhou. Rode de novo - ele retoma."

cat <<'EOF'

=== Concluido. Reinicie o ComfyUI. ===

 Modelos instalados:
   models/diffusion_models/  wan2.2_bernini_r_high|low_noise_int8_convrot.safetensors
   models/loras/             Bernini-R_LightX2V_high|low_noise.safetensors  (4 steps)
   models/text_encoders/     umt5_xxl_fp8_e4m3fn_scaled.safetensors
   models/vae/               Wan2_1_VAE_bf16.safetensors

 ATENCAO no Loader Tudo-em-1:
   - use o VAE de VIDEO (Wan2_1_VAE_bf16). Um VAE 'imageonly'/'upscale2x'
     faz o video sair PRETO/quebrado.
   - com as LoRAs LightX2V: cfg = 1.0 e steps = 6 (split_step 4).

 Os .onnx do face swap baixam sozinhos no 1o uso, em models/facefusion/.

 ----------------------------------------------------------------------------
 REQUISITOS DE TERCEIROS (so p/ alguns nodes; o resto do pacote roda sem eles):
   - Nodes MoCha (Embeds/Info/BBox) e WanVideo Context:
       precisam do ComfyUI-WanVideoWrapper (Kijai) instalado.
   - LTX Tiled Sampler:
       precisa do ComfyUI-LTXVideo (Lightricks) instalado.
   Estes instaladores NAO instalam esses pacotes (nao mexemos em nodes de
   terceiros); instale-os pelo ComfyUI-Manager se for usar esses nodes.
 ----------------------------------------------------------------------------
EOF
