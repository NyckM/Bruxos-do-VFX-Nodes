#!/usr/bin/env bash
# =============================================================================
#  Bruxos do VFX - setup RunPod (Linux)
#  Baixa os custom nodes Bruxos + modelos, ja nas pastas certas, em paralelo,
#  validando cada arquivo (rejeita download incompleto / pagina HTML).
#
#  COMO USAR (no terminal do pod, pra rodar em background e nao travar o ComfyUI):
#     cd /workspace
#     nohup bash bruxos_runpod_setup.sh > bruxos_setup.log 2>&1 &
#     tail -f bruxos_setup.log        # acompanhar o progresso
#
#  Da pra colar isso tambem no "start command" do template do pod.
# =============================================================================
set -u

# ---- onde fica o ComfyUI (ajuste se o seu template usar outro caminho) ------
COMFY="${COMFY:-/workspace/ComfyUI}"
if [ ! -d "$COMFY" ]; then
  for c in /ComfyUI /root/ComfyUI /workspace/ComfyUI; do
    [ -d "$c" ] && COMFY="$c" && break
  done
fi
echo "[Bruxos] ComfyUI em: $COMFY"

# python do ComfyUI (pra instalar deps do custom node)
PY="${PY:-python3}"

# token opcional do HuggingFace (pra repos com rate-limit). export HF_TOKEN=... antes de rodar
HF_TOKEN="${HF_TOKEN:-}"

# repo git dos seus custom nodes. Troque se mudar o repo.
BRUXOS_GIT="${BRUXOS_GIT:-https://github.com/NyckM/Bruxos-do-VFX-Nodes}"

# ---- pastas de modelos ------------------------------------------------------
mkdir -p "$COMFY/models/unet" \
         "$COMFY/models/loras" \
         "$COMFY/models/text_encoders" \
         "$COMFY/models/clip" \
         "$COMFY/models/vae" \
         "$COMFY/custom_nodes"

# ---- downloader robusto: valida tamanho e rejeita HTML/parciais -------------
# uso: fetch <url> <pasta_destino> <nome_arquivo> <min_bytes>
fetch() {
  local url="$1" dir="$2" name="$3" min="${4:-1000000}"
  local out="$dir/$name" tmp="$dir/.$name.part"

  if [ -f "$out" ]; then
    local cur; cur=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$cur" -ge "$min" ]; then
      echo "[skip] $name ja existe ($cur bytes)"; return 0
    fi
    echo "[Bruxos] $name existe mas pequeno demais ($cur bytes), rebaixando"; rm -f "$out"
  fi

  echo "[get ] $name"
  if command -v aria2c >/dev/null 2>&1; then
    local ah=()
    [ -n "$HF_TOKEN" ] && ah=(--header="Authorization: Bearer $HF_TOKEN")
    aria2c -x16 -s16 -k1M --console-log-level=warn --summary-interval=0 \
      "${ah[@]}" -d "$dir" -o ".$name.part" "$url" || { echo "[ERRO] download $name"; return 1; }
  else
    local wh=()
    [ -n "$HF_TOKEN" ] && wh=(--header="Authorization: Bearer $HF_TOKEN")
    wget -c -q --show-progress "${wh[@]}" -O "$tmp" "$url" \
      || { echo "[ERRO] download $name"; return 1; }
  fi

  # ---- validacao ----
  local first sz
  first=$(head -c1 "$tmp" 2>/dev/null || echo "")
  sz=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
  if [ "$first" = "<" ]; then
    echo "[ERRO] $name veio como HTML (link errado ou login). Apagando."; rm -f "$tmp"; return 1
  fi
  if [ "$sz" -lt "$min" ]; then
    echo "[ERRO] $name pequeno demais ($sz < $min) -> incompleto. Apagando."; rm -f "$tmp"; return 1
  fi
  mv -f "$tmp" "$out"
  echo "[ ok ] $name ($sz bytes)"
}

HF="https://huggingface.co"

# =============================================================================
#  MODELOS  (edite nomes/quant se quiser outra versao)
# =============================================================================

# 1) UNET GGUF (high + low)  -> models/unet
fetch "$HF/neuregex/Bernini-R-GGUF/resolve/main/bernini_r_high_noise_14B-Q4_K_M.gguf" \
      "$COMFY/models/unet" "bernini_r_high_noise_14B-Q4_K_M.gguf" 1000000000 &
fetch "$HF/neuregex/Bernini-R-GGUF/resolve/main/bernini_r_low_noise_14B-Q4_K_M.gguf" \
      "$COMFY/models/unet" "bernini_r_low_noise_14B-Q4_K_M.gguf" 1000000000 &

# 2) LoRA  -> models/loras
fetch "$HF/Cyph3r/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16/resolve/main/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors" \
      "$COMFY/models/loras" "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors" 10000000 &

# 3) UMT5 text encoder  -> models/text_encoders
fetch "$HF/Osrivers/umt5_xxl_fp8_e4m3fn_scaled.safetensors/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
      "$COMFY/models/text_encoders" "umt5_xxl_fp8_e4m3fn_scaled.safetensors" 50000000 &

# 4) VAE  -> models/vae   (nome do arquivo = Wan2_1_VAE_bf16.safetensors)
fetch "$HF/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors" \
      "$COMFY/models/vae" "Wan2_1_VAE_bf16.safetensors" 10000000 &

# =============================================================================
#  CUSTOM NODES BRUXOS  (em paralelo com os downloads)
# =============================================================================
(
  DST="$COMFY/custom_nodes/ComfyUI-Bruxos-do-VFX"
  if [ -n "$BRUXOS_GIT" ]; then
    if [ -d "$DST/.git" ]; then
      echo "[Bruxos] atualizando node (git pull)"; git -C "$DST" pull --ff-only
    else
      echo "[Bruxos] clonando node de $BRUXOS_GIT"; git clone --depth 1 "$BRUXOS_GIT" "$DST"
    fi
    # deps leves do pacote (a maioria ja vem na imagem do ComfyUI)
    "$PY" -m pip install --no-input imageio-ffmpeg >/dev/null 2>&1 || true
    [ -f "$DST/requirements.txt" ] && "$PY" -m pip install --no-input -r "$DST/requirements.txt" || true
    echo "[Bruxos] node instalado em $DST"
  else
    echo "[Bruxos] BRUXOS_GIT vazio -> pulei a instalacao do node."
    echo "         Defina o repo antes de rodar:  export BRUXOS_GIT=https://github.com/voce/ComfyUI-Bruxos-do-VFX"
    echo "         (ou suba a pasta manualmente em $COMFY/custom_nodes/)"
  fi
) &

# espera tudo terminar
wait
echo ""
echo "=================== RESUMO ==================="
for f in \
  "models/unet/bernini_r_high_noise_14B-Q4_K_M.gguf" \
  "models/unet/bernini_r_low_noise_14B-Q4_K_M.gguf" \
  "models/loras/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors" \
  "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  "models/vae/Wan2_1_VAE_bf16.safetensors" ; do
  if [ -f "$COMFY/$f" ]; then
    printf "  OK   %-70s %s bytes\n" "$f" "$(stat -c%s "$COMFY/$f")"
  else
    printf "  FALTA %-70s (rever link/log acima)\n" "$f"
  fi
done
echo "============================================="
echo "[Bruxos] terminado. Reinicie o ComfyUI se ele ja estava aberto."
