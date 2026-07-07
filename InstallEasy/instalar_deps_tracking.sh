#!/usr/bin/env bash
# ============================================================================
#  Bruxos do VFX - Instalar dependencias dos nodes de Tracking (Linux/Mac)
#  Baseado no install_smart do pacote de tracking, mas sem tocar em torch/numpy.
#  Rode de dentro da pasta do pacote de tracking.
# ============================================================================
set -e

echo "======================================================================"
echo "  Bruxos - Dependencias dos nodes de Tracking"
echo "======================================================================"

NODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY_DIR="$(cd "$NODE_DIR/../.." && pwd)"

# achar python do ComfyUI
PY=""
for cand in "$COMFY_DIR/../python_embeded/python" "$COMFY_DIR/venv/bin/python" "$COMFY_DIR/.venv/bin/python" "python3"; do
  if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then PY="$cand"; break; fi
done
[ -z "$PY" ] && PY="python3"

echo "Python : $PY"
echo "ComfyUI: $COMFY_DIR"
echo "NAO toca em torch/torchvision/numpy."
echo

"$PY" -m pip install --upgrade pip
# deps leves (headless no Linux pra nao puxar libGL desnecessario)
"$PY" -m pip install --upgrade opencv-python-headless scipy roma kornia trimesh einops huggingface_hub pyyaml tqdm

echo
echo "Dependencias instaladas."
[ -f "$NODE_DIR/test_import.py" ] && "$PY" "$NODE_DIR/test_import.py" || true
echo
echo "PROXIMO PASSO (modelos pesados ~8GB):"
echo "  $PY $NODE_DIR/instalar_modelos_tracking.py"
echo
echo "Reinicie o ComfyUI completamente depois."
