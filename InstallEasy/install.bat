@echo off
REM ============================================================================
REM  ComfyUI Bruxos do VFX - instalador (Windows / Python embedded)
REM  Rode este .bat de DENTRO da pasta do node:
REM     ComfyUI\custom_nodes\ComfyUI-Bruxos-do-VFX\install.bat
REM  Ele: 1) instala as dependencias com o python_embeded correto
REM       2) baixa os modelos do Bernini/Wan pras pastas certas
REM  NUNCA usa o pip solto (resolve pro Python errado) e NAO instala xformers/flash-attn.
REM ============================================================================
setlocal enabledelayedexpansion
chcp 65001 >nul

echo.
echo === Bruxos do VFX - instalador ===
echo.

REM ---- localizar a raiz do ComfyUI subindo a partir daqui ----
REM  este .bat esta em  <ComfyUI>\custom_nodes\ComfyUI-Bruxos-do-VFX\
set "NODE_DIR=%~dp0"
pushd "%NODE_DIR%\..\.." 
set "COMFY_DIR=%CD%"
popd

REM ---- achar o python embedded (fica como IRMAO da pasta ComfyUI, em ComfyUI-Easy-Install) ----
set "PY="
if exist "%COMFY_DIR%\..\python_embeded\python.exe" set "PY=%COMFY_DIR%\..\python_embeded\python.exe"
if not defined PY if exist "%COMFY_DIR%\python_embeded\python.exe" set "PY=%COMFY_DIR%\python_embeded\python.exe"

if not defined PY (
  echo [ERRO] Nao achei o python_embeded. Edite este .bat e aponte a variavel PY
  echo        para o python.exe embedded do seu ComfyUI.
  pause & exit /b 1
)
echo Python embedded: "%PY%"
echo ComfyUI: "%COMFY_DIR%"
echo.

REM ---- 1) dependencias ----
echo [1/3] Instalando dependencias base (opencv, onnx, requests, tqdm, huggingface_hub)...
"%PY%" -m pip install --upgrade opencv-python onnx requests tqdm huggingface_hub
if errorlevel 1 (
  echo [ERRO] Falha ao instalar dependencias base. Veja o log acima.
  pause & exit /b 1
)

REM ---- onnxruntime-gpu: CRAVAR a build CUDA 12 (casa com torch cu12x do ComfyUI) ----
REM  NAO usar "--upgrade onnxruntime-gpu" solto: a versao nova e build CUDA 13 e
REM  quebra o FaceFusion (erro cublasLt64_13.dll -> cai na CPU). Forcamos CUDA 12.
echo [2/3] Instalando onnxruntime-gpu (build CUDA 12)...
"%PY%" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-openvino onnxruntime-directml >nul 2>&1
"%PY%" -m pip install "onnxruntime-gpu<1.23" --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
if errorlevel 1 (
  echo [AVISO] Falha ao instalar onnxruntime-gpu CUDA 12. O face swap pode cair na CPU.
)

REM ---- deps LEVES p/ os nodes de tracking Bruxos (trajetoria/export/utils) ----
REM  So pacotes pip inofensivos. Os modelos pesados (DROID-SLAM, DUSt3R, CoTracker...)
REM  NAO sao instalados aqui: eles vem do pacote de tracking separado (git clone proprio).
echo [3/3] Instalando deps leves de tracking (roma, kornia, trimesh, einops, scipy, pyyaml)...
"%PY%" -m pip install --upgrade roma kornia trimesh einops scipy pyyaml
if errorlevel 1 (
  echo [AVISO] Falha em alguma dep de tracking. Os utilitarios de tracking podem nao carregar.
)
REM  (opcional) Qwen-VL Caption precisa disto; descomente se for usar:
REM "%PY%" -m pip install --upgrade transformers accelerate pillow

echo.
echo [2/2] Baixando modelos (pode demorar - varios GB)...
set "HF=%PY% -m huggingface_hub.commands.huggingface_cli download"

REM  UNET GGUF high/low  ->  models\unet
"%PY%" -m huggingface_hub.commands.huggingface_cli download neuregex/Bernini-R-GGUF bernini_r_high_noise_14B-Q4_K_M.gguf --local-dir "%COMFY_DIR%\models\unet"
"%PY%" -m huggingface_hub.commands.huggingface_cli download neuregex/Bernini-R-GGUF bernini_r_low_noise_14B-Q4_K_M.gguf  --local-dir "%COMFY_DIR%\models\unet"

REM  LoRA distill  ->  models\loras
"%PY%" -m huggingface_hub.commands.huggingface_cli download Cyph3r/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16 --local-dir "%COMFY_DIR%\models\loras"

REM  Text encoder umt5  ->  models\text_encoders
"%PY%" -m huggingface_hub.commands.huggingface_cli download Osrivers/umt5_xxl_fp8_e4m3fn_scaled.safetensors umt5_xxl_fp8_e4m3fn_scaled.safetensors --local-dir "%COMFY_DIR%\models\text_encoders"

REM  VAE  ->  models\vae
"%PY%" -m huggingface_hub.commands.huggingface_cli download Kijai/WanVideo_comfy Wan2_1_VAE_bf16.safetensors --local-dir "%COMFY_DIR%\models\vae"

echo.
echo === Concluido. Reinicie o ComfyUI. ===
echo  - Se um download falhar, rode o .bat de novo (ele retoma).
echo  - Os modelos ONNX de face swap baixam sozinhos no 1o uso, em models\facefusion\.
echo.
pause
endlocal
