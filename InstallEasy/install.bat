@echo off
REM ============================================================================
REM  ComfyUI Bruxos do VFX - instalador (Windows / Python embedded)
REM
REM  Rode de DENTRO da pasta do node:
REM      ComfyUI\custom_nodes\ComfyUI-Bruxos-do-VFX\install.bat
REM
REM  O que ele faz:
REM    1) instala as dependencias com o python_embeded CORRETO
REM    2) instala o onnxruntime-gpu que CASA com a sua versao de CUDA (auto)
REM    3) baixa os modelos do Bernini-R (INT8 ConvRot) + LoRAs LightX2V
REM
REM  REGRAS DE OURO (nao mude):
REM    - NUNCA usa 'pip' solto (resolve pro Python errado).
REM    - NUNCA instala/atualiza torch, numpy, triton, xformers ou flash-attn.
REM      Isso quebraria seu ambiente. Este instalador NAO toca neles.
REM ============================================================================
setlocal enabledelayedexpansion
chcp 65001 >nul
echo.
echo === Bruxos do VFX - instalador ===
echo.

REM ---- localizar a raiz do ComfyUI subindo a partir daqui --------------------
set "NODE_DIR=%~dp0"
pushd "%NODE_DIR%\..\.."
set "COMFY_DIR=%CD%"
popd

REM ---- achar o python embedded ----------------------------------------------
set "PY="
if exist "%COMFY_DIR%\..\python_embeded\python.exe" set "PY=%COMFY_DIR%\..\python_embeded\python.exe"
if not defined PY if exist "%COMFY_DIR%\python_embeded\python.exe" set "PY=%COMFY_DIR%\python_embeded\python.exe"
if not defined PY (
  echo [ERRO] Nao achei o python_embeded. Edite este .bat e aponte a variavel PY
  echo        para o python.exe embedded do seu ComfyUI.
  pause & exit /b 1
)
echo Python embedded: "%PY%"
echo ComfyUI:         "%COMFY_DIR%"
echo.

REM ---- [1/4] dependencias base ----------------------------------------------
echo [1/4] Dependencias base (opencv, onnx, requests, tqdm, huggingface_hub, psutil)...
"%PY%" -m pip install --upgrade opencv-python onnx requests tqdm huggingface_hub psutil
if errorlevel 1 (
  echo [ERRO] Falha nas dependencias base. Veja o log acima.
  pause & exit /b 1
)
echo.

REM ---- [2/4] onnxruntime-gpu que CASA com a CUDA do seu torch ----------------
REM  IMPORTANTE: nao existe uma versao "certa" fixa. A build do onnxruntime tem
REM  que bater com a CUDA do seu torch, senao o FaceFusion cai na CPU:
REM    torch cu12x -> onnxruntime-gpu build CUDA 12
REM    torch cu13x -> onnxruntime-gpu build CUDA 13 (o do PyPI padrao)
REM  Detectamos automaticamente em vez de cravar uma versao.
echo [2/4] Detectando a CUDA do seu torch...
set "CUDA_MAJOR="
for /f "delims=" %%i in ('"%PY%" -c "import torch,sys; v=torch.version.cuda or ''; sys.stdout.write(v.split('.')[0] if v else 'none')" 2^>nul') do set "CUDA_MAJOR=%%i"

if "%CUDA_MAJOR%"=="12" (
  echo       torch com CUDA 12 detectado -^> onnxruntime-gpu build CUDA 12
  "%PY%" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-openvino onnxruntime-directml >nul 2>&1
  "%PY%" -m pip install "onnxruntime-gpu<1.23" --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
) else if "%CUDA_MAJOR%"=="13" (
  echo       torch com CUDA 13 detectado -^> onnxruntime-gpu padrao do PyPI
  "%PY%" -m pip install --upgrade onnxruntime-gpu
) else (
  echo       [AVISO] Nao consegui detectar a CUDA do torch ^(valor: "%CUDA_MAJOR%"^).
  echo               Instalando o onnxruntime-gpu padrao. Se o face swap cair na CPU,
  echo               instale a build que casa com a sua CUDA manualmente.
  "%PY%" -m pip install --upgrade onnxruntime-gpu
)
if errorlevel 1 echo [AVISO] onnxruntime-gpu falhou. O face swap pode cair na CPU.
echo.

REM ---- [3/4] deps leves de tracking + Qwen (Prompt Enhancer / Caption) -------
echo [3/4] Deps de tracking (roma, kornia, trimesh, einops, pyyaml)...
"%PY%" -m pip install --upgrade roma kornia trimesh einops pyyaml
if errorlevel 1 echo [AVISO] Alguma dep de tracking falhou; esses utilitarios podem nao carregar.

echo       Deps do Qwen-VL (Prompt Enhancer / Caption)...
"%PY%" -m pip install --upgrade transformers accelerate pillow
if errorlevel 1 echo [AVISO] transformers/accelerate falharam; os nodes de Qwen podem nao carregar.
echo.

REM ---- [4/4] modelos ---------------------------------------------------------
echo [4/4] Baixando modelos (varios GB - pode demorar)...
echo       Ja existentes sao PULADOS; pode rodar este .bat de novo pra retomar.
echo.
"%PY%" "%NODE_DIR%download_models.py" --models-dir "%COMFY_DIR%\models"
if errorlevel 1 (
  echo.
  echo [AVISO] Algum modelo falhou. Rode o .bat de novo - ele retoma de onde parou.
)

echo.
echo === Concluido. Reinicie o ComfyUI. ===
echo.
echo  Modelos instalados:
echo    models\diffusion_models\  wan2.2_bernini_r_high/low_noise_int8_convrot.safetensors
echo    models\loras\             Bernini-R_LightX2V_high/low_noise.safetensors  (4 steps)
echo    models\text_encoders\     umt5_xxl_fp8_e4m3fn_scaled.safetensors
echo    models\vae\               Wan2_1_VAE_bf16.safetensors
echo.
echo  ATENCAO no Loader Tudo-em-1:
echo    - use o VAE de VIDEO (Wan2_1_VAE_bf16). Um VAE 'imageonly'/'upscale2x'
echo      faz o video sair PRETO/quebrado.
echo    - com as LoRAs LightX2V: cfg = 1.0 e steps = 6 (split_step 4).
echo.
echo  Os .onnx do face swap baixam sozinhos no 1o uso, em models\facefusion\.
echo.
pause
endlocal
