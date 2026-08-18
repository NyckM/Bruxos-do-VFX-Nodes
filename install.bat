@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title ComfyUI Bruxos do VFX - Instalador

set "NODE_DIR=%~dp0"
set "COMFY_DIR="
set "PY="
set "SKIP_MODELS=0"
if /I "%~1"=="--deps-only" set "SKIP_MODELS=1"
if /I "%~1"=="--skip-models" set "SKIP_MODELS=1"

echo.
echo ============================================================
echo   ComfyUI Bruxos do VFX - instalação / atualização
echo ============================================================
echo.

for %%D in ("%NODE_DIR%..\..") do set "COMFY_DIR=%%~fD"
if not exist "%COMFY_DIR%\custom_nodes" (
  echo [ERRO] Esta pasta não está em ComfyUI\custom_nodes.
  echo        ComfyUI detectado: "%COMFY_DIR%"
  goto :fail
)

if exist "%COMFY_DIR%\..\python_embeded\python.exe" set "PY=%COMFY_DIR%\..\python_embeded\python.exe"
if not defined PY if exist "%COMFY_DIR%\python_embeded\python.exe" set "PY=%COMFY_DIR%\python_embeded\python.exe"
if not defined PY if exist "%COMFY_DIR%\.venv\Scripts\python.exe" set "PY=%COMFY_DIR%\.venv\Scripts\python.exe"
if not defined PY if exist "%COMFY_DIR%\venv\Scripts\python.exe" set "PY=%COMFY_DIR%\venv\Scripts\python.exe"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY (
  echo [ERRO] Python do ComfyUI não encontrado.
  goto :fail
)

echo Node:    "%NODE_DIR%"
echo ComfyUI: "%COMFY_DIR%"
echo Python:  "%PY%"
echo.

echo [1/4] Instalando requirements.txt...
"%PY%" -m pip install --upgrade -r "%NODE_DIR%requirements.txt"
if errorlevel 1 goto :fail

echo.
echo [2/4] Dependências de Qwen-VL...
"%PY%" -m pip install --upgrade "transformers>=4.45" accelerate pillow
if errorlevel 1 echo [AVISO] Qwen-VL não foi instalado; caption/enhancer podem não carregar.

echo.
echo [3/4] ONNX Runtime compatível com a CUDA do torch...
set "CUDA_MAJOR=none"
for /f "delims=" %%C in ('"%PY%" -c "import torch; print((torch.version.cuda or 'none').split('.')[0])" 2^>nul') do set "CUDA_MAJOR=%%C"
if "!CUDA_MAJOR!"=="12" (
  "%PY%" -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-openvino onnxruntime-directml >nul 2>&1
  "%PY%" -m pip install "onnxruntime-gpu<1.23" --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
) else if "!CUDA_MAJOR!"=="13" (
  "%PY%" -m pip install --upgrade onnxruntime-gpu
) else (
  echo [AVISO] CUDA do torch não detectada ^(!CUDA_MAJOR!^); instalando build padrão.
  "%PY%" -m pip install --upgrade onnxruntime-gpu
)
if errorlevel 1 echo [AVISO] ONNX Runtime GPU falhou; FaceFusion pode usar CPU.

echo.
if "%SKIP_MODELS%"=="1" (
  echo [4/4] Download de modelos ignorado.
) else (
  echo [4/4] Baixando/validando modelos Bernini e Wan...
  "%PY%" "%NODE_DIR%download_models.py" --models-dir "%COMFY_DIR%\models"
  if errorlevel 1 echo [AVISO] Algum download falhou. Rode novamente para retomar.
)

echo.
echo [OK] Instalação concluída. Reinicie o ComfyUI e pressione F5.
echo.
pause
exit /b 0

:fail
echo.
echo [ERRO] Instalação interrompida. Consulte as mensagens acima.
echo Torch, numpy, triton, xformers e flash-attn não foram alterados diretamente.
echo.
pause
exit /b 1
