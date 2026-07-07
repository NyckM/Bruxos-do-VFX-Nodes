@echo off
REM ============================================================================
REM  Bruxos do VFX - Instalar dependencias dos nodes de Tracking
REM  (baseado no INSTALL_NOW.bat / install_smart.bat do pacote de tracking,
REM   mas com o caminho DETECTADO automaticamente e SEM tocar em torch/numpy)
REM
REM  Coloque este .bat na pasta do pacote de tracking:
REM    ComfyUI\custom_nodes\<pacote-de-tracking>\  e rode com 2 cliques.
REM
REM  Este e o PASSO 1 (dependencias pip leves). O PASSO 2 (baixar os MODELOS
REM  pesados: DUSt3R, DROID-SLAM, CoTracker...) e o instalar_modelos_tracking.py
REM ============================================================================
setlocal enabledelayedexpansion
chcp 65001 >nul

echo.
echo ======================================================================
echo   Bruxos - Dependencias dos nodes de Tracking
echo ======================================================================
echo.

REM ---- localizar a raiz do ComfyUI subindo a partir daqui ----
REM  este .bat esta em  <ComfyUI>\custom_nodes\<pacote>\
set "NODE_DIR=%~dp0"
pushd "%NODE_DIR%\..\.."
set "COMFY_DIR=%CD%"
popd

REM ---- achar o python embedded (irmao da pasta ComfyUI, ou dentro dela) ----
set "PY="
if exist "%COMFY_DIR%\..\python_embeded\python.exe" set "PY=%COMFY_DIR%\..\python_embeded\python.exe"
if not defined PY if exist "%COMFY_DIR%\python_embeded\python.exe" set "PY=%COMFY_DIR%\python_embeded\python.exe"
if not defined PY (
  echo [AVISO] Nao achei o python_embeded automaticamente. Usando 'python' do sistema.
  echo         Se der erro, edite este .bat e aponte a variavel PY para o python.exe
  echo         embedded do seu ComfyUI.
  set "PY=python"
)

echo Python : "%PY%"
echo ComfyUI: "%COMFY_DIR%"
echo.
echo Este script NAO toca em torch, torchvision nem numpy (pra nao quebrar seu
echo ambiente CUDA). Instala so as dependencias leves que faltam.
echo.
pause

echo.
echo [1/2] Atualizando pip...
"%PY%" -m pip install --upgrade pip

echo.
echo [2/2] Instalando dependencias leves de tracking...
echo        (opencv, scipy, roma, kornia, trimesh, einops, huggingface_hub, pyyaml, tqdm)
"%PY%" -m pip install --upgrade opencv-python scipy roma kornia trimesh einops huggingface_hub pyyaml tqdm
if errorlevel 1 (
  echo.
  echo [ERRO] Alguma dependencia falhou. Veja o log acima e rode de novo.
  pause & exit /b 1
)

echo.
echo ======================================================================
echo   Dependencias instaladas!
echo ======================================================================
echo.

REM ---- teste de import, se o pacote tiver o test_import.py ----
if exist "%NODE_DIR%test_import.py" (
  echo Testando imports...
  "%PY%" "%NODE_DIR%test_import.py"
  echo.
)

echo PROXIMO PASSO (modelos pesados):
echo   "%PY%" "%NODE_DIR%instalar_modelos_tracking.py"
echo   ^(baixa DUSt3R/DROID-SLAM/CoTracker... ~8 GB; alguns exigem compilacao^)
echo.
echo IMPORTANTE: reinicie o ComfyUI COMPLETAMENTE depois:
echo   1) feche a aba do navegador
echo   2) feche esta janela/console
echo   3) inicie o ComfyUI do zero
echo.
pause
endlocal
