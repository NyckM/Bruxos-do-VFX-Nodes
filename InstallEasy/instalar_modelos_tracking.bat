@echo off
REM ============================================================================
REM  Bruxos do VFX - Instalar Modelos de Tracking (Windows / python_embeded)
REM  Coloque este .bat e o instalar_modelos_tracking.py na pasta do pacote de
REM  tracking (ComfyUI\custom_nodes\<pacote-de-tracking>\) e rode.
REM ============================================================================
setlocal enabledelayedexpansion
chcp 65001 >nul

echo.
echo === Bruxos - Modelos de Tracking ===
echo.

set "NODE_DIR=%~dp0"
pushd "%NODE_DIR%\..\.."
set "COMFY_DIR=%CD%"
popd

set "PY="
if exist "%COMFY_DIR%\..\python_embeded\python.exe" set "PY=%COMFY_DIR%\..\python_embeded\python.exe"
if not defined PY if exist "%COMFY_DIR%\python_embeded\python.exe" set "PY=%COMFY_DIR%\python_embeded\python.exe"
if not defined PY set "PY=python"

echo Python: "%PY%"
echo.
echo Grupos disponiveis:
"%PY%" "%NODE_DIR%instalar_modelos_tracking.py" --list
echo.
echo Vou baixar TODOS os grupos. Pra escolher, rode:
echo    "%PY%" instalar_modelos_tracking.py --only dust3r,cotracker3
echo.
pause

"%PY%" "%NODE_DIR%instalar_modelos_tracking.py"

echo.
echo === Fim. Leia o RESUMO acima (downloads que falharam + compilacoes manuais). ===
pause
endlocal
