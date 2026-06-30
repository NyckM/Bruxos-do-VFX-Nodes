@echo off
REM ============================================================================
REM  Bruxos do VFX - setup Windows (.bat)  [coloque DENTRO da pasta do ComfyUI]
REM  - Detecta o ComfyUI a partir da PASTA deste .bat (funciona em qualquer PC).
REM  - Baixa os custom nodes do GitHub (git ou zip) para custom_nodes.
REM  - Baixa os modelos (GGUF high/low, LoRA, umt5, VAE) nas pastas certas,
REM    com retomada (curl -C -) e validacao de tamanho (rejeita HTML/parcial).
REM  Reexecutar pula o que ja esta completo.
REM ============================================================================
setlocal enabledelayedexpansion

REM ---- acha a raiz do ComfyUI a partir da pasta deste .bat -------------------
set "COMFY=%~dp0"
if "%COMFY:~-1%"=="\" set "COMFY=%COMFY:~0,-1%"
REM se o .bat foi posto um nivel acima (ao lado de python_embeded), desce p/ ComfyUI
if not exist "%COMFY%\models" if exist "%COMFY%\ComfyUI\models" set "COMFY=%COMFY%\ComfyUI"
if not exist "%COMFY%\custom_nodes" if exist "%COMFY%\ComfyUI\custom_nodes" set "COMFY=%COMFY%\ComfyUI"
echo [Bruxos] ComfyUI em: %COMFY%
if not exist "%COMFY%\custom_nodes" (
  echo [ERRO] Nao parece a pasta do ComfyUI ^(sem 'custom_nodes'^).
  echo        Coloque este .bat dentro da pasta do ComfyUI e rode de novo.
  pause & exit /b 1
)

set "HF=https://huggingface.co"
set "REPO=https://github.com/NyckM/Bruxos-do-VFX-Nodes"
set "ZIPURL=https://codeload.github.com/NyckM/Bruxos-do-VFX-Nodes/zip/refs/heads/main"
set "DST=%COMFY%\custom_nodes\ComfyUI-Bruxos-do-VFX"

where curl.exe >nul 2>&1
if errorlevel 1 ( echo [ERRO] curl.exe nao encontrado ^(Windows 10 1803+ ou 11^). & pause & exit /b 1 )

REM ---- python embutido (pra deps do node), se existir -----------------------
set "PY=python"
if exist "%COMFY%\..\python_embeded\python.exe" set "PY=%COMFY%\..\python_embeded\python.exe"

REM ============================================================================
REM  1) CUSTOM NODES BRUXOS (do GitHub)
REM ============================================================================
echo.
echo [Bruxos] instalando custom nodes...
where git >nul 2>&1
if not errorlevel 1 (
  if exist "%DST%\.git" (
    echo [Bruxos] git pull em %DST%
    git -C "%DST%" pull --ff-only
  ) else (
    if exist "%DST%" (
      echo [Bruxos] pasta existe sem git; atualizando por cima via zip
      call :node_zip
    ) else (
      echo [Bruxos] git clone %REPO%
      git clone --depth 1 "%REPO%" "%DST%"
    )
  )
) else (
  echo [Bruxos] git nao encontrado; usando zip
  call :node_zip
)

REM deps leves (a maioria ja vem no ComfyUI)
"%PY%" -m pip install --no-input imageio-ffmpeg >nul 2>&1
if exist "%DST%\requirements.txt" "%PY%" -m pip install --no-input -r "%DST%\requirements.txt" >nul 2>&1

REM ============================================================================
REM  2) PASTAS DE MODELOS
REM ============================================================================
for %%D in (unet loras text_encoders clip vae) do if not exist "%COMFY%\models\%%D" md "%COMFY%\models\%%D"

REM ============================================================================
REM  3) DOWNLOADS  (nome | subpasta | min_bytes | url)
REM ============================================================================
call :fetch "bernini_r_high_noise_14B-Q4_K_M.gguf" "unet" 1000000000 "%HF%/neuregex/Bernini-R-GGUF/resolve/main/bernini_r_high_noise_14B-Q4_K_M.gguf"
call :fetch "bernini_r_low_noise_14B-Q4_K_M.gguf"  "unet" 1000000000 "%HF%/neuregex/Bernini-R-GGUF/resolve/main/bernini_r_low_noise_14B-Q4_K_M.gguf"
call :fetch "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors" "loras" 10000000 "%HF%/Cyph3r/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16/resolve/main/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"
call :fetch "umt5_xxl_fp8_e4m3fn_scaled.safetensors" "text_encoders" 50000000 "%HF%/Osrivers/umt5_xxl_fp8_e4m3fn_scaled.safetensors/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
call :fetch "Wan2_1_VAE_bf16.safetensors" "vae" 10000000 "%HF%/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"

echo.
echo =================== RESUMO ===================
if exist "%DST%\__init__.py" (echo   OK    custom_nodes\ComfyUI-Bruxos-do-VFX) else (echo   FALTA custom_nodes\ComfyUI-Bruxos-do-VFX)
call :report "models\unet\bernini_r_high_noise_14B-Q4_K_M.gguf"
call :report "models\unet\bernini_r_low_noise_14B-Q4_K_M.gguf"
call :report "models\loras\lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"
call :report "models\text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors"
call :report "models\vae\Wan2_1_VAE_bf16.safetensors"
echo =============================================
echo [Bruxos] terminado. Reinicie o ComfyUI e de Ctrl+Shift+R no navegador.
pause
exit /b 0

REM ============================================================================
:node_zip
set "TMPZIP=%TEMP%\bruxos_nodes.zip"
echo [Bruxos] baixando zip do GitHub...
curl.exe -L --fail --retry 3 -o "%TMPZIP%" "%ZIPURL%"
if errorlevel 1 ( echo [ERRO] falha ao baixar o zip dos nodes. & goto :eof )
echo [Bruxos] extraindo...
if exist "%TEMP%\bruxos_extract" rd /s /q "%TEMP%\bruxos_extract"
md "%TEMP%\bruxos_extract"
tar -xf "%TMPZIP%" -C "%TEMP%\bruxos_extract"
if not exist "%DST%" md "%DST%"
for /d %%F in ("%TEMP%\bruxos_extract\*") do robocopy "%%F" "%DST%" /E /NFL /NDL /NJH /NJS /NC /NS >nul
rd /s /q "%TEMP%\bruxos_extract" 2>nul
del /q "%TMPZIP%" 2>nul
goto :eof

REM ============================================================================
:fetch
set "NAME=%~1"
set "SUB=%~2"
set "MIN=%~3"
set "URL=%~4"
set "OUT=%COMFY%\models\%SUB%\%NAME%"
if exist "%OUT%" (
  powershell -nop -c "if((Get-Item -LiteralPath '%OUT%').Length -ge %MIN%){exit 0}else{exit 1}" >nul 2>&1
  if not errorlevel 1 ( echo [skip] %NAME% ja completo & goto :eof )
  powershell -nop -c "if((Get-Item -LiteralPath '%OUT%').Length -lt 1000000){exit 0}else{exit 1}" >nul 2>&1
  if not errorlevel 1 ( echo [Bruxos] %NAME% fantasma, apagando & del /q "%OUT%" 2>nul ) else ( echo [Bruxos] %NAME% parcial, retomando... )
)
echo [get ] %NAME%
curl.exe -L --fail --retry 3 --retry-delay 2 -C - -o "%OUT%" "%URL%"
if errorlevel 1 ( echo [ERRO] download de %NAME% falhou ^(link/HTTP^). & goto :eof )
powershell -nop -c "if((Get-Item -LiteralPath '%OUT%').Length -ge %MIN%){exit 0}else{exit 1}" >nul 2>&1
if errorlevel 1 ( echo [ERRO] %NAME% incompleto, apagando. & del /q "%OUT%" 2>nul ) else ( echo [ ok ] %NAME% )
goto :eof

REM ============================================================================
:report
set "REL=%~1"
if exist "%COMFY%\%REL%" ( for %%I in ("%COMFY%\%REL%") do echo   OK    %REL%  ^(%%~zI bytes^) ) else ( echo   FALTA %REL% )
goto :eof
