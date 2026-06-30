(https://github.com/user-attachments/files/29526644/bruxos_runpod_setup.sh)<img width="832" height="480" alt="ChatGPT Image 30 de jun  de 2026, 11_28_48" src="https://github.com/user-attachments/assets/6745a8ce-00ce-4915-a60d-ed1354099311" />

Workflow para remover objetos e pessoas

https://github.com/user-attachments/assets/1575f97f-34b9-492a-accb-818e97a6cbde

resultados, tempo média de 96 frames em 204 segundos

https://github.com/user-attachments/assets/d1e47486-ac3b-4030-be42-56a0f16b0128

## Instalacao fácil

Usando o Bat dentro da pasta de instalação do ComfyUI, assim ele já baixa os modelos e o custom node automaticamente para você
versão para Runpod, ainda em desenvolvimento
[Uploading bruxos_@echo off
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
setup_windows.bat…]()


### ComfyUI Bruxos do VFX

Custom nodes para usar Bernini/Wan com videos maiores que o limite comum de 81 frames sem criar um sampler Bernini novo.

O node principal troca a logica de um `Bernini Conditioning` unico por condicionamentos em chunks:

<img width="226" height="557" alt="image" src="https://github.com/user-attachments/assets/934fd9b3-087f-47db-aeb2-6ca01967c556" />

### Bernini Long Condition

Para cada chunk, ele injeta:

- `context_latents: [encoded_chunk]`
- se `tail_memory=True`, a partir do segundo chunk: `context_latents: [encoded_chunk, tail_latent]`
- `context: {"video": encoded_chunk}` para compatibilidade com nodes que ainda leem o contexto antigo do Bernini

## Novidades 0.2.0 / 0.2.1

### Correcao de frames (4n+1) — fim do "111 entra, 109 sai"

O Wan VAE comprime o tempo ~4x: `N` frames viram `T_lat = ((N-1)//4)+1`
latentes e o decode devolve `(T_lat-1)*4 + 1` frames. So sobrevivem
comprimentos `4n+1` (1, 5, 9, ..., 109, 113, ...). Por isso `111 -> 28
latentes -> 109 frames`.

O `Bernini Infinity` agora, **internamente**, arredonda o alvo pro proximo
`4n+1` com **padding espelhado** (reflexao ping-pong, igual ao truque do
Kijai no Wan Animate, sem frame congelado), gera, e no final **corta de volta
ao numero de frames que voce pediu**. Vale pros modos `sequential` e
`context_window`, e por chunk no sequential (o ultimo chunk costumava cair
fora do grid). Resultado: `111` entra, `111` sai.
Nao precisa configurar nada — e automatico. Os logs mostram, por ex.:
`padding temporal 111->113 (4n+1, espelhado)`.

### Mascara: gerar so na area selecionada

Tres campos novos no fim do `Bernini Infinity` (ficam por ultimo de proposito
pra nao deslocar widgets de workflows ja salvos):

- `mask_mode`: `off` | `inpaint` | `bbox`
  - `inpaint`: gera o frame inteiro, mas so a area da mascara muda; o resto
    volta a ser a fonte (composite em pixels — robusto, nao depende de
    `noise_mask` 5D do Comfy).
  - `bbox`: recorta na bounding box da mascara, gera **so esse retangulo em
    resolucao menor** (mais rapido e menos VRAM) e cola de volta com a
    mascara. E o modo que de fato **otimiza** a geracao. Disponivel no modo
    `context_window` sem janelas; com janelas/sequential cai pra `inpaint`.
- `mask_grow`: dilata (+) ou contrai (-) a mascara, em pixels.
- `mask_blur`: suaviza a borda (feather) pra colagem sem costura dura.
- entrada opcional `region_mask` (`MASK`).

A mascara pode vir de qualquer lugar (SAM2/SAM3, desenho manual, etc.).
Para o estilo do `Scail2Color` (mascara colorida por objeto saindo como
`IMAGE`), use o node abaixo.

## Novidade 0.3.0 — Node novo: FaceStitchUpscale

Cola o rosto de volta no video depois de um upscale, usando os `face_bboxes`
do node "Pose and Face Detection".

- `target_frames` — video onde o rosto sera colado.
- `upscaled_faces` — rostos depois do upscale (mesma contagem de frames).
- `face_bboxes` — saida `face_bboxes` do Pose and Face Detection (conecta direto;
  o tipo coringa ignora o typo "BBOX," do kijai).
- `bbox_scale` — `1.0` se cola no video original; se o video foi upscalado Nx,
  use `N` (ex.: `2.0`) para os bboxes (medidos em 1x) casarem com o frame Nx.
- `feather` — suavizacao da borda em px (24 e um bom inicio).
- `mask_expand` — expande (+) / encolhe (-) a area colada em px.
- `blend` — `1.0` substitui o rosto inteiro; `<1.0` mistura com o original.

**Dois cenarios:**

- *So detalhar o rosto* (video fica na resolucao original): `target_frames` =
  video original, `bbox_scale = 1.0`. Mesmo voltando ao tamanho pequeno do bbox,
  o WAN recupera detalhe/nitidez (estilo FaceDetailer).
- *Video upscalado + rosto em alta de verdade*: faca o upscale do video todo
  (ex.: 2x), ligue ESSE video em `target_frames` e use `bbox_scale = 2.0`.

---

## Novidades 0.4.0

Adicionados tres nodes:

- **Pad to 4n+1 (Bruxos)** e **Trim 4n+1 back to N (Bruxos)** — envolva o
  bloco Wan do seu workflow com esses dois nodes pra que `111` entre e `111`
  saia. Padding por espelhamento ping-pong (sem frame congelado).
- **Qwen-VL Caption (Bruxos)** — substituto direto do `Florence2Run`. Saida
  `STRING`. Modos `single_frame` (= comportamento Florence) e
  `keyframes_merge` (amostra varios frames e gera UM prompt unificado).
  Requer `transformers>=4.45 accelerate pillow`.

Veja `RELATORIO_MM_Upscale.md` para a integracao detalhada no workflow
`MM_Upscale.json`.

---

## Novidades 0.5.0 — Video I/O (nodes 2.0)

Dois nodes no estilo do VideoHelperSuite, mas compativeis com o tipo `VIDEO`
nativo dos nodes 2.0 do ComfyUI. Ficam em `Bruxos do VFX/Video`.

### Load Video (Bruxos)
Importa video com as opcoes do VHS e ja entrega o tipo nativo.
- Entradas: seletor `video` (pasta `ComfyUI/input`) + `video_path` (caminho
  absoluto, ex. `C:\...\clip.mp4`, tem prioridade), `force_rate`,
  `custom_width`, `custom_height`, `frame_load_cap`, `skip_first_frames`,
  `select_every_nth`.
- Saidas: `images` (IMAGE), `video` (VIDEO nativo p/ os nodes 2.0), `audio`
  (AUDIO), `fps` (FLOAT), `frame_count` (INT), `video_info` (STRING/JSON).

### Save Video (Bruxos)
Exporta com mais opcoes que o VHS, criando pastas.
- `save_mp4` + `codec` (h264/h265/vp9/prores) + `crf` + `pix_fmt`.
- `save_png_sequence` + `png_in_subfolder` (pasta dedicada) + `png_prefix`.
- `date_subfolder` (cria subpasta com a data), `pingpong`, `audio` opcional.
- Saidas: `mp4_path`, `png_folder`.
- 
## Instalacao

Copie a pasta `comfyui-bernini-long-conditioning` para:

```text
ComfyUI/custom_nodes/
```

Depois reinicie o ComfyUI.

## Por que nao ha um Bernini Long Sampler aqui?

Porque, pelo comportamento descrito, o Bernini ja passa contexto pelo conditioning e o patch Wan ja aceita `context_latents` como lista. O ponto robusto e gerar varios conditionings corretos:

```python
context_latents = [
    encoded_chunk,
    tail_latent,
]
```

Assim voce aproveita a arquitetura nativa em vez de clonar um sampler inteiro.

Um executor automatico pode ser adicionado depois, mas ele precisa conhecer os nomes e chamadas exatas dos nodes/classes Bernini instalados na sua maquina. Este pacote deixa a parte importante isolada e compativel com o workflow existente.


---

🙏 Acknowledgements
workflow usou como base aguns nodes criados pelo KIjai, e modelos do Bernini:
agradecemos aos autores e a comunidade.

📄 License
Apache License 2.0.
