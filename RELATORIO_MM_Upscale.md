# Relatorio — MM_Upscale.json + melhorias Bruxos do VFX

## 1) Como o workflow funciona hoje (resumo tecnico)

Carreguei o `MM_Upscale.json` (96 nodes) e tracei a cadeia inteira. E um
pipeline de upscale iterativo estilo "MasterMind" rodando sobre Wan 2.2:

**Modelos / loras instalados no JSON:**
- `Wan2_2-T2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors` (UNet)
- LoRA `Wan2.2-T2V-A14B-4steps-lora-250928_low_noise_model.safetensors` (1.0)
- LoRA `stock_photography_wan22_LOW_v1.safetensors` (0.5)
- CLIP `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- VAE `Wan2.1_VAE.pth`
- Upscale model `4x_UniversalUpscalerV2-Neutral_115000_swaG.pth`

**Cadeia principal de cada iteracao:**

```
VHS_LoadVideo
    -> ImageUpscaleWithModel (4x Universal Upscaler V2)
       -> ImageScale (Lanczos -> 3840x2160)
          -> ImageSmartSharpen+ (5, 0.4, 1.5, 0.5)
             -> ImageAddNoise (strength 0.01)
                -> UltimateSDUpscaleNoUpscale (Wan, denoise 0.2, 35 steps, res_2s, beta57,
                                               tile 480x480, padding 32, batch_size 1)
                   -> save img seq + VHS_VideoCombine
```

A iteracao re-le os PNGs salvos (`VHS_LoadImagesPath`) e roda de novo,
acumulando refino.

**O que o Florence faz hoje (sua duvida):**

`Florence2Run` esta ligado em `ImageFromBatch(index=0, length=1)`. Ou seja:

> O Florence captiona **um unico frame** (o frame 0 do video) e gera **um
> prompt geral** que vale pro video todo. Nao e por frame.

Depois passa por `StringReplace("image" -> "high quality video")`,
`JoinStrings(custom_prompt + caption)` e cai no `CLIPTextEncode` positivo.

## 2) Onde o video pode perder frames

Dois pontos:

**A) Wan VAE comprime tempo ~4x.** N frames -> `((N-1)//4)+1` latentes, e o
decode devolve `(T_lat-1)*4 + 1` frames. So sobrevivem comprimentos **4n+1**
(1, 5, 9, ..., 81, 85, ..., 109, 113, ...). Por isso `111 entra -> 28
latentes -> 109 sai`. Este e o mesmo problema que voce ja resolveu dentro do
`Bernini Infinity`.

No `MM_Upscale` o `UltimateSDUpscaleNoUpscale` chama o Wan internamente,
entao a regra 4n+1 vale tambem pra cada batch que entra nele. Com
`batch_size = 1` por frame e modelo T2V isso pode passar batido (cada chamada
e 1 frame -> 1 latente -> 1 frame), mas se voce subir `batch_size` (ou
trocar pra um sampler que use janela temporal), perde os ultimos frames de
cada batch que nao for 4n+1.

**B) Re-leitura por pasta.** As iteracoes salvam PNGs em
`save_upscale_images_NN/sampling_iterations` e a proxima iteracao usa
`VHS_LoadImagesPath` pra reler. Se um PNG falhar/sumir, a contagem cai
silenciosamente.

## 3) O que entreguei na v0.4.0 do pacote

Tres nodes novos somados ao `ComfyUI-Bruxos-do-VFX`:

### `Pad to 4n+1 (Bruxos)` + `Trim 4n+1 back to N (Bruxos)`

Resolvem (A). Mesmo principio do `Bernini Infinity`, mas isolados pra
envolver **qualquer** etapa Wan (no caso, o `UltimateSDUpscaleNoUpscale`).

- `Pad to 4n+1`: faz padding temporal **espelhado** (ping-pong, sem frame
  congelado, igual ao truque do Kijai no Wan Animate). Saidas:
  `images, original_count, padded_count`.
- `Trim 4n+1 back to N`: corta de volta ao numero original de frames.
  Sobrevive a perda de frames no meio do caminho — se o pipeline devolveu
  menos do que pediu, devolve `min(disponivel, original)` em vez de quebrar.

**Onde encaixar no MM_Upscale:**

```
VHS_LoadVideo.IMAGE -> [Pad to 4n+1] -> resto do upscale ... -> [Trim back] -> VHS_VideoCombine
                              \                                        ^
                               original_count -------------------------/
```

Resultado: `111` entra, `111` sai. Sem reescrever o sampler.

### `Qwen-VL Caption (Bruxos)` — substituto do Florence

Substitui o par `DownloadAndLoadFlorence2Model` + `Florence2Run`. Saida e
**STRING** unica, igual ao Florence, entao plugga direto no
`StringReplace` + `JoinStrings` que ja existem.

Dois modos:

- `single_frame` (= comportamento atual do Florence): captiona 1 frame.
- `keyframes_merge` (recomendado): amostra N keyframes espacados no video
  inteiro e pede ao Qwen **um prompt unico**. Pega mudancas de
  cenario/roupa/luz que o frame 0 nao mostra. E o ganho real vs Florence.

Modelos suportados: `Qwen/Qwen2.5-VL-3B-Instruct` (default, leve),
`Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2-VL-2B-Instruct`,
`Qwen/Qwen2-VL-7B-Instruct`. Baixa via HuggingFace na primeira execucao.

Instrucao default ja vem afiada pra prompt de upscale (sujeitos, roupa,
materiais, luz, camera, paleta, movimento — sem meta).

**Requisitos:** `pip install -U transformers accelerate pillow` (transformers
>= 4.45 pro Qwen2.5-VL).

## 4) FaceStitchUpscale neste workflow

O `MM_Upscale.json` NAO tem hoje a cadeia de deteccao de rosto do
WanAnimate. Pra colar o `FaceStitchUpscale` (que ja esta no pacote desde a
v0.3.0), voce precisa adicionar uma branch paralela. O ponto-chave e que o
video do `target_frames` vai ser o **upscalado** (saida do USDU/Trim), e o
`bbox_scale` precisa refletir a razao entre o tamanho upscalado e o original.

**Branch nova a ligar dentro do MM_Upscale:**

```
                                        ┌── face_images ─► [pequeno upscale Wan no rosto] ─► upscaled_faces ─┐
VHS_LoadVideo ─► Pose and Face Detection┤                                                                    │
                                        └── face_bboxes ─────────────────────────────────────► face_bboxes ──┤
                                                                                                             │
saida do Trim 4n+1 (video upscalado) ───────────────────────────────► target_frames ─────────────────────────┤
                                                                                                             ▼
                                                                                              FaceStitchUpscale
                                                                                                  ├─ images ─► VHS_VideoCombine
                                                                                                  └─ face_mask
```

Configuracao:

- `bbox_scale` = `output_width / source_width`. No JSON: `ImageScale` esta
  setado pra `3840x2160`. Se sua fonte e 1920x1080 (1080p), use `2.0`. Se e
  4K cropado pra 1080p e voce vai pra 4K, use `2.0` tambem (o que importa e
  o ratio entre o frame que entra no `target_frames` e o frame original do
  qual os bboxes foram medidos).
- `feather`: comece com `24`. Pra rosto pequeno, abaixe.
- `mask_expand`: `0` na maioria dos casos. Se ver costura, `+4` a `+8`.

Voce nao precisa ainda do `Pose and Face Detection` upscalado: rode ele
**no video original** (antes do upscale), guarde `face_bboxes`, e use
`bbox_scale` pra projetar pro frame upscalado.

> Observacao: nao reescrevi o JSON pra voce porque com 96 nodes, IDs e links
> auto-gerenciados pelo Comfy, edicao manual e fragil. O caminho seguro e
> arrastar os 3 nodes novos (`Pose and Face Detection`, sua etapa de upscale
> do rosto, `FaceStitchUpscale`) na UI e ligar como acima.

## 5) Ordem de instalacao

1. Substitua `nodes.py`, `README.md`, `pyproject.toml` na pasta
   `ComfyUI/custom_nodes/ComfyUI-Bruxos-do-VFX/`.
2. Apague o `__pycache__` da pasta (pra nao carregar bytecode antigo).
3. `pip install -U transformers accelerate pillow` (so se for usar o Qwen).
4. Reinicie o ComfyUI.

Os 4 nodes novos aparecem em:

- `Bruxos do VFX/Face` → **FaceStitchUpscale** (ja vinha da v0.3.0)
- `Bruxos do VFX/Video` → **Pad to 4n+1 (Bruxos)**, **Trim 4n+1 back to N (Bruxos)**
- `Bruxos do VFX/Caption` → **Qwen-VL Caption (Bruxos)**

## 6) Resumo das perguntas

| Pergunta                                         | Resposta                                                                                       |
|--------------------------------------------------|------------------------------------------------------------------------------------------------|
| Florence faz prompt por frame ou geral?          | **Geral**, a partir do frame `0` (`ImageFromBatch(index=0, length=1)`).                       |
| Como nao perder frames no Wan?                   | `BruxosPad4n1` antes + `BruxosTrim4n1` depois. Espelho ping-pong, sem frame congelado.        |
| Como inserir o `FaceStitchUpscale` no MM_Upscale?| Branch paralela com `Pose and Face Detection` no video original + `bbox_scale = out_w/src_w`. |
| Como substituir o Florence por Qwen?             | `BruxosQwenVLCaption` (STRING out, drop-in). Use `keyframes_merge` pra ganhar vs frame unico. |
