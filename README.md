# ComfyUI Bruxos do VFX

Custom nodes para usar Bernini/Wan com videos maiores que o limite comum de 81 frames sem criar um sampler Bernini novo.

O node principal troca a logica de um `Bernini Conditioning` unico por condicionamentos em chunks:

```python
context["video"] = vae.encode(video_chunk)

positive = conditioning_set_values(
    positive,
    {"context_latents": [encoded_chunk, optional_tail_memory]}
)
```

## Nodes

### Bernini Long Condition

Entradas:

- `positive`
- `negative`
- `vae`
- `source_video`
- `width`
- `height`
- `chunk_size`, padrao `81`
- `overlap`, padrao `5`
- `batch_size`, padrao `1`
- `tail_memory`, padrao `True`
- `tail_frames`, padrao `5`

Saidas:

- `positive_chunks`
- `negative_chunks`
- `latent_chunks`
- `video_chunks`
- `chunk_ranges`
- `chunk_count`

Para cada chunk, ele injeta:

- `context_latents: [encoded_chunk]`
- se `tail_memory=True`, a partir do segundo chunk: `context_latents: [encoded_chunk, tail_latent]`
- `context: {"video": encoded_chunk}` para compatibilidade com nodes que ainda leem o contexto antigo do Bernini

### Bernini Long Chunk Select

Seleciona um indice de chunk e retorna:

- `positive`
- `negative`
- `latent`

Use isso para renderizar cada chunk com o fluxo Bernini atual.

### Bernini Long Video Merge

Recebe uma lista de videos renderizados e faz blend linear no overlap.

### Bernini Long Empty Video Chunks / Append Video Chunk

Utilitarios para montar manualmente a lista de chunks renderizados antes do merge.

### Bernini Long Info

Mostra os ranges de frames de cada chunk.

## Instalacao

Copie a pasta `comfyui-bernini-long-conditioning` para:

```text
ComfyUI/custom_nodes/
```

Depois reinicie o ComfyUI.

## Workflow sugerido

1. Substitua `Bernini Conditioning` por `Bernini Long Condition`.
2. Conecte `positive_chunks`, `negative_chunks` e `latent_chunks` em `Bernini Long Chunk Select`.
3. Renderize o indice `0`, depois `1`, depois `2`, etc., usando o render/sampler Bernini que voce ja usa.
4. Junte os videos renderizados com `Bernini Long Empty Video Chunks`, `Bernini Long Append Video Chunk` e `Bernini Long Video Merge`.

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

Formula usada:

```python
aligned = ((target - 1 + 3) // 4) * 4 + 1   # proximo 4n+1
# ... gera com 'aligned' frames ...
saida = saida[:target]                       # corta de volta
```

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

### Node novo: Bernini Region Mask

Prepara a mascara antes de conectar no `region_mask`. Aceita uma `MASK` ou,
opcionalmente, um `IMAGE` colorido em `image_mask` (qualquer pixel nao-preto
vira regiao — igual ao `SCAIL2ColoredMask`). Faz `grow`/`blur`/`threshold`/
`invert`. Saida: `MASK`.

Fluxo tipico:

```text
SAM3_VideoTrack / SAM2 / desenho  ->  (IMAGE ou MASK)
   -> Bernini Region Mask (grow/blur/threshold)
      -> Bernini Infinity.region_mask  (mask_mode = inpaint ou bbox)
```

### Encaixe da imagem: stretch vs crop (`resize_mode`)

Quando o aspect ratio do video de entrada nao bate com o `width`/`height`
pedido, o `Bernini Infinity` agora deixa voce escolher no widget
`resize_mode`:

- `stretch` (padrao): estica pro tamanho exato, **sem cortar** nada das
  bordas (pode distorcer um pouco se a proporcao for bem diferente).
- `crop`: corta as bordas (center crop) pra preservar a proporcao — era o
  comportamento antigo, que causava aquele "leve crop nas laterais".

Bonus: a mascara passou a usar o **mesmo** encaixe da fonte, entao em `crop`
ela continua alinhada com o video (antes a mascara esticava e a fonte
cortava, desalinhando levemente a regiao).

Dica pra nao distorcer **nem** cortar: ajuste `width`/`height` pra mesma
proporcao do video de entrada (ex.: fonte 1920x1080 -> use 16:9 como 832x468
arredondado pra multiplo de 16).

---

## Novidade 0.3.0 — Node novo: FaceStitchUpscale

Cola o rosto de volta no video depois de um upscale, usando os `face_bboxes`
do node "Pose and Face Detection" (ComfyUI-WanAnimatePreprocess / kijai).
Resolve o problema de nao conseguir encaixar o rosto upscalado frame a frame.

**Por que funciona:** o crop do WanAnimate e, por frame,
`face = frame[y1:y2, x1:x2]` -> `cv2.resize(face, (512,512))`, e `face_bboxes`
guarda exatamente `(x1,y1,x2,y2)` em pixels do frame original. Este node faz o
inverso: redimensiona o rosto upscalado de volta para `(x2-x1, y2-y1)` e compoe
na posicao certa, com borda suave. E 100% reversivel.

**Fiacao:**

```text
Pose and Face Detection
   |- face_images --> [seu upscale WAN] --> upscaled_faces -.
   '- face_bboxes ------------------------> face_bboxes ----+
                                                            |
   video (original OU ja upscalado) ------> target_frames --+
                                                            v
                                                   FaceStitchUpscale
                                                      |- images   (video final)
                                                      '- face_mask (onde colou)
```

**Parametros:**

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

Backends em camadas: PyAV (vem com o ComfyUI) -> imageio-ffmpeg -> OpenCV.
Se for usar imageio como backend: `pip install imageio imageio-ffmpeg`.
O audio no MP4 e best-effort (precisa de PyAV ou ffmpeg no PATH).

### FaceStitchUpscale: color_match
Adicionado `color_match` (off/mean/mean_std) pra casar tom/exposicao do rosto
novo com a regiao original e acabar com a mesclagem feia. E `mask_shape`
(ellipse/rectangle) + `external_mask` (ligue uma mascara SAM2 se quiser).

### Face Crop Expand (Bruxos)
Re-crop dos rostos com expansao controlavel (`scale`, `top_extra` p/ cabelo),
crop quadrado real (sem estica-desestica). Gera `face_images` + `face_bboxes`
expandidos — da mais contexto pro upscaler entender o sujeito.

---

## Novidades 0.5.1

- **Botao de upload** no node Load Video (Bruxos), via extensao JS em `web/`
  (igual ao "choose video to upload" do VideoHelperSuite). Envia pra
  `ComfyUI/input` e ja seleciona no widget.
- **Explicacao em portugues** em todos os nodes: `DESCRIPTION` (aparece ao
  passar o mouse no node) + `tooltip` em cada parametro dos nodes do pacote.

---

## Novidades 0.6.0 — Prompt Guide multimodelo

Node novo **Prompt Guide (Bruxos)** (`Bruxos do VFX/Prompt`), inspirado no
Bernini Prompt Guide do Deno, mas com varios modelos. Cada tarefa e um "comando"
com seu system prompt, e cada modelo tem negativos prontos.

Modelos: **Bernini, Wan 2.2, Wan 2.1, LTX 2.3 (Edit Anything), Seedance 2**.

- `model`: filtra as tarefas e negativos disponiveis (via extensao JS).
- `task`: o comando; auto-preenche o `system_prompt` (editavel).
- `negative_preset`: negativo pronto (ex: Wan2.2 oficial em chines e EN).
- `prompt`: sua instrucao.
- `clip` (opcional): se ligado, sai CONDITIONING; senao, use as saidas de texto.
- Saidas: `positive`, `negative` (CONDITIONING), `positive_text`, `negative_text`.

As tarefas do LTX 2.3 (Add / Remove / Replace / Style / Motion Transfer /
Reference Add / Reference Replace) seguem o guia oficial do LoRA Edit Anything
(ex: Remove = 4-10 palavras; Replace = descreva o antigo e o novo; Style =
"Convert the video into a <STYLE> style").

Obs.: os presets do Wan usam o negativo oficial difundido. Os de Seedance 2 e
alguns do Bernini sao pontos de partida curados e 100% editaveis no node.
