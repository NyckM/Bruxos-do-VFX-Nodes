<img width="832" height="480" alt="ChatGPT Image 30 de jun  de 2026, 11_28_48" src="https://github.com/user-attachments/assets/6745a8ce-00ce-4915-a60d-ed1354099311" />
﻿# ComfyUI Bruxos do VFX

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

