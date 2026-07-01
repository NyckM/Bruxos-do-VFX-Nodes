<img width="832" height="480" alt="ChatGPT Image 30 de jun  de 2026, 11_28_48" src="https://github.com/user-attachments/assets/6745a8ce-00ce-4915-a60d-ed1354099311" />

Workflow para remover objetos e pessoas

https://github.com/user-attachments/assets/1575f97f-34b9-492a-accb-818e97a6cbde

resultados, tempo média de 96 frames em 204 segundos

https://github.com/user-attachments/assets/d1e47486-ac3b-4030-be42-56a0f16b0128


Essa ferramenta foi desenvolvida especificamente para duas produções: Dr Monstro de Marcos jorge e Alice júnior 2 de Gil Baroni.
dois longas metragens com alguns vfx realizados aqui na produtora bruxos do VFX, onde a demanda gerou a necessidade de criar a ferramenta de remoção de objetos para auxiliar na composição e integração das cenas. 
## Instalacao fácil

Usando o Bat dentro da pasta de instalação do ComfyUI, assim ele já baixa os modelos e o custom node automaticamente para você.
Versão para Runpod, ainda em desenvolvimento

## Instalacao

Copie a pasta `comfyui-bernini-long-conditioning` para:

```text
ComfyUI/custom_nodes/
```

https://huggingface.co/neuregex/Bernini-R-GGUF/tree/main
Modelo GGufs
que vão instalados na pasta ComfyUI\models\unet

https://huggingface.co/Cyph3r/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16
lora que vai na pasta ComfyUI\models\lora 

https://huggingface.co/Osrivers/umt5_xxl_fp8_e4m3fn_scaled.safetensors/tree/main
Umt clip  que vai na pasta ComfyUI\models\clip

https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1_VAE_bf16.safetensors
vae que vai na pasta ComfyUI\models\vae

Depois reinicie o ComfyUI.

### ComfyUI Bruxos do VFX


https://github.com/user-attachments/assets/e35f6caa-1b53-46bf-860d-90db0eb51b0f


Custom nodes para usar Bernini/Wan com videos maiores que o limite comum de 81 frames sem criar um sampler Bernini novo.

O node principal troca a logica de um `Bernini Conditioning` unico por condicionamentos em chunks:

<img width="226" height="557" alt="image" src="https://github.com/user-attachments/assets/934fd9b3-087f-47db-aeb2-6ca01967c556" />

### Bernini Long Condition

Para cada chunk, ele injeta:

- `context_latents: [encoded_chunk]`
- se `tail_memory=True`, a partir do segundo chunk: `context_latents: [encoded_chunk, tail_latent]`
- `context: {"video": encoded_chunk}` para compatibilidade com nodes que ainda leem o contexto antigo do Bernini
sequential vs context_window — qual é a diferença?

Ambos controlam como o node processa um vídeo maior do que um único chunk, mas fazem isso de formas bem diferentes.

O sequential processa o vídeo em chunks, um após o outro, avançando chunk_size − overlap frames a cada etapa. A opção tail_memory reutiliza alguns frames já editados do chunk anterior como contexto para o próximo, ajudando a manter a continuidade. É o modo mais simples e econômico em VRAM, porém, como o modelo é executado novamente em cada janela sobreposta, usar um chunk_size pequeno com um overlap muito grande pode aumentar drasticamente o número de execuções (por exemplo: chunk=17 e overlap=16 → avanço de apenas 1 frame → 61 passagens). Importante: o modo de máscara bbox não é utilizado no sequential; ele faz fallback automaticamente para inpaint.

O context_window trata o vídeo inteiro como uma única geração, mas fornece ao modelo uma janela deslizante com os frames vizinhos como contexto. Em vez de unir várias execuções independentes, o modelo leva em consideração os frames ao redor para manter a consistência temporal ao longo de todo o vídeo. Esse modo geralmente produz movimentos e continuidade mais suaves, além de ser o único em que a máscara bbox funciona, permitindo o ganho de desempenho por meio do recorte da região de interesse.

Regra prática:

Vídeos curtos que cabem em uma única execução: ambos funcionam, mas context_window oferece melhor consistência e habilita o uso de bbox.
Vídeos muito longos ou com VRAM limitada: use sequential, preferencialmente com um chunk_size grande e um overlap pequeno para reduzir a quantidade de passagens.
Quer economizar tempo e VRAM usando o recorte por bbox? Então é necessário usar context_window. No modo sequential, o bbox é ignorado.
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
