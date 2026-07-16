# ComfyUI Bruxos do VFX

Custom nodes para VFX de vídeo com Wan 2.2 / Bernini: remoção de objetos e pessoas, upscale por batch, face swap, MoCha, tiling em resolução maior, comparação A/B e utilitários — tudo em português, feitos pra produção real.

<img width="832" height="480" alt="Workflow para remover objetos e pessoas" src="https://github.com/user-attachments/assets/6745a8ce-00ce-4915-a60d-ed1354099311" />

## Por que existe

Desenvolvido na **Bruxos do VFX** para dois longas-metragens — *Dr. Monstro* (Marcos Jorge) e *Alice Júnior 2* (Gil Baroni) — onde a demanda de composição e integração de cenas gerou a necessidade de uma ferramenta própria de remoção de objetos sobre Bernini/Wan.

**Resultados:**


Remoção de objetos — média de 96 frames em 204 segundos (832×480):

https://github.com/user-attachments/assets/1575f97f-34b9-492a-accb-818e97a6cbde


https://github.com/user-attachments/assets/d1e47486-ac3b-4030-be42-56a0f16b0128


Remoção em **1920×1080 Full HD** — 39 frames em 325 segundos:

https://github.com/user-attachments/assets/55426421-bdbd-4c04-a727-6dfc95839e19

https://github.com/user-attachments/assets/2c0222a3-7b5f-4c4f-b430-c3fb6a178218

---

## Instalação

### Fácil (recomendado)

Rode o instalador de dentro da pasta do node — ele instala as dependências **e baixa os modelos** nas pastas certas:

```text
ComfyUI\custom_nodes\ComfyUI-Bruxos-do-VFX\install.bat     (Windows)
bash ComfyUI/custom_nodes/ComfyUI-Bruxos-do-VFX/install.sh (Linux / RunPod)
```

É **idempotente**: modelos já baixados são pulados, então pode rodar de novo pra retomar um download interrompido.

> **O instalador nunca mexe em `torch`, `numpy`, `triton`, `xformers` ou `flash-attn`.** Ele detecta a CUDA do seu torch e instala o `onnxruntime-gpu` que **casa** com ela (cu12x → build CUDA 12; cu13x → build CUDA 13).

### Manual

Copie a pasta para `ComfyUI/custom_nodes/ComfyUI-Bruxos-do-VFX` e reinicie.

### Modelos

O `install.bat` / `install.sh` baixa tudo automaticamente. Se preferir na mão:

| Modelo | Onde baixar | Pasta destino |
|---|---|---|
| **Bernini-R INT8 ConvRot** (high + low) | [Comfy-Org/Bernini-R](https://huggingface.co/Comfy-Org/Bernini-R/tree/main/diffusion_models) | `models/diffusion_models` |
| **LoRA LightX2V 4-step** (high + low) | [rzgar/Bernini-R-LightX2V-4step-loras](https://huggingface.co/rzgar/Bernini-R-LightX2V-4step-loras) | `models/loras` |
| Text encoder (umt5-xxl fp8) | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) | `models/text_encoders` |
| **VAE de vídeo** (Wan 2.1 bf16) | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1_VAE_bf16.safetensors) | `models/vae` |

> ⚠️ **Use o VAE de VÍDEO** (`Wan2_1_VAE_bf16`). Um VAE `imageonly` / `upscale2x` devolve o tensor em outro layout e o vídeo sai **preto/quebrado**.

> Baixe por CLI/gerenciador, não pelo navegador — download incompleto gera arquivo corrompido.

**Por que INT8 ConvRot:** os pesos e ativações rodam em 8 bits nos tensor cores (`torch._int_mm`), não é desquantização on-the-fly como o GGUF. O ConvRot rotaciona os pesos antes de quantizar pra eliminar outliers. Precisa de ComfyUI recente + Triton.

**Configuração com as LoRAs LightX2V:** `cfg = 1.0` e `steps = 6` com `split_step = 4`.

---

## Nodes

<img width="226" height="557" alt="Bernini Long Condition" src="https://github.com/user-attachments/assets/934fd9b3-087f-47db-aeb2-6ca01967c556" />

### Bernini / Geração
- **Bernini Infinity** — o renderer principal para vídeos maiores que o limite de 81 frames, sem precisar de um sampler novo. Injeta `context_latents` por chunk (com `tail_memory` opcional). Inclui correção automática de frames **4n+1**, gerenciamento de VRAM, máscara de região e `guidance_mode` (off / multi).
- **Bernini Region Mask** — normaliza máscara colorida (SAM2/SAM3/Scail2Color) em B/W, com invert/grow/blur.
- **Bernini Prompt Enhancer** — reescreve sua instrução via Qwen-VL **local**, com visão de keyframes. Self-text reasoning do paper do Bernini.
- **First-Frame CoT: Extrair / Compor** — edite o primeiro frame como imagem e propague pro vídeo (self-vision-text do paper).
- **Bernini Long** *(Conditioning / ChunkSelect / VideoMerge / AppendVideoChunk / EmptyVideoChunks / Info)* — helpers de vídeo longo.
- **FaceStitchUpscale** — cola o rosto upscalado de volta no vídeo usando os `face_bboxes`.
- **Editor de Pontos SAM3** — clique **verde = selecionar**, **roxo = negar** sobre o frame.

### Bernini Infinity Tiled *(novo)*
- **Bernini Infinity Tiled (Bruxos)** — roda o Bernini COMPLETO **por ladrilho em pixels** para alcançar resoluções maiores em **qualquer função** (remover, modificar, gerar, refinar). Cada ladrilho recebe o próprio pedaço do vídeo-fonte — a posição nunca se perde. A **costura viva** cola o resultado já gerado dos vizinhos na faixa de sobreposição (máscara zerada ali) → os ladrilhos casam em cor e conteúdo.

  Três modos de máscara por ladrilho:
  - `off` — modifica o shot todo (guiado pelo prompt).
  - `inpaint` — só a área da máscara muda; tiles sem máscara são pulados (`pular_tiles_vazios`).
  - `bbox` — **duplo recorte**: dentro do tile, recorta ainda na bounding box da máscara e roda só essa área. É o modo mais rápido quando o objeto a remover é pequeno relativo ao tile.

  **Ganho real do `bbox` medido** (objeto = 10% do tile, shot 1664×960, grade 2×2):

  | Modo | Área processada | Velocidade relativa |
  |---|---|---|
  | `off` | 2.334.720 px (4 tiles) | 38× mais lento |
  | `inpaint` | 583.680 px (1 tile) | 9,6× mais lento |
  | **`bbox`** | **60.928 px (1 bbox)** | **referência** |

  Custo honesto: N ladrilhos = N renders do Bernini, cada um menor. Não é "mais rápido" no caso geral: é "cabe na VRAM e sem mosaico". Arquitetura inspirada no [TiledWan](https://github.com/Baverne/comfyUI-TiledWan) (Baverne), reimplementada do zero.

### Tiles (utilitários de corte/costura)
- **Tile Split / Select / Accumulate / Merge (Bruxos)** — corta a imagem/vídeo em ladrilhos pela **contagem** (2×2, 8×8…), com tamanho calculado automaticamente e alinhado ao múltiplo de 16. Para uso com For Loop + seu próprio sampler. `1×1` = passthrough. Merge detecta upscale automaticamente.
- **Wan Tiled Sampler (Bruxos)** — guider step-fused (estilo Deno/LTX): corta o latente em tiles e funde a cada passo de denoise com janela Hann complementar (soma = 1, sem emenda). Saída `GUIDER` → `SamplerCustomAdvanced`. ⚠️ Com `context_latents` (V2V Bernini) desliga automaticamente com aviso — use o **Bernini Infinity Tiled** nesse caso.

### MoCha
- **Mocha Embeds (Bruxos)** — substitui o `MochaEmbeds` do WanVideoWrapper. Corrige o bug de frames do original (111 → 109, perdendo 2; aqui o padding é **espelhado** e você corta de volta sem perder nada). Aceita **MASK ou IMAGE colorida** (SAM3/SCAIL/FaceFusion). O MoCha usa **uma única máscara** pro vídeo inteiro: se você ligar uma por frame, o node reduz por **união** ou **primeiro frame**. Inclui `tiled_vae`, limpeza de memória e cronômetro.
- **Mocha Info (Bruxos)** — calcula antes de rodar (sem gastar VRAM) os frames alinhados, o `seq_len` e o plano de blocos.

### Face / Troca de rosto *(precisa das libs ONNX)*
- **FaceFusion Swap (Bruxos)** — troca de rosto **100% local** (ONNX, sem API). Aceita imagem única ou vídeo inteiro, 13 swappers (`hyperswap_1c_256` recomendado), `pixel_boost` até 1024, seleção `one`/`many`/`reference` e máscaras combináveis. Sai com uma **MASK dos rostos** que liga direto no `region_mask` do Bernini Infinity.
- **FaceFusion Detectar Rostos (Bruxos)** — preview com caixas e landmarks, MASK por frame e contagem.

### Vídeo
- **Load Video** / **Save Video** — equivalentes ao VHS com tipo `VIDEO` nativo, preview já cortado por skip/cap/nth/force_rate, mais controle de codec/CRF. O **Load Video** tem widget **`reverse`** (inverte a ordem dos frames, aplica depois de skip/cap/nth). O **Save Video** diagnostica tensores malformados e denuncia NaN/vídeo preto em vez de gravar em silêncio.
- **Comparar Vídeos A/B** — player embutido (cortina, lado a lado, diferença, alternar).
- **Prever BBox da Máscara** — desenha a caixa que o modo `bbox` recortaria, antes de rodar.

### Upscale
- **Config de Upscale** / **Blend de Batches** — super-nodes que substituem os subgraphs de Settings/Blend Frames.
- **Pad to 4n+1** / **Trim 4n+1 back to N** — envolvem qualquer etapa Wan pra não perder frames.

### Utilidades
Crescer+Borrar Máscara, Máscara em Blocos, Desenhar Máscara na Imagem, Face Crop Expand, Nitidez Inteligente, Texto/Mostrar Texto, Seed, Carregar Imagens da Pasta, Info do Vídeo, **Loader Tudo-em-1 Wan**, **Qwen-VL Caption**, **Prompt Guide** (35 presets Bernini, incluindo as 22 tarefas do Bernini-Bench), **Cronômetro / Relatório de Tempo**, Tracking (Camera/Point/Object + Export + Visualizer).

---

## Troca de rosto (FaceFusion) — incluída no pacote

Os nodes de face swap já vêm **dentro** deste pacote (pasta `facefusion/`), mas dependem de `onnxruntime-gpu`/`opencv`. Se essas libs não estiverem instaladas, **só os dois nodes de rosto** deixam de aparecer — todo o resto do pacote carrega normalmente.

O `install.bat` / `install.sh` já instalam as dependências certas automaticamente. Se quiser instalar manualmente com o Python embedded:

```text
cd C:\Users\nyckm\Documents\c3\ComfyUI-Easy-Install
.\python_embeded\python.exe -m pip install opencv-python onnx requests tqdm huggingface_hub psutil
```

> **Nunca** rode `pip` solto (resolve pro Python errado) nem instale `xformers`/`flash-attn`.

Os `.onnx` (swapper, scrfd, arcface, xseg, bisenet) baixam sozinhos no 1º uso, para `ComfyUI/models/facefusion/`.

Encaixe típico:
```text
Load Image (rosto novo) ─→ source_face ─┐
Load Video ─────────────→ target_images ─┴→ FaceFusion Swap ─images→ Save Video
                                              └─face_mask→ Bernini Infinity (region_mask)
```

---

## Memória (VRAM/RAM)

**`limpar_vram`**

| Valor | O que faz | Quando usar |
|---|---|---|
| `off` | Sem limpeza entre passos. | Raramente. |
| `leve` *(padrão)* | `gc.collect()` + esvazia cache da VRAM. Barato e seguro. | **Uso geral.** |
| `agressivo` | Também descarrega os modelos entre os passos → menor pico de VRAM. | Só se estourar VRAM. |

> 🛡️ **Guard automático:** se o modelo tem muitos patches (LoRA = centenas), descarregar **entre passos** obriga a refazer o staging de GBs e re-aplicar todos os patches — custa muito mais do que economiza sob DynamicVRAM. O `agressivo` vira `leve` automaticamente nesse caso e avisa no console.

**`monitor_memoria`** — imprime RAM e VRAM em tempo real no console. Precisa de CUDA (VRAM) e `psutil` (RAM).

---

## `sequential` vs `context_window`

| | `sequential` | `context_window` |
|---|---|---|
| Como processa | Chunks em sequência, avançando `chunk_size − overlap` | Vídeo inteiro, com janela deslizante |
| VRAM | Mais econômico | Um pouco mais pesado |
| `mask_mode: bbox` | ❌ (cai pra `inpaint`) | ✅ |
| Continuidade | Boa, via `tail_memory` | Melhor (nativa) |
| Quando usar | Vídeo muito longo / VRAM curta | Vídeo cabe numa geração, ou quer `bbox` |

⚠️ `chunk_size` pequeno com `overlap` grande multiplica passagens. O `chunk_size` define a janela de atenção — quando o vídeo ultrapassa esse limite, o Wan roda **múltiplas janelas por step** e o `s/it` sobe linearmente. Se o `s/it` aumentar com vídeos mais longos, aumente o `chunk_size` para cobrir o vídeo inteiro (próximo 4n+1 acima do número de frames).

**`mask_mode`:** `off` (regenera tudo) · `inpaint` (edita só a área da máscara) · `bbox` (recorta a região e gera em resolução menor — só em `context_window`).

**`bbox_compose`:** `silhouette` usa a silhueta da máscara como alpha; `rectangle` cola o retângulo inteiro com feather, eliminando a "linha" de contorno.

---

## Changelog

- **0.2** — correção automática de frames 4n+1 (padding espelhado + corte de volta); `mask_mode`, `mask_grow`, `mask_blur`.
- **0.3** — `FaceStitchUpscale`.
- **0.5** — Load/Save Video com tipo `VIDEO` nativo.
- **0.6–0.9** — suíte de utilitários próprios, Comparar Vídeos A/B, Prever BBox, Config de Upscale / Blend de Batches.
- **0.10** — Editor de Pontos SAM3.
- **0.11** — `bbox_compose` (`silhouette`/`rectangle`); máscara acompanha o `resize_mode`.
- **0.12** — troca de rosto local (ONNX) incluída no pacote.
- **0.13** — **gerenciamento de memória**: limpeza de VRAM entre high/low e entre blocos, com **guard contra re-stage/re-patch** sob DynamicVRAM. Widgets `limpar_vram` e `monitor_memoria`.
- **0.14** — **reasoning do paper**: `Bernini Prompt Enhancer` (self-text CoT via Qwen local), `First-Frame CoT` (self-vision-text), `Bernini Multi-Guidance` (eq. 8–12, experimental). Prompt Guide expandido para as 22 tarefas do Bernini-Bench (35 presets).
- **0.15** — **MoCha** (`Mocha Embeds` + `Mocha Info`), com fix de frames que o node original não tem. **Save Video** blindado (normaliza tensores malformados; denuncia NaN e vídeo preto). **Instalador refeito**: modelos Bernini-R INT8 ConvRot + LoRAs LightX2V 4-step, detecção automática de CUDA para o `onnxruntime-gpu`, downloads idempotentes.
- **0.16** — **Tile Split / Select / Accumulate / Merge**: corte por contagem com tamanho automático, alinhamento a múltiplo de 16, costura com feather e detecção de upscale.
- **0.17** — **Wan Tiled Sampler** (step-fused guider): tile no latente com janela Hann complementar — funciona pra T2V puro; com `context_latents` (V2V) desliga automaticamente com aviso.
- **0.18** — `guidance_mode=tiled` integrado ao Bernini Infinity (mesma proteção automática). Guard de re-stage/re-patch no `limpar_vram=agressivo` corrigido.
- **0.19** — **Bernini Infinity Tiled**: tile em pixels com pipeline COMPLETO por ladrilho, costura viva e três modos (`off` / `inpaint` / `bbox`). O modo `bbox` faz duplo recorte (tile + bounding box da máscara dentro do tile) — até **9,6× mais rápido** que `inpaint` e **38×** que `off` quando o objeto é pequeno. **Load Video**: widget `reverse` (inverte os frames).

---

## Por que não há um "Bernini Long Sampler"?

Porque o Bernini já passa contexto pelo `conditioning`, e o patch Wan já aceita `context_latents` como lista. O ponto que precisa ser robusto é *gerar os conditionings certos*:

```python
context_latents = [encoded_chunk, tail_latent]
```

Assim o pacote aproveita a arquitetura nativa em vez de clonar um sampler inteiro — fica isolado e compatível com workflows já existentes.

---

## 🙏 Agradecimentos

Baseado em nodes do **Kijai** e nos modelos **Bernini** (ByteDance). O módulo de troca de rosto reconstrói o [FaceFusion ComfyUI](https://github.com/huygiatrng/Facefusion_comfyui) (huygiatrng) em modo local. Os nodes de MoCha se apoiam no [MoCha](https://github.com/Orange-3DV-Team/MoCha) (Orange-3DV-Team) e no WanVideoWrapper. O `Bernini Infinity Tiled` foi inspirado pelo [comfyUI-TiledWan](https://github.com/Baverne/comfyUI-TiledWan) (Baverne). O `Wan Tiled Sampler` foi inspirado pelo [comfyui-deno-custom-nodes](https://github.com/Deno2026/comfyui-deno-custom-nodes) (Deno2026). Obrigado aos autores e à comunidade.

## 📄 Licença

Apache License 2.0. A parte FaceFusion (pasta `facefusion/`) é MIT (engine ONNX vendorizado). **Respeite as licenças dos modelos**: vários swappers são non-commercial (InsightFace); os `ghost_*` são Apache-2.0.
