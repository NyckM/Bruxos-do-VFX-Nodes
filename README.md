# ComfyUI Bruxos do VFX

Custom nodes para VFX de vídeo com Wan 2.2 / Bernini: remoção de objetos e pessoas, upscale por batch, face swap, comparação A/B e utilitários — tudo em português, feitos pra produção real.

<img width="832" height="480" alt="Workflow para remover objetos e pessoas" src="https://github.com/user-attachments/assets/6745a8ce-00ce-4915-a60d-ed1354099311" />

## Por que existe

Desenvolvido na **Bruxos do VFX** para dois longas-metragens — *Dr. Monstro* (Marcos Jorge) e *Alice Júnior 2* (Gil Baroni) — onde a demanda de composição e integração de cenas gerou a necessidade de uma ferramenta própria de remoção de objetos sobre Bernini/Wan.

**Resultado:** média de 96 frames em 204 segundos.

https://github.com/user-attachments/assets/1575f97f-34b9-492a-accb-818e97a6cbde

https://github.com/user-attachments/assets/d1e47486-ac3b-4030-be42-56a0f16b0128

---

## Instalação

### Fácil (recomendado)
Rode o `.bat` de dentro da pasta de instalação do ComfyUI — ele baixa os modelos e instala o node pack automaticamente.
*(Versão RunPod ainda em desenvolvimento.)*

### Manual
Copie a pasta do pacote para:
```text
ComfyUI/custom_nodes/ComfyUI-Bruxos-do-VFX
```
Reinicie o ComfyUI.

### Modelos necessários

| Modelo | Onde baixar | Pasta destino |
|---|---|---|
| UNET GGUF (high/low) | [neuregex/Bernini-R-GGUF](https://huggingface.co/neuregex/Bernini-R-GGUF/tree/main) | `ComfyUI/models/unet` |
| LoRA distill (4 steps) | [Cyph3r/lightx2v_T2V_14B...](https://huggingface.co/Cyph3r/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16) | `ComfyUI/models/loras` |
| Text encoder (umt5) | [Osrivers/umt5_xxl_fp8...](https://huggingface.co/Osrivers/umt5_xxl_fp8_e4m3fn_scaled.safetensors) | `ComfyUI/models/text_encoders` |
| VAE | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1_VAE_bf16.safetensors) | `ComfyUI/models/vae` |

> Baixe por CLI/gerenciador, não pelo navegador — download incompleto gera arquivo corrompido.

---

## Nodes

<img width="226" height="557" alt="Bernini Long Condition" src="https://github.com/user-attachments/assets/934fd9b3-087f-47db-aeb2-6ca01967c556" />

**Bernini / Geração**
- **Bernini Infinity** — renderer principal para vídeos maiores que o limite de 81 frames, sem precisar de um sampler novo. Injeta `context_latents` por chunk (com `tail_memory` opcional) em vez de um único conditioning — aproveita a arquitetura nativa do Bernini/Wan. Tem **gerenciamento de memória** embutido (limpeza de VRAM entre os passos high/low e entre os blocos de frames, + monitor de RAM/VRAM) para rodar resoluções maiores e vídeos longos sem entupir a GPU — ver [Memória (VRAM/RAM)](#memória-vramram).
- **Bernini Region Mask** — normaliza máscara colorida (SAM2/SAM3/Scail2Color) em B/W, com invert/grow/blur.
- **Bernini Long** *(Conditioning / ChunkSelect / VideoMerge / AppendVideoChunk / EmptyVideoChunks / Info)* — helpers de vídeo longo.
- **FaceStitchUpscale** — cola o rosto upscalado de volta no vídeo usando os `face_bboxes` do Pose and Face Detection.
- **Editor de Pontos SAM3** — clique **verde = selecionar**, **roxo = negar** sobre o frame, pra fixar o alvo do tracking (mais estável que prompt de texto puro).

**Face / Troca de rosto** *(precisa das libs ONNX — ver instalação abaixo)*
- **FaceFusion Swap (Bruxos)** — troca de rosto **100% local** (ONNX, sem API). Aceita imagem única ou vídeo inteiro (batch), 13 modelos de swapper (`hyperswap_1c_256` recomendado, `inswapper_128_fp16` mais rápido), `pixel_boost` até 1024, seleção de rosto `one`/`many`/`reference` e máscaras combináveis (box + oclusão xseg + região bisenet + área). Sai já com uma **MASK dos rostos** que liga direto no `region_mask` do Bernini Infinity.
- **FaceFusion Detectar Rostos (Bruxos)** — preview com caixas verdes e landmarks roxos, MASK por frame e contagem. Útil pra calibrar `score_threshold`/`face_position` antes do swap, ou gerar máscara de rosto pro Bernini sem trocar nada.

**Vídeo**
- **Load Video** / **Save Video** — equivalentes ao VHS com tipo `VIDEO` nativo (nodes 2.0), preview já cortado por skip/cap/nth/force_rate, export com mais controle de codec/CRF.
- **Comparar Vídeos A/B** — player embutido (cortina, lado a lado, diferença, alternar) pra conferir antes/depois sem sair do Comfy.
- **Prever BBox da Máscara** — desenha a caixa que o modo `bbox` recortaria, antes de rodar.

**Upscale**
- **Config de Upscale** / **Blend de Batches** — super-nodes que substituem os subgraphs de Settings/Blend Frames.

**Utilidades**
- Crescer+Borrar Máscara, Máscara em Blocos, **Desenhar Máscara na Imagem** (visualização — pinta a máscara existente sobre a imagem, não seleciona/clica), **Face Crop Expand**, Nitidez Inteligente, Texto/Mostrar Texto, Seed, Carregar Imagens da Pasta, Info do Vídeo, Loader Tudo-em-1 Wan, Qwen-VL Caption, Prompt Guide (presets oficiais Bernini), **Cronômetro / Relatório de Tempo** (cronometra trechos do fluxo no backend, funciona em qualquer render mode).

---

## Troca de rosto (FaceFusion) — incluída no pacote

Os nodes de face swap já vêm **dentro** deste pacote (pasta `facefusion/`), mas dependem de
`onnxruntime-gpu`/`opencv`. Se essas libs não estiverem instaladas, **só os dois nodes de rosto**
deixam de aparecer — todo o resto do pacote carrega normalmente (o console mostra um aviso do
`_merge`).

Instale as dependências com o Python **embedded** do ComfyUI:
```text
cd C:\Users\nyckm\Documents\c3\ComfyUI-Easy-Install
.\python_embeded\python.exe -m pip install onnxruntime-gpu opencv-python onnx requests tqdm
```
> **Nunca** rode `pip` solto (resolve pro Python errado) nem instale `xformers`/`flash-attn`.
> O `install.bat` / `install.sh` deste pacote já fazem isso pra você.

Os `.onnx` (swapper, scrfd, arcface, xseg, bisenet) baixam sozinhos no 1º uso, do
[facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/), para
`ComfyUI/models/facefusion/`.

Categoria no menu: **Bruxos do VFX/Face**. Encaixe típico:
```text
Load Image (rosto novo) ─→ source_face ─┐
Load Video ─────────────→ target_images ─┴→ FaceFusion Swap ─images→ Save Video
                                              └─face_mask→ Bernini Infinity (region_mask) / Comparar A/B
```

---

## `sequential` vs `context_window`

| | `sequential` | `context_window` |
|---|---|---|
| Como processa | Chunks em sequência, avançando `chunk_size − overlap` frames | Vídeo inteiro, com janela deslizante de contexto |
| VRAM | Mais econômico | Um pouco mais pesado |
| `mask_mode: bbox` | ❌ (cai pra `inpaint`) | ✅ |
| Continuidade | Boa, via `tail_memory` | Melhor (nativa) |
| Quando usar | Vídeo muito longo / VRAM curta | Vídeo cabe numa geração, ou quer usar `bbox` |

⚠️ `chunk_size` pequeno com `overlap` grande multiplica passagens (ex.: chunk=17, overlap=16 → 61 passagens). Prefira chunk grande + overlap pequeno.

**`mask_mode`:** `off` (regenera tudo) · `inpaint` (edita só a área da máscara, resto = fonte) · `bbox` (recorta a região, gera em resolução menor — só em `context_window`).

**`bbox_compose`** *(no modo `bbox`)*: `silhouette` compõe usando a própria silhueta da máscara como alpha; `rectangle` cola o **retângulo inteiro** do bbox com feather nas bordas (usa `mask_blur` como feather) — elimina a "linha" de contorno que aparecia na composição por silhueta.

---

## Memória (VRAM/RAM)

Para resoluções maiores e vídeos longos, o **Bernini Infinity** limpa a memória em dois pontos críticos: **entre o high pass e o low pass** (os dois modelos não precisam ficar residentes na VRAM ao mesmo tempo) e **entre cada bloco de frames** (no modo `sequential` e a cada janela). Isso previne o acúmulo (memory leak) que enche a RAM/VRAM ao longo de renders grandes. Dois widgets controlam o comportamento:

**`limpar_vram`**

| Valor | O que faz | Quando usar |
|---|---|---|
| `off` | Sem limpeza entre os passos (comportamento legado). A limpeza essencial por bloco continua acontecendo. | Só se quiser o comportamento antigo. |
| `leve` *(padrão)* | `gc.collect()` + esvazia o cache da VRAM (`soft_empty_cache` + `empty_cache` + `ipc_collect`) entre high/low e entre blocos. Barato e seguro. | Uso geral. Deixe aqui. |
| `agressivo` | Além do acima, **descarrega os modelos** da VRAM entre os passos (high e low nunca ficam juntos) → menor pico de VRAM. Recarrega o modelo a cada troca (mais lento). | Só quando estourar VRAM em resolução alta. |

**`monitor_memoria`** (liga/desliga) — imprime no console o uso de **RAM e VRAM** em tempo real (no início, entre high/low, por bloco e no fim), pra você ver exatamente onde a memória enche. Precisa de CUDA (para a VRAM) e do `psutil` (para a RAM); o que faltar aparece em branco, sem quebrar. Exemplo de linha:

```text
[Bernini Infinity][mem] pos-high: VRAM 19.00/24.00GB (alloc 12.00 reserv 15.00) | RAM 58.0/98.0GB
```

> Portátil por padrão: tudo é tolerante a ambiente (sem CUDA, sem `psutil`, versões antigas do ComfyUI) e **não depende de nenhuma flag de launch**. Os dois widgets são opcionais — workflows salvos antes desta versão continuam válidos (carregam com o padrão `leve`). Como os widgets novos mudam os tipos do node, se o Bernini Infinity já estiver no grafo, **remova e readicione** (ou apenas religue os fios) para eles aparecerem.

---

## Changelog (principais marcos)

- **0.2** — correção automática de frames 4n+1 (padding espelhado + corte de volta); `mask_mode` (off/inpaint/bbox), `mask_grow`, `mask_blur`.
- **0.3** — `FaceStitchUpscale`.
- **0.5** — Load/Save Video com tipo `VIDEO` nativo (nodes 2.0).
- **0.6–0.9** — suíte de utilitários próprios (reduz dependência de terceiros), Comparar Vídeos A/B, Prever BBox da Máscara, Config de Upscale / Blend de Batches, preview de corte no Load Video direto no servidor.
- **0.10** — Editor de Pontos SAM3 (seleção verde/negação roxa) para tracking mais estável.
- **0.11** — `bbox_compose` (`silhouette`/`rectangle`): o modo `rectangle` cola o retângulo do bbox com feather (`mask_blur`) e elimina a "linha" de contorno na composição `bbox`; a máscara passa a acompanhar o `resize_mode` (`stretch`/`crop`) da fonte.
- **0.12** — troca de rosto local (ONNX) **incluída no pacote** (pasta `facefusion/`): nodes **FaceFusion Swap** e **Detectar Rostos**, com saída de máscara integrada ao Bernini. Reconstrução de [huygiatrng/Facefusion_comfyui](https://github.com/huygiatrng/Facefusion_comfyui) em modo 100% local.
- **0.13** — gerenciamento de memória no **Bernini Infinity**: limpeza de VRAM **entre o high pass e o low pass** e **entre os blocos de frames** (para resoluções maiores / vídeos longos), com prevenção de acúmulo (leak) via garbage collection. Widget **`limpar_vram`** (`off` / `leve` / `agressivo`) e widget **`monitor_memoria`** (relatório de RAM/VRAM em tempo real no console). Portátil (tolerante a ausência de CUDA/`psutil`) e compatível com workflows antigos. Ver [Memória (VRAM/RAM)](#memória-vramram).

---

## Por que não há um "Bernini Long Sampler"?

Porque o Bernini já passa contexto pelo `conditioning`, e o patch Wan já aceita `context_latents` como lista. O ponto que precisa ser robusto é *gerar os conditionings certos*:

```python
context_latents = [encoded_chunk, tail_latent]
```

Assim o pacote aproveita a arquitetura nativa em vez de clonar um sampler inteiro — fica isolado e compatível com workflows já existentes. Um executor automático pode vir depois, mas dependeria dos nomes/classes exatos dos nodes Bernini instalados em cada máquina.

---

## 🙏 Agradecimentos

Baseado em nodes do **Kijai** e nos modelos **Bernini**. O módulo de troca de rosto reconstrói o [FaceFusion ComfyUI](https://github.com/huygiatrng/Facefusion_comfyui) (huygiatrng) em modo local. Obrigado aos autores e à comunidade.

## 📄 Licença

Apache License 2.0. A parte FaceFusion (pasta `facefusion/`) é MIT (engine ONNX vendorizado — ver `facefusion/LICENSE.upstream`). **Respeite as licenças dos modelos** de swap: vários são non-commercial (InsightFace); os `ghost_*` são Apache-2.0.
