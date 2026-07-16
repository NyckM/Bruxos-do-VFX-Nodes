# ComfyUI Bruxos do VFX

Custom nodes para VFX de vídeo com Bernini-R / Wan 2.2: remoção de objetos e pessoas, upscale por batch, face swap, MoCha, comparação A/B e utilitários — tudo em português, feitos pra produção real.

<img width="832" height="480" alt="Workflow para remover objetos e pessoas" src="https://github.com/user-attachments/assets/6745a8ce-00ce-4915-a60d-ed1354099311" />

## Por que existe

Desenvolvido na **Bruxos do VFX** para dois longas-metragens — *Dr. Monstro* (Marcos Jorge) e *Alice Júnior 2* (Gil Baroni) — onde a demanda de composição e integração de cenas gerou a necessidade de uma ferramenta própria de remoção de objetos sobre Bernini/Wan.

**Resultado:** média de 96 frames em 204 segundos.

https://github.com/user-attachments/assets/1575f97f-34b9-492a-accb-818e97a6cbde

https://github.com/user-attachments/assets/d1e47486-ac3b-4030-be42-56a0f16b0128

---

## Instalação

### Fácil (recomendado)
Rode o instalador de dentro da pasta do node — ele instala as dependências **e baixa os modelos** nas pastas certas:

```text
ComfyUI\custom_nodes\ComfyUI-Bruxos-do-VFX\install.bat     (Windows)
bash ComfyUI/custom_nodes/ComfyUI-Bruxos-do-VFX/install.sh (Linux / RunPod)
```

É **idempotente**: modelos já baixados são pulados, então pode rodar de novo pra retomar um download interrompido.

> **O instalador nunca mexe em `torch`, `numpy`, `triton`, `xformers` ou `flash-attn`.** Ele detecta a CUDA do seu torch e instala o `onnxruntime-gpu` que **casa** com ela (cu12x → build CUDA 12; cu13x → build CUDA 13) — instalar a build errada faz o face swap cair na CPU.

### Manual
Copie a pasta para `ComfyUI/custom_nodes/ComfyUI-Bruxos-do-VFX` e reinicie.

### Modelos

O `install.bat` baixa tudo isto sozinho. Se preferir na mão:

| Modelo | Onde baixar | Pasta destino |
|---|---|---|
| **Bernini-R INT8 ConvRot** (high + low) | [Comfy-Org/Bernini-R](https://huggingface.co/Comfy-Org/Bernini-R/tree/main/diffusion_models) | `models/diffusion_models` |
| **LoRA LightX2V 4-step** (high + low) | [rzgar/Bernini-R-LightX2V-4step-loras](https://huggingface.co/rzgar/Bernini-R-LightX2V-4step-loras) | `models/loras` |
| Text encoder (umt5-xxl fp8) | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged) | `models/text_encoders` |
| **VAE de vídeo** (Wan 2.1 bf16) | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan2_1_VAE_bf16.safetensors) | `models/vae` |

> ⚠️ **Use o VAE de VÍDEO.** Um VAE `imageonly` / `upscale2x` devolve o tensor em outro layout e o vídeo sai **preto/quebrado**. O `Save Video` agora detecta isso e avisa, mas a correção é usar o VAE certo.

> Baixe por CLI/gerenciador, não pelo navegador — download incompleto gera arquivo corrompido.

**Por que INT8 ConvRot:** os pesos e ativações rodam em 8 bits nos tensor cores (`torch._int_mm`), não é desquantização on-the-fly como o GGUF. O ConvRot rotaciona os pesos antes de quantizar pra eliminar outliers, o que preserva a qualidade. Precisa de ComfyUI recente (INT8 é nativo) + Triton.

**Configuração com as LoRAs LightX2V:** `cfg = 1.0` (o CFG está destilado dentro da LoRA — valores altos **queimam**) e `steps = 6` com `split_step = 4` (4 passos no high + 2 no low).

---

## Nodes

### Bernini / Geração
- **Bernini Infinity** — o renderer principal, para vídeos maiores que o limite de 81 frames, sem precisar de um sampler novo. Injeta `context_latents` por chunk (com `tail_memory` opcional). Inclui correção automática de frames **4n+1**, **gerenciamento de VRAM** (ver abaixo), máscara de região e guidance multi-stream opcional.
- **Bernini Region Mask** — normaliza máscara colorida (SAM2/SAM3/Scail2Color) em B/W, com invert/grow/blur.
- **Bernini Prompt Enhancer** *(novo)* — reescreve sua instrução crua numa versão detalhada e estruturada via Qwen-VL **local**, opcionalmente olhando keyframes do vídeo-fonte. É o *self-text reasoning* do paper do Bernini, que sobe as métricas de edição. Saída drop-in pro seu encoder.
- **First-Frame CoT: Extrair / Compor** *(novo)* — o *self-vision-text* do paper: edite o **primeiro frame** como imagem e **propague** pro vídeo. O `Compor` devolve o vídeo-guia, a imagem de referência e a máscara da região alterada.
- **Bernini Multi-Guidance** *(novo)* — guidance **independente por stream** (texto / vídeo-fonte / referências), a decomposição da eq. 8–12 do paper. Saída `GUIDER` pro `SamplerCustomAdvanced`. ⚠️ Custa até 4 forwards por step e, em modelo cfg-destilado (LightX2V), é fora da distribuição — é experimental.
- **Bernini Long** *(Conditioning / ChunkSelect / VideoMerge / AppendVideoChunk / EmptyVideoChunks / Info)* — helpers de vídeo longo.
- **FaceStitchUpscale** — cola o rosto upscalado de volta no vídeo usando os `face_bboxes`.
- **Editor de Pontos SAM3** — clique **verde = selecionar**, **roxo = negar** sobre o frame, pra fixar o alvo do tracking.

### MoCha *(novo)*
- **Mocha Embeds (Bruxos)** — substitui o `MochaEmbeds` do WanVideoWrapper. Corrige o bug de frames do original (ele **trunca** e descarta frames: 111 → 109; aqui o padding é **espelhado** e você corta de volta sem perder nada), aceita **MASK ou IMAGE colorida** (SAM3/SCAIL/FaceFusion) com grow/blur, tem `tiled_vae` pra pouca VRAM, limpeza de memória e cronômetro.
  > O MoCha usa **uma única máscara** pro vídeo inteiro (não uma por frame). Se você ligar uma máscara por frame, o node reduz por **união** (cobre o sujeito onde quer que ele passe) ou pelo primeiro frame.
- **Mocha Info (Bruxos)** — calcula **antes de rodar** (sem gastar VRAM) os frames alinhados, o tamanho do latente, o `seq_len` (o custo real) e o plano de blocos.

### Tiles *(novo)*
Substituem o subgraph inteiro de "Tile Settings" (Rounding up num, Set dimension properly, Padding, imageSplitTiles, Total tiles, ImageComposite+, Split Images...) por 3 nodes.

- **Tile Split (Bruxos)** — corta a imagem/vídeo em ladrilhos pela **contagem** (`tile_count_width` × `tile_count_height`). O **tamanho de cada tile é calculado sozinho** e alinhado ao múltiplo de 16 do Wan. `1×1` = imagem inteira, sem corte.
- **Tile Select (Bruxos)** — pega o ladrilho N (ligue o `index` do For Loop) **com todos os frames** do vídeo — que é o que o Wan precisa.
- **Tile Merge (Bruxos)** — costura de volta com **feather na sobreposição** (sem linha de emenda) e **detecta upscale sozinho**: se os tiles voltarem 2× maiores, a imagem final sai 2× maior.

Exemplos reais com uma fonte 1920×1080 (`tile_padding = 80`):

| Grade | Tiles | Tamanho calculado |
|---|---|---|
| `1×1` | 1 | 1920×1088 — imagem inteira |
| `2×2` | 4 | 1120×704 |
| `8×8` | 64 | 400×304 |
| `16×16` | 256 | 288×240 |

> **1080 não é múltiplo de 16.** O Split preenche o canvas até 1088 (replicando a borda) e o Merge **recorta de volta** para 1080 — sem isso, 8 linhas seriam perdidas. O round-trip (split → merge) reconstrói a fonte exatamente.

Ligação típica:
```text
Load Video ─► Tile Split ─┬─ plan ──────────────────────────────────┐
                          ├─ total_tiles ─► For Loop Start (total)   │
                          └─ tiles ─► Tile Select ─► [Wan] ─► acumula┴─► Tile Merge ─► Save
                                          ▲ index (do For Loop)
```

### Face / Troca de rosto *(precisa das libs ONNX)*
- **FaceFusion Swap (Bruxos)** — troca de rosto **100% local** (ONNX, sem API). Imagem única ou vídeo inteiro, 13 swappers (`hyperswap_1c_256` recomendado), `pixel_boost` até 1024, seleção `one`/`many`/`reference` e máscaras combináveis. Sai com uma **MASK dos rostos** que liga direto no `region_mask` do Bernini Infinity.
- **FaceFusion Detectar Rostos (Bruxos)** — preview com caixas e landmarks, MASK por frame e contagem.

### Vídeo
- **Load Video** / **Save Video** — equivalentes ao VHS com tipo `VIDEO` nativo, preview já cortado por skip/cap/nth/force_rate, mais controle de codec/CRF. O **Save Video** agora **tolera e diagnostica** tensores malformados (5D, channels-first, canais extras) e **denuncia NaN/vídeo preto** em vez de gravar um arquivo preto em silêncio.
- **Comparar Vídeos A/B** — player embutido (cortina, lado a lado, diferença, alternar).
- **Prever BBox da Máscara** — desenha a caixa que o modo `bbox` recortaria, antes de rodar.

### Upscale
- **Config de Upscale** / **Blend de Batches** — super-nodes que substituem os subgraphs de Settings/Blend Frames.
- **Pad to 4n+1** / **Trim 4n+1 back to N** — envolvem qualquer etapa Wan pra não perder frames.

### Utilidades
Crescer+Borrar Máscara, Máscara em Blocos, Desenhar Máscara na Imagem, Face Crop Expand, Nitidez Inteligente, Texto/Mostrar Texto, Seed, Carregar Imagens da Pasta, Info do Vídeo, **Loader Tudo-em-1 Wan**, **Qwen-VL Caption**, **Prompt Guide** (35 presets Bernini, incluindo as 22 tarefas do Bernini-Bench), **Cronômetro / Relatório de Tempo**, Tracking (Camera/Point/Object + Export + Visualizer).

---

## Memória (VRAM/RAM)

Para resoluções maiores e vídeos longos, o **Bernini Infinity** limpa a memória **entre o high pass e o low pass** e **entre os blocos de frames**, prevenindo o acúmulo que enche a VRAM ao longo de renders grandes.

**`limpar_vram`**

| Valor | O que faz | Quando usar |
|---|---|---|
| `off` | Sem limpeza entre passos (legado). | Raramente. |
| `leve` *(padrão)* | `gc.collect()` + esvazia o cache da VRAM. Barato e seguro. | **Uso geral.** |
| `agressivo` | Também **descarrega os modelos** entre os passos → menor pico de VRAM. | Só se estourar VRAM em resolução alta. |

> 🛡️ **Guard automático:** se o modelo tem muitos patches (LoRA = centenas), descarregar **entre passos** obriga a refazer o staging de GBs **e re-aplicar todos os patches** na passada seguinte — custa muito mais do que economiza (especialmente sob DynamicVRAM / async offload). Nesse caso o `agressivo` vira `leve` **automaticamente** entre passos, e avisa no console. O unload do **fim** da run continua valendo.

**`monitor_memoria`** — imprime RAM e VRAM em tempo real no console (início, entre high/low, por bloco, fim). Precisa de CUDA (VRAM) e `psutil` (RAM).

```text
[Bernini Infinity][mem] pos-high: VRAM 19.00/24.00GB (alloc 12.00 reserv 15.00) | RAM 58.0/98.0GB
```

**`guidance_mode`** — `off` (CFG único, **recomendado**) · `multi` (guidance independente por stream). ⚠️ Em `cfg = 1.0` o ComfyUI já pula o passe negativo (**1 forward/step**); o `multi` faz **4** → ~4× mais lento. E num modelo cfg-destilado (LightX2V) ele é fora da distribuição. É experimental; o node avisa alto ao ligar.

---

## `sequential` vs `context_window`

| | `sequential` | `context_window` |
|---|---|---|
| Como processa | Chunks em sequência, avançando `chunk_size − overlap` | Vídeo inteiro, com janela deslizante |
| VRAM | Mais econômico | Um pouco mais pesado |
| `mask_mode: bbox` | ❌ (cai pra `inpaint`) | ✅ |
| Continuidade | Boa, via `tail_memory` | Melhor (nativa) |
| Quando usar | Vídeo muito longo / VRAM curta | Vídeo cabe numa geração, ou quer `bbox` |

⚠️ `chunk_size` pequeno com `overlap` grande multiplica passagens (chunk=17, overlap=16 → 61 passagens). Prefira chunk grande + overlap pequeno.

**`mask_mode`:** `off` (regenera tudo) · `inpaint` (edita só a área da máscara) · `bbox` (recorta a região e gera em resolução menor — **é o que otimiza de verdade**; só em `context_window`).

**`bbox_compose`:** `silhouette` usa a silhueta da máscara como alpha; `rectangle` cola o retângulo inteiro com feather (`mask_blur`), eliminando a "linha" de contorno.

---

## Changelog

- **0.2** — correção automática de frames 4n+1 (padding espelhado + corte de volta); `mask_mode`, `mask_grow`, `mask_blur`.
- **0.3** — `FaceStitchUpscale`.
- **0.5** — Load/Save Video com tipo `VIDEO` nativo.
- **0.6–0.9** — suíte de utilitários próprios, Comparar Vídeos A/B, Prever BBox, Config de Upscale / Blend de Batches.
- **0.10** — Editor de Pontos SAM3.
- **0.11** — `bbox_compose` (`silhouette`/`rectangle`); máscara acompanha o `resize_mode` da fonte.
- **0.12** — troca de rosto local (ONNX) incluída no pacote.
- **0.13** — **gerenciamento de memória** no Bernini Infinity: limpeza de VRAM entre high/low e entre blocos, com **guard contra o re-stage/re-patch** sob DynamicVRAM. Widgets `limpar_vram` e `monitor_memoria`.
- **0.14** — **reasoning do paper do Bernini**: `Bernini Prompt Enhancer` (self-text CoT via Qwen local), `First-Frame CoT` (self-vision-text), `Bernini Multi-Guidance` (eq. 8–12, experimental) e o `guidance_mode` no Infinity. Prompt Guide expandido para as **22 tarefas do Bernini-Bench** (35 presets no total).
- **0.15** — **MoCha** (`Mocha Embeds` + `Mocha Info`), com o fix de frames que o node original não tem. **Save Video** blindado (normaliza tensores malformados; denuncia NaN e vídeo preto em vez de gravar em silêncio). **Instalador refeito**: modelos **Bernini-R INT8 ConvRot** + **LoRAs LightX2V 4-step**, detecção automática da CUDA para o `onnxruntime-gpu`, e downloads idempotentes.
- **0.16** — **Tiles**: `Tile Split` / `Tile Select` / `Tile Merge` — corte por contagem (2×2, 8×8...) com tamanho automático alinhado a 16, costura com feather (sem emenda) e detecção de upscale. Substituem o subgraph "Tile Settings" inteiro.

---

## Por que não há um "Bernini Long Sampler"?

Porque o Bernini já passa contexto pelo `conditioning`, e o patch Wan já aceita `context_latents` como lista. O ponto que precisa ser robusto é *gerar os conditionings certos*:

```python
context_latents = [encoded_chunk, tail_latent]
```

Assim o pacote aproveita a arquitetura nativa em vez de clonar um sampler inteiro — fica isolado e compatível com workflows já existentes.

---

## 🙏 Agradecimentos

Baseado em nodes do **Kijai** e nos modelos **Bernini**. O módulo de troca de rosto reconstrói o [FaceFusion ComfyUI](https://github.com/huygiatrng/Facefusion_comfyui) (huygiatrng) em modo local. Os nodes de MoCha se apoiam no [MoCha](https://github.com/Orange-3DV-Team/MoCha) (Orange-3DV-Team) e no WanVideoWrapper. Obrigado aos autores e à comunidade.

## 📄 Licença

Apache License 2.0. A parte FaceFusion (pasta `facefusion/`) é MIT (engine ONNX vendorizado). **Respeite as licenças dos modelos**: vários swappers são non-commercial (InsightFace); os `ghost_*` são Apache-2.0.
