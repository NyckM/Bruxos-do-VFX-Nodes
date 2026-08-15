# ComfyUI Bruxos do VFX

Nodes de produção para ComfyUI: MiniMax H3, Bernini/Wan, inpaint, tracking, vídeo, tiles, upscale e composição. A interface e os relatórios são em português.

## Instalação no Windows

Coloque esta pasta em `ComfyUI\custom_nodes\ComfyUI-Bruxos-do-VFX`, feche o ComfyUI e execute:

```text
install.bat
```

Modos rápidos:

```text
install.bat --deps-only
install.bat --skip-models
```

O instalador encontra o Python do ComfyUI, instala as dependências e escolhe o `onnxruntime-gpu` compatível com a CUDA do PyTorch. Ele não altera diretamente `torch`, `numpy`, `triton`, `xformers` ou `flash-attn`. Reinicie o ComfyUI e pressione `F5`.

## Nodes novos

### Imagem e vídeo

| Node | Função |
|---|---|
| **Load Image + Crop** | Preview em tempo real, crop visual, presets de proporção, resize/pad/stretch, rotação e flip horizontal/vertical. |
| **Load Video** | Carrega vídeo, áudio e metadados; limita, pula, inverte ou reamostra frames e pode criar cache. |
| **Save Video** | Exporta H.264/H.265/VP9/ProRes, áudio e sequência PNG, verificando frames inválidos. |

### Inpaint e composição

| Node | Função |
|---|---|
| **Flux2 Klein Inpaint + Outpaint** | Usa máscara branca para preencher com Klein Base + LoRA; aceita uma segunda imagem como referência visual. |
| **Máscara → BBox → Recorte** | Recorta somente a região da máscara para processar menos pixels. |
| **Stitch pelo BBox** | Recoloca o resultado no quadro original com máscara e feather. |
| **Composite & Refine** | Compõe uma referência sobre o vídeo e produz a máscara de refinamento. |
| **Chromakey** | Chroma/luma key, despill, matte externo e composição de fundo. |

### Tracking e rosto

| Node | Função |
|---|---|
| **Tracked Crop / SAM3** | Segue a máscara por frame e suaviza separadamente centro e tamanho do crop. |
| **Tracked Stitch** | Reencaixa o crop editado com feather, blend e correção de cor. |
| **H3 Face Denoise por Frame** | Usa mais denoise em rostos pequenos e preserva rostos grandes sem alterar a máscara de áudio. |
| **AutoEdit Router / Mask** | Transforma uma instrução em alvo, prompt de edição e máscara rastreada pelo SAM3. |

### Movimento

| Node | Função |
|---|---|
| **Motion Analyzer** | Mede movimento entre frames e cria um mapa temporal. |
| **Motion Timeline** | Edita intensidade e hold do movimento ao longo do vídeo. |
| **Motion Time Smear** | Distribui frames rápidos para reduzir saltos temporais. |
| **Motion Recover** | Recupera a duração original depois do processamento. |

### MiniMax H3

| Node | Função |
|---|---|
| **H3 Loader tudo-em-um** | Carrega transformer, text encoder e os VAEs de vídeo e áudio. |
| **H3 Frames** | Calcula comprimentos válidos `17k+5` e ajusta a referência. |
| **H3 Prompt Rápido** | Expande macros `@` e `#` para prompts estruturados. |
| **H3 Context-IR / Referência** | Organiza ideia, câmera, ação, áudio e referências no formato do H3. |
| **H3 Força do Condicionamento** | Controla quanto o H3 segue vídeo e referências. |
| **H3 Latent Upscale 2-pass** | Amplia o latent entre samplers e ajusta as referências. |
| **H3 Row Chunk Exato** | Divide QKV/MLP em linhas para reduzir o pico de memória sem aproximação. |
| **H3 Reference Isolate** | Impede referências de consultarem o alvo ruidoso durante parte do denoise. |
| **H3 Bloco / Memória Rolante** | Processa vídeos longos mantendo cauda e memória visual. |
| **H3 Latent SSD** | Salva, lê, separa e recompõe vídeo/áudio latente no SSD. |
| **H3 Clay Reference Auto** | Combina referência principal e clay em resolução e tempos adequados. |

### Tiles e SSD

| Node | Função |
|---|---|
| **Video Tiler** | Divide vídeo e referências por tile e recompõe em alta resolução. |
| **Tile Color Match** | Iguala cor e contraste entre tiles. |
| **Seam-Aware Video Tile Merge** | Busca o alinhamento de menor erro no overlap para reduzir emendas. |
| **Custom Tile Layout / Slice** | Desenha uma grade personalizada e produz os recortes. |
| **Video/Images → Disk Cache** | Grava frames no SSD para reduzir uso de RAM. |
| **Disk Cache → Window** | Lê apenas a janela necessária, com overlap e prefetch. |
| **Bernini Infinity SSD 81** | Processa Bernini em blocos de 81 frames usando cache e tail. |

### Bernini e Wan

| Node | Função |
|---|---|
| **Bernini Infinity** | Vídeo longo com chunks, overlap, máscara e memória de cauda. |
| **Bernini Infinity Tiled / Optimized** | Executa por ladrilho; a versão Optimized reduz encode/decode e trocas de modelo. |
| **Bernini TeaCache** | Reaproveita blocos semelhantes para acelerar a inferência. |
| **Block Swap RAM Offload** | Move blocos para RAM quando o modelo não cabe na VRAM. |
| **Bernini I2V / Ref-to-Video** | Gera vídeo a partir de imagem sem `source_video`. |
| **Wan Tiled Upscale** | Upscale Wan com tiles, janelas temporais e cache. |

## Dependências opcionais

- **ComfyUI-WanVideoWrapper:** MoCha e WanVideo Context.
- **ComfyUI-LTXVideo:** samplers LTX.
- **SAM3/easy-use:** workflows que usam segmentação SAM3.
- **FlashAttention:** opcional; deve corresponder à versão de CUDA/PyTorch.

Os modelos ONNX do FaceFusion são baixados no primeiro uso para `ComfyUI/models/facefusion`.

## Modelos baixados pelo instalador

- Bernini-R HIGH/LOW INT8 ConvRot → `models/diffusion_models`
- LoRAs Bernini LightX2V HIGH/LOW → `models/loras`
- UMT5 XXL FP8 → `models/text_encoders`
- Wan 2.1 Video VAE → `models/vae`

Flux2 Klein não é baixado automaticamente. Selecione manualmente Klein Base 4B, `flux2-vae`, Qwen 3 4B e a LoRA de outpaint.

## Workflows e licença

Exemplos ficam em [`workflows`](workflows). Licença Apache-2.0; adaptações externas mantêm a atribuição indicada no código.
