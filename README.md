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
- **Bernini Infinity** — renderer principal para vídeos maiores que o limite de 81 frames, sem precisar de um sampler novo. Injeta `context_latents` por chunk (com `tail_memory` opcional) em vez de um único conditioning — aproveita a arquitetura nativa do Bernini/Wan.
- **Bernini Region Mask** — normaliza máscara colorida (SAM2/SAM3/Scail2Color) em B/W, com invert/grow/blur.
- **Bernini Long** *(Conditioning / ChunkSelect / VideoMerge / AppendVideoChunk / EmptyVideoChunks / Info)* — helpers de vídeo longo.
- **FaceStitchUpscale** — cola o rosto upscalado de volta no vídeo usando os `face_bboxes` do Pose and Face Detection.
- **Editor de Pontos SAM3** — clique **verde = selecionar**, **roxo = negar** sobre o frame, pra fixar o alvo do tracking (mais estável que prompt de texto puro).

**Vídeo**
- **Load Video** / **Save Video** — equivalentes ao VHS com tipo `VIDEO` nativo (nodes 2.0), preview já cortado por skip/cap/nth/force_rate, export com mais controle de codec/CRF.
- **Comparar Vídeos A/B** — player embutido (cortina, lado a lado, diferença, alternar) pra conferir antes/depois sem sair do Comfy.
- **Prever BBox da Máscara** — desenha a caixa que o modo `bbox` recortaria, antes de rodar.

**Upscale**
- **Config de Upscale** / **Blend de Batches** — super-nodes que substituem os subgraphs de Settings/Blend Frames.

**Utilidades**
- Crescer+Borrar Máscara, Máscara em Blocos, Nitidez Inteligente, Texto/Mostrar Texto, Seed, Carregar Imagens da Pasta, Info do Vídeo, Loader Tudo-em-1 Wan, Qwen-VL Caption, Prompt Guide (presets oficiais Bernini).

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

---

## Changelog (principais marcos)

- **0.2** — correção automática de frames 4n+1 (padding espelhado + corte de volta); `mask_mode` (off/inpaint/bbox), `mask_grow`, `mask_blur`.
- **0.3** — `FaceStitchUpscale`.
- **0.5** — Load/Save Video com tipo `VIDEO` nativo (nodes 2.0).
- **0.6–0.9** — suíte de utilitários próprios (reduz dependência de terceiros), Comparar Vídeos A/B, Prever BBox da Máscara, Config de Upscale / Blend de Batches, preview de corte no Load Video direto no servidor.
- **0.10** — Editor de Pontos SAM3 (seleção verde/negação roxa) para tracking mais estável.

---

## Por que não há um "Bernini Long Sampler"?

Porque o Bernini já passa contexto pelo `conditioning`, e o patch Wan já aceita `context_latents` como lista. O ponto que precisa ser robusto é *gerar os conditionings certos*:

```python
context_latents = [encoded_chunk, tail_latent]
```

Assim o pacote aproveita a arquitetura nativa em vez de clonar um sampler inteiro — fica isolado e compatível com workflows já existentes. Um executor automático pode vir depois, mas dependeria dos nomes/classes exatos dos nodes Bernini instalados em cada máquina.

---

## 🙏 Agradecimentos

Baseado em nodes do **Kijai** e nos modelos **Bernini**. Obrigado aos autores e à comunidade.

## 📄 Licença

Apache License 2.0.
