# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Prompt Enhancer (Qwen3), com os parametros oficiais do modelo
=============================================================================
Escreve/melhora prompts de video usando Qwen3, respeitando as recomendacoes
oficiais de amostragem do modelo -- que sao DIFERENTES das do Qwen2.5 e
diferentes entre os dois modos do Qwen3:

    modo PENSAR (enable_thinking=True):
        temperature 0.6 | top_p 0.95 | top_k 20 | min_p 0
        NUNCA use decodificacao gulosa (temperature=0) -- o Qwen3 nesse modo
        entra em loop e repete texto sem parar. E o erro mais comum.

    modo DIRETO (enable_thinking=False):
        temperature 0.7 | top_p 0.80 | top_k 20 | min_p 0

O widget 'modo_qwen3' ja aplica esses valores sozinho. So mexa nos numeros se
souber o que esta fazendo -- por isso eles ficam em 'avancado' e so valem
quando voce escolhe 'personalizado'.

Outras particularidades do Qwen3 tratadas aqui:
  * O chat template aceita `enable_thinking`. Se a versao do transformers for
    antiga e nao aceitar, caimos no template normal sem quebrar.
  * No modo pensar a saida vem com um bloco <think>...</think> ANTES da
    resposta. Ele e removido do 'prompt' e devolvido separado em 'raciocinio',
    pra voce ler sem sujar o prompt que vai pro modelo de video.
  * Qwen3 tambem existe em versao com visao (Qwen3-VL). Se voce escolher um
    modelo VL e ligar 'imagem', ele OLHA o frame antes de escrever.
"""

import logging
import re

try:
    import torch
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Prompt"

_MODELOS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]

# valores OFICIAIS recomendados pelo Qwen3 (nao inventar)
_PRESETS = {
    "pensar":  dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, enable_thinking=True),
    "direto":  dict(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, enable_thinking=False),
}

_CACHE = {"chave": None, "modelo": None, "tok": None, "proc": None}

_SISTEMA = (
    "You are a prompt engineer for text-to-video and video-editing diffusion models "
    "(MiniMax H3, Wan, Bernini). Rewrite the user's request into ONE vivid, concrete "
    "instruction the model can follow.\n"
    "Rules:\n"
    "- The request may come in ANY language (Portuguese, Spanish...). ALWAYS answer in "
    "ENGLISH: these models were trained mostly on English captions. Translate, then enrich.\n"
    "- Keep the user's intent EXACTLY. Never add edits they did not ask for.\n"
    "- Be concrete and visual: subject, materials, lighting, colors, camera move, motion. "
    "Avoid abstract or emotional wording the model cannot render.\n"
    "- Describe MOTION explicitly -- it is a video, not a still.\n"
    "- Require temporal coherence: identity, clothing and lighting stay consistent across frames.\n"
    "- If a frame is provided, ground the description in what you actually see.\n"
    "Output ONLY the final instruction as plain text: no preamble, no quotes, no markdown, "
    "no bullet points."
)


def _carregar(nome, dtype, device):
    chave = (nome, dtype, device)
    if _CACHE["chave"] == chave and _CACHE["modelo"] is not None:
        return _CACHE["modelo"], _CACHE["tok"], _CACHE["proc"]
    try:
        import transformers
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("[Qwen3] transformers nao instalado. Rode: pip install -U transformers accelerate") from e

    td = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(dtype, torch.bfloat16)
    proc = None
    ehVL = "-VL" in nome.upper()
    modelo = None
    erros = []
    nomes_cls = (["Qwen3VLForConditionalGeneration", "AutoModelForImageTextToText"] if ehVL
                 else ["Qwen3ForCausalLM", "AutoModelForCausalLM"])
    for cn in nomes_cls:
        try:
            Cls = getattr(transformers, cn, None)
            if Cls is None:
                continue
            modelo = Cls.from_pretrained(nome, torch_dtype=td, device_map=device)
            break
        except Exception as e:
            erros.append(f"{cn}: {e}")
    if modelo is None:
        raise RuntimeError(f"[Qwen3] nao consegui carregar '{nome}'. Tentativas: {erros}")

    tok = AutoTokenizer.from_pretrained(nome)
    if ehVL:
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(nome)
        except Exception as e:
            log.warning("[Qwen3] processor VL indisponivel (%s); vou usar so texto.", e)
    _CACHE.update({"chave": chave, "modelo": modelo, "tok": tok, "proc": proc})
    return modelo, tok, proc


def _separa_think(txt):
    """Qwen3 no modo pensar devolve <think>...</think> antes da resposta."""
    m = re.search(r"<think>(.*?)</think>", txt, flags=re.S)
    if m:
        return txt[m.end():].strip(), m.group(1).strip()
    # as vezes o </think> vem sem abertura (o template ja injeta o <think>)
    if "</think>" in txt:
        a, b = txt.split("</think>", 1)
        return b.strip(), a.replace("<think>", "").strip()
    return txt.strip(), ""


class BruxosQwen3PromptEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instrucao": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "Sua ideia, PODE SER EM PORTUGUES. Ex.: 'a mulher caminha pela ponte enquanto a camera se afasta'. "
                    "O node reescreve em ingles, detalhado e com movimento explicito."}),
                "modelo": (_MODELOS, {"default": _MODELOS[0], "tooltip":
                    "Qual Qwen3. Os '-VL' enxergam imagem (ligue 'imagem'); os outros sao so texto. "
                    "4B ja escreve bem e cabe folgado na 4090; 14B melhora um pouco e pesa bem mais."}),
                "modo_qwen3": (["direto", "pensar", "personalizado"], {"default": "direto", "tooltip":
                    "APLICA OS VALORES OFICIAIS do Qwen3 automaticamente:\n"
                    "direto  -> temp 0.7 | top_p 0.80 | top_k 20  (enable_thinking=False). Rapido, ideal pra prompt.\n"
                    "pensar  -> temp 0.6 | top_p 0.95 | top_k 20  (enable_thinking=True). Raciocina antes; mais lento, "
                    "melhor em pedidos complexos. A saida vem separada em 'raciocinio'.\n"
                    "personalizado -> usa os campos avancados abaixo.\n"
                    "AVISO: no modo pensar NUNCA use temperatura 0 -- o Qwen3 entra em loop."}),
            },
            "optional": {
                "imagem": ("IMAGE", {"tooltip":
                    "[so nos modelos -VL] Um frame pro modelo OLHAR antes de escrever. Deixa o prompt fiel ao que "
                    "existe na cena em vez de generico."}),
                "regras_extra": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "Instrucoes suas somadas ao system prompt. Ex.: 'sempre citar a referencia como from image0', "
                    "'estilo documentario, sem camera lenta'."}),
                "max_tokens": ("INT", {"default": 320, "min": 32, "max": 4096, "step": 16, "tooltip":
                    "Teto de tokens da resposta. 320 basta pra um prompt. No modo 'pensar' suba (800+), porque o "
                    "raciocinio consome tokens antes da resposta sair."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip":
                    "[so em 'personalizado'] Oficial: 0.7 no direto, 0.6 no pensar."}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                    "[so em 'personalizado'] Oficial: 0.80 no direto, 0.95 no pensar."}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1, "tooltip":
                    "[so em 'personalizado'] Oficial: 20 nos dois modos."}),
                "pensar_personalizado": ("BOOLEAN", {"default": False, "tooltip":
                    "[so em 'personalizado'] Liga o enable_thinking do chat template."}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "bf16 e o recomendado pro Qwen3."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "manter_carregado": ("BOOLEAN", {"default": True, "tooltip":
                    "Mantem o modelo na VRAM entre execucoes. DESLIGUE se precisar da VRAM pro sampler de video."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "raciocinio", "info")
    OUTPUT_TOOLTIPS = (
        "O prompt final em ingles -> ligue no CLIP Text Encode / no 'prompt' do node do H3.",
        "O bloco <think> do modo pensar (vazio no modo direto). So pra voce ler.",
        "Modelo, modo e parametros de amostragem realmente usados.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Prompt Enhancer Qwen3 (Bruxos): reescreve sua ideia (em portugues mesmo) num prompt de video detalhado em "
        "ingles, aplicando os PARAMETROS OFICIAIS de amostragem do Qwen3 -- que mudam entre o modo direto "
        "(temp 0.7/top_p 0.80) e o modo pensar (temp 0.6/top_p 0.95), e que NAO sao os mesmos do Qwen2.5. "
        "Separa o bloco <think> do texto final e suporta os modelos -VL (olham um frame antes de escrever)."
    )

    def run(self, instrucao, modelo, modo_qwen3="direto", imagem=None, regras_extra="",
            max_tokens=320, temperature=0.7, top_p=0.8, top_k=20, pensar_personalizado=False,
            dtype="bf16", device="auto", manter_carregado=True, seed=0):
        if not _OK:
            raise RuntimeError("[Qwen3] torch indisponivel.")
        instr = (instrucao or "").strip()
        if not instr:
            raise ValueError("[Qwen3] 'instrucao' esta vazia -- escreva o que voce quer que aconteca no video.")

        # ---- parametros: preset oficial ou personalizado ------------------
        if modo_qwen3 in _PRESETS:
            p = dict(_PRESETS[modo_qwen3])
            origem = f"preset oficial '{modo_qwen3}'"
        else:
            p = dict(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k),
                     min_p=0.0, enable_thinking=bool(pensar_personalizado))
            origem = "personalizado"
            if p["enable_thinking"] and p["temperature"] <= 0.0:
                p["temperature"] = 0.6
                print("[Qwen3] AVISO: temperatura 0 com 'pensar' faz o Qwen3 entrar em LOOP. "
                      "Corrigi pra 0.6 (valor oficial).", flush=True)

        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        mdl, tok, proc = _carregar(modelo, dtype, dev)
        ehVL = proc is not None and imagem is not None

        sistema = _SISTEMA + (("\nExtra rules:\n" + regras_extra.strip()) if (regras_extra or "").strip() else "")

        # ---- monta a conversa --------------------------------------------
        if ehVL:
            from PIL import Image
            fr = imagem[0]
            arr = (fr[..., :3].detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
            pil = Image.fromarray(arr)
            msgs = [{"role": "system", "content": sistema},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
        else:
            msgs = [{"role": "system", "content": sistema},
                    {"role": "user", "content": instr}]

        alvo = proc if ehVL else tok
        try:
            texto = alvo.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=p["enable_thinking"])
        except TypeError:
            # transformers antigo: sem suporte a enable_thinking
            texto = alvo.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            print("[Qwen3] aviso: seu 'transformers' nao aceita enable_thinking -- usando o template padrao. "
                  "Atualize (pip install -U transformers) pra controlar o modo pensar.", flush=True)

        if ehVL:
            inputs = proc(text=[texto], images=[pil], return_tensors="pt").to(dev)
        else:
            inputs = tok([texto], return_tensors="pt").to(dev)

        if seed:
            torch.manual_seed(int(seed))

        ger = dict(max_new_tokens=int(max_tokens), do_sample=p["temperature"] > 0,
                   temperature=p["temperature"], top_p=p["top_p"], top_k=int(p["top_k"]))
        try:
            import inspect
            if "min_p" in inspect.signature(mdl.generate).parameters or True:
                ger["min_p"] = p["min_p"]
        except Exception:
            pass

        with torch.inference_mode():
            try:
                out = mdl.generate(**inputs, **ger)
            except TypeError:
                ger.pop("min_p", None)
                out = mdl.generate(**inputs, **ger)

        corte = out[:, inputs["input_ids"].shape[1]:]
        dec = (proc if ehVL else tok).batch_decode(corte, skip_special_tokens=True)[0]
        prompt, pensamento = _separa_think(dec)

        if not manter_carregado:
            _CACHE.update({"chave": None, "modelo": None, "tok": None, "proc": None})
            try:
                del mdl, tok, proc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        info = (f"{modelo} | {origem} | temp={p['temperature']} top_p={p['top_p']} top_k={p['top_k']} "
                f"thinking={p['enable_thinking']}{' | +imagem(VL)' if ehVL else ''} | "
                f"{len(instr)}ch -> {len(prompt)}ch")
        print(f"[Qwen3 Enhancer] {info}", flush=True)
        print(f"[Qwen3 Enhancer] prompt: {prompt[:220]!r}{'...' if len(prompt) > 220 else ''}", flush=True)
        if not prompt:
            print("[Qwen3 Enhancer] ATENCAO: prompt vazio. No modo 'pensar' isso costuma ser max_tokens curto "
                  "demais -- o raciocinio consumiu tudo. Suba pra 800+.", flush=True)
        return (prompt, pensamento, info)


NODE_CLASS_MAPPINGS = {"BruxosQwen3PromptEnhancer": BruxosQwen3PromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosQwen3PromptEnhancer": "Prompt Enhancer Qwen3 (Bruxos)"}
