# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Edicao automatica por linguagem natural (roteador + mascara)
===========================================================================
A ideia: voce escreve UMA frase ("troque o objeto da mao do personagem por um
rato") e o grafo se vira pra: achar o objeto, rastrear ele no video, gerar a
mascara e entregar tudo pronto pro Bernini editar so aquela regiao.

Isso exige DOIS textos DIFERENTES, e e por isso que existe o roteador:

    SAM3   quer o que JA ESTA no video   -> "the blue mug held in the hand"
    Bernini quer o que DEVE VIRAR        -> "Replace the mug with a rat..."

Uma frase so nao serve pros dois. O `BruxosAutoEditRouter` manda a instrucao +
um keyframe pro Qwen-VL (o mesmo do Prompt Enhancer) e pede um JSON com os dois
campos. Como o Qwen VE o frame, ele troca "o objeto da mao" pelo nome CONCRETO
da coisa -- que e o que o SAM3 entende bem (grounding por frase relativa/
espacial e o ponto fraco dele).

O `BruxosAutoEditMask` chama o SAM3 Video Segmentation por dentro e devolve a
MASK [T,H,W] pronta pro `region_mask` do Bernini Infinity, mais um PREVIEW
visual pra voce conferir a mascara ANTES de gastar 10 min de render.

Fluxo tipico:

    Load Video ─┬─> AutoEdit Router ─┬─ sam_prompt ─> AutoEdit Mask ─ mask ─┐
                │                    └─ edit_prompt ─> CLIP Encode ─> pos   │
                └────────────────────────────────────> AutoEdit Mask       │
                                                             └─ preview    │
    Bernini Infinity(region_mask=mask, mask_mode=bbox) <────────────────────┘

Nada aqui e obrigatorio: os dois nodes tem override manual. Se o LLM errar o
alvo, voce digita o `sam_prompt` na mao; se o SAM3 errar mesmo assim, voce
aponta pontos positivos/negativos no frame.
"""

import json
import logging
import re

try:
    import torch
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Auto Edit"


# ---------------------------------------------------------------------------
# helpers do nodes.py (Qwen-VL): reusamos o MESMO loader/cache do Prompt
# Enhancer pra nao carregar um segundo Qwen na VRAM.
# ---------------------------------------------------------------------------
try:
    from .nodes import (
        _bx_qwen_load as _qwen_load,
        _bx_tensor_to_pil as _to_pil,
        _BX_QWEN_MODELS as _QWEN_MODELS,
    )
    _HAS_QWEN = True
except Exception:  # pragma: no cover
    try:
        from nodes import (
            _bx_qwen_load as _qwen_load,
            _bx_tensor_to_pil as _to_pil,
            _BX_QWEN_MODELS as _QWEN_MODELS,
        )
        _HAS_QWEN = True
    except Exception as e:
        logging.warning(f"[Bruxos AutoEdit] Qwen-VL indisponivel: {e}")
        _qwen_load = _to_pil = None
        _QWEN_MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct"]
        _HAS_QWEN = False


# ---------------------------------------------------------------------------
# ROTEADOR
# ---------------------------------------------------------------------------
_ROUTER_SYSTEM = (
    "You are the planner for an automatic video-editing pipeline. You receive ONE "
    "frame of a video and ONE edit instruction (which may be in ANY language: "
    "Portuguese, Spanish, etc).\n"
    "Your job is to split that instruction into TWO different texts, because two "
    "different models consume them:\n"
    "\n"
    "1. \"segment\": what must be SEGMENTED -- the thing that ALREADY EXISTS in the "
    "frame and is going to be replaced/removed/changed. This goes to SAM3, an "
    "open-vocabulary segmenter. CRITICAL: SAM3 is weak at relative or spatial "
    "phrases ('the object in his hand', 'the thing on the left'). LOOK at the frame "
    "and name the object CONCRETELY by what it actually is, with a short visual "
    "attribute if it helps: 'blue mug', 'red plastic bottle', 'silver phone'. Use a "
    "short noun phrase, 1-4 words, in ENGLISH, no articles like 'the' if avoidable.\n"
    "\n"
    "2. \"edit\": the full editing instruction for the Bernini/Wan video model, in "
    "ENGLISH, detailed and concrete. State the operation and the target clearly, say "
    "what must stay UNCHANGED (identity, hand, background, lighting) and require "
    "temporal coherence across frames. If the user refers to a reference image, use "
    "the literal trained marker 'from image0' (never 'reference_image_0' or any "
    "paraphrase). Never name the source video itself.\n"
    "\n"
    "Reply with ONLY a JSON object, no markdown fence, no commentary:\n"
    "{\"segment\": \"...\", \"edit\": \"...\"}"
)


def _parse_router_json(raw):
    """Extrai {'segment','edit'} da resposta do LLM. Tolerante a cerca de
    markdown, texto antes/depois e aspas curvas. Retorna (seg, edit, erro)."""
    txt = (raw or "").strip()
    # tira cerca ```json ... ```
    txt = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", txt).strip()
    # pega o primeiro bloco {...} balanceado de forma simples
    i, j = txt.find("{"), txt.rfind("}")
    if i >= 0 and j > i:
        txt = txt[i:j + 1]
    try:
        d = json.loads(txt)
        seg = str(d.get("segment", "") or "").strip()
        edt = str(d.get("edit", "") or "").strip()
        if seg or edt:
            return seg, edt, ""
        return "", "", "JSON sem os campos 'segment'/'edit'"
    except Exception as e:
        return "", "", f"resposta nao era JSON valido ({e})"


class BruxosAutoEditRouter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frames do video-fonte. So um keyframe e enviado ao LLM (o do 'frame_index')."}),
                "instrucao": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Sua instrucao em linguagem natural, PODE SER EM PORTUGUES. "
                               "Ex.: 'troque o objeto da mao do personagem por um rato'. "
                               "O node quebra isso em: o que SEGMENTAR (pro SAM3) e o que GERAR (pro Bernini)."}),
                "model_name": (_QWEN_MODELS, {"default": _QWEN_MODELS[0],
                    "tooltip": "Qwen-VL usado pra planejar. O 3B da conta; o 7B acerta mais o nome do objeto. "
                               "Usa o MESMO cache do Prompt Enhancer (nao carrega um segundo modelo)."}),
            },
            "optional": {
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Qual frame o LLM olha pra nomear o objeto. Escolha um em que o objeto apareca BEM visivel."}),
                "sam_prompt_override": ("STRING", {"multiline": False, "default": "",
                    "tooltip": "MANDA MAIS QUE O LLM. Se preenchido, este texto vai pro SAM3 e a sugestao do LLM e ignorada. "
                               "Use quando o SAM3 estiver pegando o objeto errado. Ex.: 'blue mug'. Em INGLES e curto."}),
                "edit_prompt_override": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "MANDA MAIS QUE O LLM. Se preenchido, vira o prompt de edicao do Bernini e a sugestao do LLM e ignorada."}),
                "max_new_tokens": ("INT", {"default": 300, "min": 32, "max": 2048, "step": 16,
                    "tooltip": "Teto de tokens da resposta do LLM. 300 basta pro JSON."}),
                "dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "keep_loaded": ("BOOLEAN", {"default": True,
                    "tooltip": "Mantem o Qwen na VRAM entre execucoes (mais rapido em varias rodadas)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("sam_prompt", "edit_prompt", "info")
    OUTPUT_TOOLTIPS = (
        "O QUE SEGMENTAR -> ligue no 'sam_prompt' do AutoEdit Mask (ou no 'prompt' do SAM3 Video Segmentation).",
        "O QUE GERAR -> ligue no seu CLIP Text Encode / Prompt Guide (positivo) do Bernini.",
        "Resumo do que foi decidido + avisos (o que o LLM respondeu, se houve override, etc).",
    )
    FUNCTION = "route"
    CATEGORY = CAT
    DESCRIPTION = (
        "AutoEdit Router (Bruxos): quebra UMA instrucao em linguagem natural (pode ser em portugues) "
        "nos DOIS textos que o pipeline precisa: o que SEGMENTAR (pro SAM3, com o nome concreto do objeto, "
        "porque o SAM3 e ruim com frases relativas tipo 'o objeto da mao') e o que GERAR (pro Bernini, em ingles). "
        "O Qwen-VL OLHA um keyframe pra nomear o objeto de verdade. Tem override manual dos dois campos."
    )

    def route(self, images, instrucao, model_name,
              frame_index=0, sam_prompt_override="", edit_prompt_override="",
              max_new_tokens=300, dtype="fp16", device="auto", keep_loaded=True, seed=0):
        sam_ov = (sam_prompt_override or "").strip()
        edit_ov = (edit_prompt_override or "").strip()
        instr = (instrucao or "").strip()

        # Se o usuario sobrescreveu OS DOIS, nem carregamos o LLM.
        if sam_ov and edit_ov:
            info = "override total: LLM nao foi chamado."
            print(f"[Bruxos AutoEdit Router] {info}", flush=True)
            return (sam_ov, edit_ov, info)

        if not instr and not (sam_ov and edit_ov):
            raise ValueError(
                "[Bruxos AutoEdit Router] 'instrucao' vazia. Escreva o que voce quer mudar "
                "(ex.: 'troque o objeto da mao do personagem por um rato'), ou preencha "
                "os dois overrides na mao."
            )
        if not _HAS_QWEN:
            raise RuntimeError(
                "[Bruxos AutoEdit Router] Qwen-VL indisponivel (transformers nao importou). "
                "Rode: pip install -U transformers accelerate -- ou preencha "
                "sam_prompt_override e edit_prompt_override na mao."
            )

        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        model, processor = _qwen_load(model_name, dtype, dev)

        idx = max(0, min(int(frame_index), int(images.shape[0]) - 1))
        pil = [_to_pil(images[idx])]

        user_txt = f"{_ROUTER_SYSTEM}\n\nEdit instruction: {instr}"
        content = [{"type": "image", "image": pil[0]}, {"type": "text", "text": user_txt}]
        messages = [{"role": "user", "content": content}]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=pil, return_tensors="pt", padding=True).to(dev)
        except Exception:
            inputs = processor(text=[user_txt], images=pil, return_tensors="pt", padding=True).to(dev)

        if seed:
            try:
                torch.manual_seed(int(seed))
            except Exception:
                pass
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0].strip()

        if not keep_loaded:
            try:
                from .nodes import _BX_QWEN_CACHE
            except Exception:
                try:
                    from nodes import _BX_QWEN_CACHE
                except Exception:
                    _BX_QWEN_CACHE = None
            if _BX_QWEN_CACHE is not None:
                _BX_QWEN_CACHE.update({"name": None, "model": None, "processor": None})
            try:
                del model, processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        seg, edt, err = _parse_router_json(raw)
        avisos = []
        if err:
            # Fallback honesto: sem JSON, nao inventamos alvo de segmentacao.
            avisos.append(f"LLM nao devolveu JSON ({err}); usando a resposta crua como edit_prompt.")
            edt = edt or raw
            seg = seg or ""

        # overrides tem prioridade
        if sam_ov:
            avisos.append(f"sam_prompt sobrescrito na mao (LLM sugeriu: {seg!r}).")
            seg = sam_ov
        if edit_ov:
            avisos.append("edit_prompt sobrescrito na mao.")
            edt = edit_ov

        if not seg:
            avisos.append(
                "ATENCAO: sem alvo de segmentacao. O AutoEdit Mask vai falhar -- "
                "preencha 'sam_prompt_override' (ex.: 'blue mug')."
            )

        info = (f"segment={seg!r} | edit={edt[:90]!r}{'...' if len(edt) > 90 else ''}"
                + (" | " + " ".join(avisos) if avisos else ""))
        print(f"[Bruxos AutoEdit Router] frame={idx}\n"
              f"[Bruxos AutoEdit Router]   SAM3  <- {seg!r}\n"
              f"[Bruxos AutoEdit Router]   Bernini <- {edt[:160]!r}{'...' if len(edt) > 160 else ''}"
              + ("\n[Bruxos AutoEdit Router]   avisos: " + " ".join(avisos) if avisos else ""),
              flush=True)
        return (seg, edt, info)


# ---------------------------------------------------------------------------
# MASCARA (chama o SAM3 por dentro)
# ---------------------------------------------------------------------------
def _sam3_nodes():
    """Importa as classes do comfyui-easy-sam3. Retorna (Loader, VideoSeg) ou
    (None, None) com o motivo em log -- nunca quebra o import do pacote."""
    try:
        import importlib
        m = importlib.import_module("custom_nodes.comfyui-easy-sam3.nodes")
        return m.LoadSam3Model, m.Sam3VideoSegmentation
    except Exception:
        pass
    # o nome da pasta tem hifens -> import por caminho
    try:
        import importlib.util
        import os
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p = os.path.join(here, "comfyui-easy-sam3", "nodes.py")
        if not os.path.isfile(p):
            return None, None
        spec = importlib.util.spec_from_file_location("_bx_easy_sam3_nodes", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "LoadSam3Model", None), getattr(mod, "Sam3VideoSegmentation", None)
    except Exception as e:
        logging.warning(f"[Bruxos AutoEdit] nao consegui importar o comfyui-easy-sam3: {e}")
        return None, None


def _first_out(out):
    """Extrai a 1a saida (cobre NodeOutput V3 com .args, tupla e indexavel)."""
    args = getattr(out, "args", None)
    if isinstance(args, (tuple, list)) and len(args):
        return args[0]
    if isinstance(out, (tuple, list)):
        return out[0]
    try:
        return out[0]
    except Exception:
        return out


def _overlay_preview(images, mask, cor=(1.0, 0.15, 0.45), alpha=0.55):
    """images [T,H,W,3] 0..1 + mask [T,H,W] -> preview [T,H,W,3] com a mascara
    pintada por cima. E so pra CONFERENCIA visual antes do render."""
    T = int(images.shape[0])
    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0)
    # alinha contagem de frames
    if m.shape[0] != T:
        if m.shape[0] == 1:
            m = m.repeat(T, 1, 1)
        elif m.shape[0] > T:
            m = m[:T]
        else:
            m = torch.cat([m, m[-1:].repeat(T - m.shape[0], 1, 1)], dim=0)
    # alinha resolucao
    if m.shape[-2:] != images.shape[1:3]:
        m = torch.nn.functional.interpolate(
            m.unsqueeze(1), size=(int(images.shape[1]), int(images.shape[2])),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
    m = m.clamp(0, 1).unsqueeze(-1).to(images.dtype).to(images.device)
    tint = torch.tensor(cor, dtype=images.dtype, device=images.device).view(1, 1, 1, 3)
    return (images * (1.0 - m * alpha) + tint * (m * alpha)).clamp(0, 1)


class BruxosAutoEditMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": ("EASY_SAM3_MODEL", {"tooltip": "Saida do 'Load SAM3 Model'. IMPORTANTE: carregue com segmentor='video' (nao 'image'), senao o rastreio nao funciona."}),
                "images": ("IMAGE", {"tooltip": "Frames do video-fonte (os MESMOS que vao pro Bernini)."}),
                "sam_prompt": ("STRING", {"multiline": False, "default": "", "forceInput": True,
                    "tooltip": "O que segmentar. Ligue aqui o 'sam_prompt' do AutoEdit Router (ou digite via sam_prompt_manual)."}),
            },
            "optional": {
                "sam_prompt_manual": ("STRING", {"multiline": False, "default": "",
                    "tooltip": "MANDA MAIS que a entrada de fio. Preencha quando o SAM3 estiver pegando o objeto errado. Curto e em INGLES: 'blue mug'."}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Frame onde o prompt inicial e aplicado. Escolha um em que o objeto apareca INTEIRO e desobstruido."}),
                "propagation_direction": (["both", "forward", "backward"], {"default": "both",
                    "tooltip": "Pra que lado propagar o rastreio a partir do frame_index. 'both' cobre o video todo mesmo comecando no meio."}),
                "score_threshold_detection": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Confianca minima da deteccao. Abaixe se ele NAO acha o objeto; suba se ele pega coisa demais."}),
                "new_det_thresh": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Confianca pra criar um objeto NOVO durante o rastreio. Suba pra evitar que ele invente objetos parecidos no meio do clipe."}),
                "object_id": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "ID do objeto rastreado. So mexe se for rastrear varios."}),
                "pontos_positivos": ("STRING", {"multiline": False, "default": "",
                    "tooltip": "Cliques APONTANDO o objeto, quando o texto nao basta. JSON: [{\"x\": 512, \"y\": 300}]. "
                               "Coordenadas em pixels do frame_index. Compativel com o Points Editor do KJNodes."}),
                "pontos_negativos": ("STRING", {"multiline": False, "default": "",
                    "tooltip": "Cliques do que EXCLUIR (ex.: a mao, se ele estiver pegando a mao junto). Mesmo formato JSON."}),
                "mask_grow": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1,
                    "tooltip": "Dilata (+) ou contrai (-) a mascara final, em px. SUBA quando o objeto novo for MAIOR que o original "
                               "(ex.: caneca -> rato): inpaint mascarado nao cresce alem da mascara."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Suaviza a borda da mascara (feather), em px. Evita emenda dura. Voce tambem pode deixar isso pro Bernini."}),
                "keep_model_loaded": ("BOOLEAN", {"default": False,
                    "tooltip": "Mantem o SAM3 na VRAM. Desligue se estiver apertado de memoria pro Bernini."}),
                "preview_alpha": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Opacidade do rosa do preview. So afeta a saida 'preview', nunca a mascara real."}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "preview", "info")
    OUTPUT_TOOLTIPS = (
        "Mascara rastreada [T,H,W] -> ligue no 'region_mask' do Bernini Infinity (mask_mode=bbox ou inpaint).",
        "Video com a mascara pintada de rosa -> CONFIRA AQUI antes de rodar o render inteiro.",
        "Cobertura da mascara por frame e avisos (frames vazios, etc).",
    )
    FUNCTION = "build"
    CATEGORY = CAT
    DESCRIPTION = (
        "AutoEdit Mask (Bruxos): chama o SAM3 Video Segmentation por dentro e devolve a mascara "
        "JA RASTREADA em todos os frames, no formato que o region_mask do Bernini Infinity espera, "
        "mais um PREVIEW visual pra voce conferir antes de gastar o render. Aceita override do texto "
        "e cliques positivos/negativos pra quando o grounding por texto errar o objeto."
    )

    def build(self, sam3_model, images, sam_prompt,
              sam_prompt_manual="", frame_index=0, propagation_direction="both",
              score_threshold_detection=0.5, new_det_thresh=0.7, object_id=1,
              pontos_positivos="", pontos_negativos="",
              mask_grow=0, mask_blur=0, keep_model_loaded=False, preview_alpha=0.55):
        if not _OK:
            raise RuntimeError("[Bruxos AutoEdit Mask] torch indisponivel.")
        _, VideoSeg = _sam3_nodes()
        if VideoSeg is None:
            raise RuntimeError(
                "[Bruxos AutoEdit Mask] nao encontrei o pacote 'comfyui-easy-sam3'. "
                "Ele precisa estar em ComfyUI/custom_nodes/comfyui-easy-sam3. "
                "Alternativa: use o node 'SAM3 Video Segmentation' direto e ligue a saida "
                "'masks' no region_mask do Bernini."
            )

        prompt = (sam_prompt_manual or "").strip() or (sam_prompt or "").strip()
        if not prompt and not (pontos_positivos or "").strip():
            raise ValueError(
                "[Bruxos AutoEdit Mask] sem alvo: 'sam_prompt' vazio e sem pontos positivos. "
                "Ligue o AutoEdit Router, ou digite em 'sam_prompt_manual' (ex.: 'blue mug'), "
                "ou aponte pontos em 'pontos_positivos'."
            )

        T = int(images.shape[0])
        idx = max(0, min(int(frame_index), T - 1))

        def _coords(s):
            s = (s or "").strip()
            return s if s else None

        try:
            out = VideoSeg.execute(
                sam3_model=sam3_model,
                video_frames=images,
                prompt=prompt,
                frame_index=idx,
                object_id=int(object_id),
                score_threshold_detection=float(score_threshold_detection),
                new_det_thresh=float(new_det_thresh),
                propagation_direction=str(propagation_direction),
                start_frame_index=0,
                max_frames_to_track=-1,
                close_after_propagation=True,
                keep_model_loaded=bool(keep_model_loaded),
                session_id=None,
                extra_config=None,
                positive_coords=_coords(pontos_positivos),
                negative_coords=_coords(pontos_negativos),
                bbox=None,
            )
        except Exception as e:
            raise RuntimeError(
                f"[Bruxos AutoEdit Mask] o SAM3 Video Segmentation falhou: {e}\n"
                f"Checagens: (1) o 'Load SAM3 Model' esta com segmentor='video'? "
                f"(2) o prompt {prompt!r} descreve algo visivel no frame {idx}? "
                f"(3) tente abaixar o score_threshold_detection."
            ) from e

        mask = _first_out(out)
        if mask is None:
            raise RuntimeError("[Bruxos AutoEdit Mask] o SAM3 nao devolveu mascara.")
        if mask.dim() == 4:                      # [T,1,H,W] ou [T,H,W,C]
            mask = mask.squeeze(1) if mask.shape[1] == 1 else mask[..., 0]
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float().clamp(0, 1)

        # grow/blur opcionais (o Bernini tambem faz, mas as vezes voce quer ver
        # o resultado JA crescido no preview)
        if int(mask_grow) != 0 or int(mask_blur) > 0:
            x = mask.unsqueeze(1)
            g = int(mask_grow)
            if g > 0:
                x = torch.nn.functional.max_pool2d(x, kernel_size=g * 2 + 1, stride=1, padding=g)
            elif g < 0:
                a = -g
                x = -torch.nn.functional.max_pool2d(-x, kernel_size=a * 2 + 1, stride=1, padding=a)
            b = int(mask_blur)
            if b > 0:
                k = b * 2 + 1
                co = torch.arange(k, dtype=torch.float32, device=x.device) - b
                sig = b * 0.5 + 1e-6
                g1 = torch.exp(-(co ** 2) / (2 * sig * sig))
                g1 = g1 / g1.sum()
                x = torch.nn.functional.conv2d(x, g1.view(1, 1, 1, k), padding=(0, b))
                x = torch.nn.functional.conv2d(x, g1.view(1, 1, k, 1), padding=(b, 0))
            mask = x.squeeze(1).clamp(0, 1)

        # diagnostico: quantos frames ficaram SEM mascara (rastreio perdido)
        per_frame = mask.flatten(1).mean(dim=1)
        vazios = int((per_frame < 1e-4).sum())
        cob = float(per_frame.mean()) * 100.0
        avisos = ""
        if vazios:
            avisos = (f" | ATENCAO: {vazios}/{int(mask.shape[0])} frames SEM mascara "
                      f"(o rastreio perdeu o objeto). Tente outro frame_index, "
                      f"propagation_direction=both, ou abaixe o score_threshold_detection.")
        if cob > 45.0:
            avisos += (f" | ATENCAO: a mascara cobre {cob:.0f}% da tela -- provavelmente pegou "
                       f"o objeto errado (fundo/pessoa inteira). Use um sam_prompt mais especifico "
                       f"ou pontos negativos.")

        preview = _overlay_preview(images, mask, alpha=float(preview_alpha))
        info = (f"prompt={prompt!r} frame={idx} dir={propagation_direction} | "
                f"mask {tuple(mask.shape)} | cobertura media {cob:.1f}%{avisos}")
        print(f"[Bruxos AutoEdit Mask] {info}", flush=True)
        return (mask, preview, info)


NODE_CLASS_MAPPINGS = {
    "BruxosAutoEditRouter": BruxosAutoEditRouter,
    "BruxosAutoEditMask": BruxosAutoEditMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosAutoEditRouter": "AutoEdit Router · instrucao -> alvo + prompt (Bruxos)",
    "BruxosAutoEditMask": "AutoEdit Mask · SAM3 rastreado -> mascara (Bruxos)",
}
