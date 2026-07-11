"""
ComfyUI-Bruxos-do-VFX — Prompt Guide
====================================
Inspirado no "Bernini Prompt Guide" do Deno: um encoder de CLIP que guarda
'system prompts' por tarefa (cada opcao e um comando pro modelo) e negativos
oficiais por modelo. Aqui com mais modelos: Bernini, Wan 2.2, Wan 2.1,
LTX 2.3 (Edit Anything LoRA) e Seedance 2.

Tudo e editavel: ao escolher um modelo + tarefa, o system prompt do preset e
usado (e auto-preenchido pela extensao JS); se voce digitar algo no campo de
system prompt, esse texto tem prioridade.
"""

import logging

# ---------------------------------------------------------------------------
# Biblioteca de presets.  Estrutura:
#   PRESETS[model] = {
#       "tasks": { "Nome da tarefa": "system prompt ...", ... },
#       "negatives": { "Nome do preset": "texto negativo ...", ... },
#       "default_negative": "Nome do preset",
#   }
# Os system prompts sao pontos de partida curados e 100% editaveis no node.
# ---------------------------------------------------------------------------

# Negativo padrao do Wan (o mesmo difundido pela comunidade / oficial).
_WAN_NEG = (
    "色彩艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
_WAN_NEG_EN = (
    "overexposed, static, blurry details, subtitles, style, artwork, painting, "
    "still, grayish, worst quality, low quality, JPEG artifacts, ugly, mutilated, "
    "extra fingers, badly drawn hands, badly drawn face, deformed, disfigured, "
    "malformed limbs, fused fingers, motionless frame, cluttered background, "
    "three legs, crowd in background, walking backwards"
)
_GEN_NEG = (
    "worst quality, low quality, blurry, low detail, jpeg artifacts, deformed, "
    "disfigured, extra limbs, fused fingers, flicker, ghosting, smearing, "
    "warping, watermark, text, logo"
)


# Base comum das 22 tarefas de edicao do Bernini-Bench (5 dimensoes). Cada
# entrada = system prompt de edicao + diretriz curta da categoria. Editaveis
# no node; sao pontos de partida alinhados a taxonomia do paper.
_BERN_EDIT = "You are a helpful assistant specialized in video editing. "


PRESETS = {
    # =====================================================================
    "Bernini": {
        "tasks": {
            "Default": "You are a helpful assistant.",
            "Text to Image": "You are a helpful assistant specialized in text-to-image generation.",
            "Text to Video": "You are a helpful assistant specialized in text-to-video generation.",
            "Image Edit": "You are a helpful assistant specialized in image editing.",
            "Subject to Image": "You are a helpful assistant specialized in subject-to-image generation.",
            "Image to Video": "You are a helpful assistant specialized in image-to-video generation.",
            "Video Edit": "You are a helpful assistant specialized in video editing.",
            "Subject to Video": "You are a helpful assistant specialized in subject-to-video generation.",
            "Video Propagation": "You are a helpful assistant specialized in video editing on content propagation.",
            "Reference Video Edit": "You are a helpful assistant specialized in video editing with reference.",
            "Ads Insertion": "You are a helpful assistant specialized in ads insertion.",
            "Video Reference Control": "You are a helpful assistant for editing. You may need to adjust the subject's action or position.",
            "Motion / Style Edit": "You are a helpful assistant for editing. You might need to adjust the video's style, lighting, colors, textures, and the subject's pose or action.",

            # ---- Bernini-Bench: 22 tarefas de edicao (5 dimensoes) ----
            # Subject Editing (6)
            "Subject: Remove": _BERN_EDIT + "Task: object removal. Remove the specified subject/object and reconstruct a plausible background where it was; keep everything else unchanged and temporally stable. Prefer a short, specific instruction (name the object and its position).",
            "Subject: Add": _BERN_EDIT + "Task: object addition. Add a new subject/object; describe its appearance, position in the frame, scale, and how it interacts with the existing scene and lighting; keep the rest of the video unchanged.",
            "Subject: Replace": _BERN_EDIT + "Task: subject replacement. Describe BOTH what is being replaced (and its location) AND the new subject; keep a similar scale and region, and preserve the surrounding content.",
            "Subject: Pose/Action": _BERN_EDIT + "Task: pose/action change (e.g., sitting to standing, walking to running) while preserving the subject's identity and clothing and the scene; keep the motion natural and temporally coherent.",
            "Subject: Attribute": _BERN_EDIT + "Task: attribute change (color, clothing, material, hairstyle) on the target subject while preserving its identity and everything else; keep the change consistent across all frames.",
            "Subject: Expression": _BERN_EDIT + "Task: facial expression change (e.g., neutral to smiling) while strictly preserving identity, pose, and background; keep it subtle and temporally smooth.",

            # Scene & Environment (4)
            "Scene: Background": _BERN_EDIT + "Task: background/scene change behind the subject while keeping the subject unchanged and correctly composited; match lighting, perspective, and shadows.",
            "Scene: Atmosphere": _BERN_EDIT + "Task: atmosphere/mood change (e.g., add fog, dust, haze, god rays) while preserving subjects and layout; keep it consistent across frames.",
            "Scene: Weather": _BERN_EDIT + "Task: weather change (rain, snow, sunshine) applied consistently across the whole video; adjust lighting and reflections accordingly while preserving subjects and scene structure.",
            "Scene: Time/Season": _BERN_EDIT + "Task: time-of-day or season change (day to night, summer to winter); adjust lighting, color temperature, and shadows consistently while preserving scene geometry and subjects.",

            # Visual & Style (4)
            "Style: Material": _BERN_EDIT + "Task: material/texture change of a specified surface or object (e.g., wood to metal) while keeping its shape and the rest of the scene unchanged.",
            "Style: Style Transfer": _BERN_EDIT + "Task: style transfer. Convert the whole video into a target artistic style, applied consistently across all frames while preserving structure and motion. Use the template: 'Convert the video into a <STYLE> style.' (e.g. Watercolor, Van Gogh, Claymation, Ghibli, 3D animation).",
            "Style: Color": _BERN_EDIT + "Task: color adjustment/grade (white balance, saturation, tint) applied consistently across the whole video while preserving content and detail.",
            "Style: Filter": _BERN_EDIT + "Task: visual filter/effect (e.g., vintage, black-and-white, film grain, bloom) applied uniformly across the video while preserving content.",

            # Camera & Motion (4)
            "Camera: Motion": _BERN_EDIT + "Task: change or add camera motion (pan, zoom, dolly, orbit) while keeping subject identity and scene consistent and sharply tracked throughout.",
            "Camera: Position": _BERN_EDIT + "Task: reposition a subject/object within the frame (e.g., move the distant boat closer into the midground) while preserving its identity and matching perspective, scale, and occlusion.",
            "Camera: Viewpoint": _BERN_EDIT + "Task: viewpoint/perspective change of the scene while preserving subject identity and scene content; keep the geometry plausible.",
            "Camera: Focus": _BERN_EDIT + "Task: focus control (rack focus / depth of field). Shift the focus to the specified subject, keeping it sharp and consistently tracked while the rest is appropriately defocused.",

            # Reasoning (4)
            "Reasoning: Causal": _BERN_EDIT + "Task: causal reasoning edit. Infer the physically plausible consequence implied by the instruction (e.g., prolonged heavy rain extinguishes a fire) and render the resulting state and its evolution over time.",
            "Reasoning: Temporal": _BERN_EDIT + "Task: temporal reasoning edit. Show how the scene evolves over time as implied by the instruction (before to after, progression), keeping it physically consistent.",
            "Reasoning: Spatial": _BERN_EDIT + "Task: spatial reasoning edit. Perform layout/position changes that require understanding 3D scene relationships (occlusion, depth, relative placement) while keeping content consistent.",
            "Reasoning: Interaction": _BERN_EDIT + "Task: interaction edit. Introduce or modify a natural interaction between subjects (e.g., a new person interacting with the existing ones) while keeping identities and the scene coherent.",
        },
        "negatives": {"Official Bernini": _GEN_NEG, "None": ""},
        "default_negative": "Official Bernini",
    },
    # =====================================================================
    "Wan 2.2": {
        "tasks": {
            "Default": "",
            "Text to Video": "",
            "Image to Video": "",
            "Cinematic": "Cinematic, filmic lighting, shallow depth of field, natural color grading, smooth camera motion.",
            "Realistic": "Photorealistic, natural lighting, true-to-life skin and textures, high detail.",
            "Anime": "Anime style, clean lineart, vibrant cel shading, expressive characters.",
        },
        "negatives": {"Official Wan2.2 (中文)": _WAN_NEG, "Official Wan2.2 (EN)": _WAN_NEG_EN, "None": ""},
        "default_negative": "Official Wan2.2 (中文)",
    },
    # =====================================================================
    "Wan 2.1": {
        "tasks": {
            "Default": "",
            "Text to Video": "",
            "Image to Video": "",
            "Cinematic": "Cinematic, filmic lighting, shallow depth of field, smooth camera motion.",
            "Realistic": "Photorealistic, natural lighting, high detail.",
        },
        "negatives": {"Official Wan2.1 (中文)": _WAN_NEG, "Official Wan2.1 (EN)": _WAN_NEG_EN, "None": ""},
        "default_negative": "Official Wan2.1 (中文)",
    },
    # =====================================================================
    # LTX 2.3 + Edit Anything LoRA — baseado no guia oficial do LoRA.
    "LTX 2.3 (Edit Anything)": {
        "tasks": {
            "Default": "",
            "Add": ("Edit task: ADD. Describe the new subject in detail (15-30+ words), its position in "
                    "the frame, and the surrounding context. Pattern: 'Add <detailed subject>, <position>, "
                    "<surrounding context>.'"),
            "Remove": ("Edit task: REMOVE. Keep it very short (4-10 words): 'Remove the <object> (+ optional "
                       "position)'. Do NOT over-describe — long remove prompts drift out of distribution and fail."),
            "Replace": ("Edit task: REPLACE (20-35 words). Describe BOTH the original subject and its location "
                        "AND the new subject. Pattern: 'Replace <original + location> with <new subject>.'"),
            "Style": ("Edit task: STYLE. Use EXACTLY this template: 'Convert the video into a <STYLE> style.' "
                      "(e.g. Watercolor Painting, Van Gogh, Claymation, Ghibli, Pop Art). Deviations degrade quality."),
            "Motion Transfer (v0.1)": ("Motion transfer: the first frame was edited externally to insert the new "
                                       "subject; the model copies motion from the guide. Describe the inserted subject "
                                       "and the action being preserved. Avoid fast/chaotic motion and hard cuts."),
            "Reference Add (Ref V2V)": ("Reference Add (25-40 words): the reference image holds the new subject's "
                                        "appearance; the caption carries position, pose, action and context. "
                                        "Pattern: 'Add <full subject description>, <position>, <context>.'"),
            "Reference Replace (Ref V2V)": ("Reference Replace (25-40 words): describe what is being replaced and "
                                            "its location, plus the new subject (whose appearance comes from the "
                                            "reference image). Keep similar scale/region as the original."),
        },
        "negatives": {"LTX default": _GEN_NEG, "None": ""},
        "default_negative": "LTX default",
    },
    # =====================================================================
    # Seedance 2 — presets genericos (pontos de partida; ajuste ao seu fluxo).
    "Seedance 2": {
        "tasks": {
            "Default": "",
            "Text to Video": "High-quality video, coherent motion, consistent subject and lighting across frames.",
            "Image to Video": "Animate the input image into a coherent video, preserving its content and identity.",
            "Cinematic": "Cinematic look, dynamic but smooth camera, filmic color, shallow depth of field.",
            "Style": "Apply a consistent visual style across the whole video while keeping the subject recognizable.",
        },
        "negatives": {"Seedance default": _GEN_NEG, "None": ""},
        "default_negative": "Seedance default",
    },
}

MODELS = list(PRESETS.keys())


def _ordered_union(seqs):
    out = []
    for s in seqs:
        for x in s:
            if x not in out:
                out.append(x)
    return out


ALL_TASKS = _ordered_union([list(PRESETS[m]["tasks"].keys()) for m in MODELS])
ALL_NEGATIVES = _ordered_union([list(PRESETS[m]["negatives"].keys()) for m in MODELS])


def presets_payload():
    """JSON-serializavel p/ a extensao JS (filtro de dropdown + auto-fill)."""
    return {
        "models": MODELS,
        "presets": {
            m: {
                "tasks": PRESETS[m]["tasks"],
                "negatives": PRESETS[m]["negatives"],
                "default_negative": PRESETS[m]["default_negative"],
            } for m in MODELS
        },
    }


def _encode(clip, text):
    if clip is None:
        return None
    tokens = clip.tokenize(text)
    if hasattr(clip, "encode_from_tokens_scheduled"):
        return clip.encode_from_tokens_scheduled(tokens)
    out = clip.encode_from_tokens(tokens, return_pooled=True)
    if isinstance(out, tuple):
        cond, pooled = out
        return [[cond, {"pooled_output": pooled}]]
    return [[out, {}]]


class BruxosPromptGuide:
    """Guia de prompts multimodelo (Bernini / Wan / LTX / Seedance) com presets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODELS, {"default": MODELS[0],
                          "tooltip": "Qual modelo. Define quais tarefas e negativos ficam disponiveis."}),
                "task": (ALL_TASKS, {"default": "Default",
                          "tooltip": "A tarefa/comando. Cada opcao carrega um system prompt proprio (auto-preenchido abaixo)."}),
                "negative_preset": (ALL_NEGATIVES, {"default": "None",
                          "tooltip": "Negativo pronto do modelo (ex: Official Wan2.2). Editavel no campo abaixo."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                          "tooltip": "Sua instrucao/descricao principal (o que voce quer gerar ou editar)."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "CLIP/encoder do modelo. Se ligado, sai CONDITIONING; senao, so os textos."}),
                "prepend_system": ("BOOLEAN", {"default": True,
                          "tooltip": "Coloca o system prompt antes da sua instrucao no positivo."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                          "tooltip": "System prompt da tarefa (auto-preenchido). Se voce editar, seu texto tem prioridade."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "",
                          "tooltip": "Negativo. Se vazio, usa o negative_preset; se preenchido, tem prioridade."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "positive_text", "negative_text")
    FUNCTION = "build"
    CATEGORY = "Bruxos do VFX/Prompt"
    DESCRIPTION = (
        "Prompt Guide (Bruxos) — encoder com presets de comando por modelo, no estilo do "
        "Bernini Prompt Guide do Deno, mas multimodelo: Bernini, Wan 2.2, Wan 2.1, "
        "LTX 2.3 (Edit Anything) e Seedance 2.\n"
        "- model: escolhe o modelo (filtra as tarefas e negativos).\n"
        "- task: o comando/tarefa; cada um traz um system prompt proprio.\n"
        "- negative_preset: negativo pronto (ex: Official Wan2.2 oficial em chines/EN).\n"
        "- prompt: sua instrucao principal.\n"
        "- clip (opcional): se ligado, sai CONDITIONING positivo/negativo; senao, use as saidas de texto.\n"
        "- prepend_system: junta o system prompt antes da sua instrucao.\n"
        "- system_prompt / negative_prompt: editaveis; se preenchidos, tem prioridade sobre o preset.\n"
        "SAIDAS: positive, negative (CONDITIONING), positive_text, negative_text (STRING)."
    )

    def build(self, model, task, negative_preset, prompt,
              clip=None, prepend_system=True, system_prompt="", negative_prompt=""):
        mp = PRESETS.get(model, PRESETS[MODELS[0]])

        sys_text = (system_prompt or "").strip()
        if not sys_text:
            sys_text = mp["tasks"].get(task, mp["tasks"].get("Default", "")).strip()

        user_text = (prompt or "").strip()
        if prepend_system and sys_text:
            positive_text = f"{sys_text}\n\n{user_text}".strip()
        else:
            positive_text = user_text

        neg_text = (negative_prompt or "").strip()
        if not neg_text:
            neg_text = mp["negatives"].get(negative_preset, "")
            if not neg_text and negative_preset not in mp["negatives"]:
                # preset de outro modelo selecionado por engano -> usa o default do modelo
                neg_text = mp["negatives"].get(mp["default_negative"], "")

        pos_cond = _encode(clip, positive_text) if clip is not None else None
        neg_cond = _encode(clip, neg_text) if clip is not None else None
        if clip is None:
            logging.info("[BruxosPromptGuide] sem CLIP: retornando apenas os textos.")
        return (pos_cond, neg_cond, positive_text, neg_text)


NODE_CLASS_MAPPINGS = {"BruxosPromptGuide": BruxosPromptGuide}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosPromptGuide": "Prompt Guide (Bruxos)"}
