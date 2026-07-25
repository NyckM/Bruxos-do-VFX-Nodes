# -*- coding: utf-8 -*-
r"""
Templates OFICIAIS do prompt-enhancer do Bernini (ByteDance)
============================================================
System prompts + templates de tarefa copiados de `bernini.prompt_enhancer`
(https://github.com/bytedance/Bernini), Apache-2.0, (c) Bytedance Ltd.

So DADOS + helpers de montagem (sem I/O, sem LLM). O node de Prompt Enhancer
(Qwen-VL) do pacote usa isto pra montar a requisicao por tarefa, seguindo a
receita oficial — incluindo o r2v/i2v com referencia a `image0`, `image1`...
que e o que faz o Bernini PRESERVAR a identidade da referencia.

Creditos: ByteDance/Bernini (Apache-2.0). Reproduzido com atribuicao.
"""

import re

# codigo -> rotulo bilingue (igual ao upstream + variante keyframe da comunidade)
TASK_TYPES = (
    "off (nao usar template oficial)",
    "t2i - text to image",
    "t2v - text to video",
    "i2i - image editing",
    "r2i - subject to image",
    "ri2i - reference-guided image edit",
    "i2v - image to video",
    "i2v_kf - image to video (keyframe/sequencial)",
    "v2v - video editing",
    "mv2v - multi-source video edit",
    "r2v - reference to video",
    "vi2v - video insert reference",
    "rv2v - reference-guided video edit",
    "vrc2v - reference-content video edit",
    "ads2v - ad insertion",
)

TASK_CODES = ("t2i", "t2v", "i2i", "r2i", "ri2i", "i2v", "i2v_kf", "v2v",
              "mv2v", "r2v", "vi2v", "rv2v", "vrc2v", "ads2v")

# tarefas cuja saida do LLM e um JSON {"rewritten_text": "..."} a ser extraido
JSON_MODE_TASKS = frozenset({"r2i", "r2v", "rv2v", "vrc2v", "ri2i"})


def parse_task_code(task_type):
    """Extrai o codigo curto ('i2v') de um rotulo ('i2v - image to video')."""
    if not task_type:
        return None
    head = task_type.split(" - ", 1)[0]
    head = re.sub(r"\s*\(.*?\)\s*", "", head).strip()
    return head if head in TASK_CODES else None


SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "r2i": "You are a helpful assistant specialized in subject-to-image generation.",
    "ri2i": "You are a helpful assistant specialized in reference-guided image editing.",
    "i2v": "You are a helpful assistant specialized in image-to-video generation.",
    "i2v_kf": "You are a helpful assistant specialized in image-to-video generation.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "vi2v": "You are a helpful assistant specialized in video editing on content propagation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
    "ads2v": "You are a helpful assistant specialized in ads insertion.",
    "vrc2v": ("You are a helpful assistant for editing. "
              "You may need to adjust the subject's action or position."),
    "mv2v": ("You are a helpful assistant for editing. You might need to adjust the "
             "video's style, lighting, colors, textures, and the subject's pose or action."),
}


def get_system_prompt(task_code):
    return SYSTEM_PROMPTS.get(task_code, SYSTEM_PROMPTS["default"])


# --- T2V/T2I: system prompt cinematografico oficial (versao EN condensada) ---
# O upstream traz a versao completa (bilingue). Aqui usamos a diretriz de
# aplicacao (o LLM ja recebe a lista de opcoes de luz/lente/composicao).
T2V_CINEMATIC = (
    "You are a film director. Rewrite the user's raw prompt into a high-quality ENGLISH prompt, "
    "complete and expressive. Rules:\n"
    "1. Without changing the original meaning (subject, action), add up to 4 cinematic settings "
    "from: time [Day/Night/Dawn/Sunrise; default Day], light source [Daylight/Artificial/Moonlight/"
    "Practical/Firelight/Fluorescent/Overcast/Sunny], light intensity [Soft/Hard], color tone "
    "[Warm/Cool/Mixed], light angle [Top/Side/Under/Edge], shot size [Medium/Medium close-up/Wide/"
    "Medium wide/Close-up/Extreme close-up/Extreme wide; default Medium or Wide], camera angle "
    "[Over-the-shoulder/Low/High/Dutch/Aerial/Overhead; skip if prompt already has camera motion], "
    "composition [Center/Balanced/Right-heavy/Left-heavy/Symmetrical/Short-side; default Center]. "
    "Pick only what fits; not every field is required.\n"
    "2. Enrich subject features (appearance, expression, count, pose) WITHOUT adding subjects not in "
    "the original prompt; add background detail.\n"
    "3. Do NOT output literary mood/feeling descriptions.\n"
    "4. For actions, describe the motion step by step; if none, add natural motion (body sway, dancing) "
    "and subtle background motion (clouds drifting, leaves in wind).\n"
    "5. If no style in the original, add none; if there is a style, put it first; for 2D/illustration "
    "styles, do NOT add cinematic-realism descriptions.\n"
    "6. If the sky is mentioned, make it a clear blue sky (avoid overexposure).\n"
    "7. Output must be ALL English, 60-200 words, no 'Rewritten prompt:' label.\n"
)
T2I_CINEMATIC = (
    "This is a TEXT-TO-IMAGE task. Rewrite into a STATIC image prompt. There is no time sequence — "
    "do NOT describe motion / camera movement / action process; describe only the static state of "
    "scene and subject. Keep the other cinematic aesthetics (light source / intensity / tone / shot "
    "size / camera angle / composition) per the rules below.\n\n" + T2V_CINEMATIC
)


R2V_TEMPLATE = """You are an expert at writing subject-driven video generation prompts. I'm providing you with:
1. {image_num} reference image(s) of the subject(s) that will appear in the video (referred to as image0, image1, image2, ... in order).
2. An original video description text.

Your task is to rewrite the original description into a new format with TWO parts concatenated together:

**Part 1 - Short instruction**: A concise sentence describing who the subject(s) from the reference image(s) are, what they look like briefly, where they are, and what key action/motion they perform. Reference the subject(s) using "image0", "image1", etc. to link them to the provided reference images.

**Part 2 - Long instruction**: A detailed "Generate a video where..." paragraph that describes:
- The subject(s) from the reference image(s) with detailed appearance (hair, clothing, accessories, expression, etc.), referencing them as "the person/man/woman from image0" etc.
- The scene/environment in detail (background, lighting, objects, atmosphere).
- The motion and actions in a step-by-step temporal sequence (at the start..., then..., after that...).
- The motion should remain natural and realistic.

Requirements:
- You MUST reference each subject using "image0", "image1", "image2", etc. to correspond to the provided reference images in order.
- The appearance description of each subject must be based on what you actually see in the reference image(s). Do NOT hallucinate details not visible in the images.
- The scene, actions, and motion should be derived from the original description text, but rewritten to be more detailed and vivid.
- The output must be entirely in English.
- Return ONLY a JSON object with one key: "rewritten_text". The value should be the full rewritten text (short instruction + long instruction concatenated as one string). No extra text.

Original description:
{original_text}
"""

R2I_TEMPLATE = R2V_TEMPLATE.replace("video", "image").replace(
    "action/motion they perform", "visual composition"
).replace(
    "The motion and actions in a step-by-step temporal sequence (at the start..., then..., after that...).\n- The motion should remain natural and realistic.",
    "The composition, framing, and visual emphasis."
).replace("Generate a video where", "Generate an image where")

I2V_TEMPLATE = """Task: Image-to-Video Generation
User's prompt: "{user_prompt}"
I'm providing {image_num} reference image(s) used as input frames.

This may be a single-image or multi-image I2V task; decide by image count and prompt, and return an English prompt:
* Single-image I2V: directly write an English prompt describing the video content (action, camera, scene), following the T2V prompt format.
* First+last-frame I2V (2 images): return "Generate a video based on the first and last frames. " + video description.
* First+middle+last I2V (3 images): return "Generate a video based on the first, middle, and last frames. " + video description.

Output ONLY the final English prompt, no other text.
"""

I2V_KEYFRAME_TEMPLATE = """Task: Image-to-Video Generation with ORDERED KEYFRAMES
User's prompt: "{user_prompt}"
I'm providing {image_num} reference image(s) that are KEYFRAMES in order: image0, image1, image2, ...

Write ONE English video prompt for a Bernini i2v model that produces a continuous video passing through the keyframes IN ORDER. Follow these rules (they come from how Bernini was trained):
- You MUST reference each keyframe explicitly as "from image0", "from image1", "from image2", ... — use the exact wording "from imageN". Do NOT write "in image0" or "as shown in image1" (those do NOT work well).
- The video STARTS at image0 and ENDS at the last image, transitioning smoothly through the intermediate images in order.
- For 2 images: the first frame is image0 and the last frame is image1 (first-frame / last-frame).
- Describe the subject/scene of each keyframe FAITHFULLY based on what is actually visible; do NOT hallucinate details not in the images.
- Describe the action/motion that carries the subject from one keyframe to the next, step by step (starts as ... from image0, then transitions ... , finally settles as ... from image1).
- Keep identity, clothing, and style consistent across the transition.
- Output ONLY the final English prompt, no other text.
"""

V2V_TEMPLATE = """Task: Video Editing
# ROLE
You are an expert Video-to-Video (V2V) Prompt Engineer. Analyze the user's raw editing instruction and the provided source video frames and generate a detailed V2V editing prompt in English.

# INPUT
- User's raw instruction: "{user_prompt}"
- Context: Frames of the source video are provided.

# CORE GENERATION RULE
Unless specified otherwise by the task type, follow a two-part structure:
1. Modifications: what to change (appearance, spatial location, lighting, motion tracking).
2. Preservations: what MUST remain unchanged.
3. Concretization: replace vague references ("more cartoon characters", "change outfits") with specific named instances matching the video's style (e.g. "Hello Kitty, Pikachu, Mickey Mouse"; "a kung fu gi, a navy three-piece suit, a black hoodie with cargo pants"). Never leave generic placeholders.
Describe naturally, e.g. "Add an apple. The table and curtains remain unchanged." (no need to write the labels literally.)

# TASK CATEGORIES
Determine the task type, then use the matching format:
1. Replacement: "Replace [original] with [new]."
2. Addition: "Add [element] + [location/action]."
3. Removal: "Delete [object] + [location]."
4. Subtitle Removal: "Remove subtitles from the video."
5. Depth-to-Video / 6. Sketch-to-Video / 7. Colorization / 8. Inpainting / 9. Detection (mask region) — as their names.
10. Stylization: "Convert the video to [style]: [brief details]." Concise.
11. Mixed: integrate into one cohesive instruction (do NOT list subtasks).
12. Camera Movement: "Apply camera motion: [description]" (e.g. "Apply camera motion: orbit down").
13. Change Camera Perspective: "Switch the camera to a [first/third]-person perspective" OR "Move the camera [how it moves to the desired angle]".
14. Change Focus: "Shift the focus to [subject], making her/him/it sharp. Blur [objects]."
15. Other: generate logically per the Core Generation Rule.

# OUTPUT
Output ONLY the final enhanced English prompt (no explanations/greetings/category name). Do not imagine things not in the video. For camera movement/perspective cases, describe ONLY the camera transformation in one sentence.
"""

I2I_TEMPLATE = V2V_TEMPLATE.replace("Video-to-Video (V2V)", "Image-to-Image (I2I)").replace(
    "video", "image").replace("Video Editing", "Image Editing").replace(
    "the source video frames", "the source image").replace("Frames of the source image are provided.",
    "The source image is provided.")

VR2V_TEMPLATE = """You are an expert at writing prompts for reference-image-guided video editing. I'm providing you with:
1. The first 3 images are uniformly sampled frames from the **source video** to be edited (temporal order: frame0, frame1, frame2).
2. The next {image_num} image(s) are **reference image(s)** guiding the edit (image0, image1, ... in order).
3. An original editing instruction (may be Chinese).

Infer the role of the reference image(s) from the instruction (target object/person for replace/add, target style, target motion/pose, etc.).

Rewrite and enhance into a detailed English prompt: editing instruction + detailed description of the target edited video, as ONE paragraph. Rules:
1. Instruction sentence + detailed target description, one continuous paragraph.
2. Match the edit type verb ("Replace/Remove/Add/Restyle.../Transfer the motion of..."). Do NOT force "replace".
3. Add != Replace: additions are additions; do not change count/positions of existing subjects.
4. Allow natural shape/size differences; do NOT force identical shape/size.
5. Describe the target video directly (no "after editing...").
6. Faithful reference appearance: match what is visible in the reference image; do NOT hallucinate.
7. Screen-perspective left/right (camera view, not the subject's).
8. Explicitly state which source elements remain unchanged (camera framing/motion, lighting, background, other objects, shadows/reflections, scene motion).
9. For style/motion references, describe the resulting style/motion concretely.
10. No parentheses.
11. English only.
12. Similar length/detail to a typical high-quality example.

Return ONLY a JSON object with one key: "rewritten_text". No extra text.

Original instruction:
{original_text}
"""

VI2V_TEMPLATE = """Task: Video Editing with Reference Image (vi2v)
User's editing instruction: "{user_prompt}"
Provided: 3 uniformly sampled source-video frames + {image_num} reference image(s).

Decide among propagation / reference insertion / reference replacement and return an English prompt:
* propagation: return exactly "edit the video following the first frame."
* reference insertion: e.g. "Integrate the tree from the image into the video in a reasonable way."
* reference replacement: describe replacing the corresponding object in the video with the reference object.

Output ONLY the final English prompt.
"""

ADS2V_TEMPLATE = """Task: Ads Insertion in Video
User's instruction: "{user_prompt}"
Provided: 3 uniformly sampled source-video frames for context.

Generate a concise one-sentence English ad-insertion instruction, e.g.:
"Add Starbucks Latte wallpaper on the second floor across the street"

Output ONLY the final English prompt.
"""

RI2I_TEMPLATE = """You are an expert at writing prompts for reference-image-guided image editing. I'm providing you with:
1. The first image is the **source image** to be edited (image0).
2. The next {ref_num} image(s) are **reference image(s)** guiding the edit (image1, image2, ... in order).
3. An original editing instruction (may be Chinese).

Rewrite into a detailed English prompt: editing instruction + detailed description of the target edited image, ONE paragraph. Rules: match the edit verb (not always "replace"); additions are additions; allow natural shape/size differences; describe the target image directly; faithful reference appearance (no hallucination); explicitly preserve unchanged elements (composition, lighting, background, other objects, shadows/reflections); no parentheses; English only; similar length/detail to a high-quality example.

Return ONLY a JSON object with one key: "rewritten_text". No extra text.

Original instruction:
{original_text}
"""

_TEMPLATES = {
    "t2i": T2I_CINEMATIC, "t2v": T2V_CINEMATIC,
    "i2i": I2I_TEMPLATE, "i2v": I2V_TEMPLATE, "i2v_kf": I2V_KEYFRAME_TEMPLATE, "v2v": V2V_TEMPLATE,
    "r2i": R2I_TEMPLATE, "r2v": R2V_TEMPLATE, "ri2i": RI2I_TEMPLATE,
    "rv2v": VR2V_TEMPLATE, "vi2v": VI2V_TEMPLATE, "ads2v": ADS2V_TEMPLATE,
    "mv2v": V2V_TEMPLATE, "vrc2v": VR2V_TEMPLATE,
}


def build_request(task_code, user_prompt, image_num=0):
    """Monta (system_text, user_text, is_json) pro LLM, seguindo a receita
    oficial da tarefa. image_num = quantas referencias/frames o modelo ve."""
    task_code = (task_code or "").strip()
    system = get_system_prompt(task_code)
    tpl = _TEMPLATES.get(task_code)
    up = (user_prompt or "").strip()
    n = int(image_num)
    if tpl is None:
        return system, up, False
    try:
        if task_code in ("t2i", "t2v"):
            user = tpl + "\n\nUser prompt:\n" + up
        elif task_code in ("r2v", "r2i", "rv2v"):
            user = tpl.format(image_num=max(1, n), original_text=up)
        elif task_code == "ri2i":
            user = tpl.format(ref_num=max(1, n - 1) if n > 1 else 1, original_text=up)
        else:
            user = tpl.format(user_prompt=up, image_num=n)
    except Exception:
        user = tpl + "\n\n" + up
    return system, user, (task_code in JSON_MODE_TASKS)


def parse_json_rewritten(text):
    """Extrai 'rewritten_text' do JSON que as tarefas JSON-mode devolvem.
    Tolerante: acha o objeto JSON no meio de texto, com fallback pro texto cru."""
    if not text:
        return ""
    import json
    t = text.strip()
    # tenta bloco ```json ... ```
    m = re.search(r"\{.*\"rewritten_text\".*\}", t, re.DOTALL)
    if m:
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            v = obj.get("rewritten_text")
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            # regex do valor da chave (aspas escapadas simples)
            m2 = re.search(r'"rewritten_text"\s*:\s*"(.*)"\s*\}?', frag, re.DOTALL)
            if m2:
                return m2.group(1).replace('\\"', '"').replace("\\n", "\n").strip()
    return t
