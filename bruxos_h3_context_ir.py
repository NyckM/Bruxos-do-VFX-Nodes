# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — H3 Context-IR local (Qwen3)
===========================================
O QUE ESTE NODE RESOLVE
-----------------------
O MiniMax H3 completo tem TRES modulos:

    H3-Context-IR  ->  H3-Base (768p)  ->  H3-Regenerate-2K
      (FECHADO)         (o que roda           (FECHADO)
                         no ComfyUI)

O H3-Base NAO foi treinado pra receber prompt em prosa livre. Ele espera a
saida do Context-IR: uma REPRESENTACAO ESTRUTURADA em secoes fixas. A propria
MiniMax diz no model card:

    "H3-Context-IR is critical to the quality of the final output, so we
     strongly recommend incorporating it into your generation pipeline or
     following the 'Prompting Guidance' to build your own."

O Context-IR oficial e um servico hospedado (nao aberto). Este node e a
implementacao LOCAL dele, seguindo os dois guias oficiais:
    docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md    (ref2va)
    docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md   (t2va / i2va / fl2va / l2va)

REGRAS DO FORMATO QUE O NODE GARANTE
------------------------------------
  * ref2va  -> SEIS secoes, nesta ordem exata:
        subject_definitions / summary / retention_analysis /
        detailed_description / overall_soundscape / non_diegetic_music
  * t2va/i2va/fl2va/l2va -> linha de instrucao (quando ha imagem) + TRES campos:
        integrated_multimodal_description / overall_soundscape / non_diegetic_music
  * <Subject N> e a peca central, NAO <Picture N>. Imagem que so define um
    personagem/cenario/estilo e citada DENTRO de um <Subject>, sem virar
    entrada propria de <Picture>. <Picture N> so quando a imagem e literalmente
    um frame (primeiro, ultimo, keyframe, ancora de composicao).
  * <d>[Idioma] fala</d> -- o `<d>` e TOKEN ESPECIAL no tokenizer do H3.
    Fala fora de <d> nao e tratada como fala.
  * Falantes recebem IDs estaveis (S1), (S2)... na ordem dos eventos vocais.
  * As tags sao numeradas POR CATEGORIA, na ordem em que voce conectou os
    slots no node do H3 -- e exatamente isso que o campo 'referencias' faz.

LIMITES DO MODELO (validados aqui)
----------------------------------
    duracao ........ 4 a 15 s          |  saida 24 fps, audio estereo 32 kHz
    imagens ref .... <= 9              |  videos ref <= 3 (cada 2-15 s)
    audios ref ..... <= 3, NUNCA sozinho (exige imagem ou video junto)
    total arquivos . <= 12
    length ......... a template oficial prende a  length % 17 == 5
    width/height ... multiplos de 32 (VAE 16x + patchify 1x2x2 = 32x efetivo)
"""

import logging
import os
import re

try:
    import torch
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

try:
    from .bruxos_qwen3_enhancer import _MODELOS, _PRESETS, _carregar, _separa_think
except Exception:  # pragma: no cover - import solto (fora do pacote)
    from bruxos_qwen3_enhancer import _MODELOS, _PRESETS, _carregar, _separa_think

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/MiniMax H3"

# Os 11 idiomas com suporte ESTAVEL de dialogo, conforme o model card.
_LINGUAS = ["English", "Portuguese", "Spanish", "Chinese", "Japanese", "Korean",
            "French", "German", "Italian", "Russian", "Arabic"]

_TAREFAS = ["ref2va", "t2va", "i2va", "fl2va", "l2va"]

# ordem OBRIGATORIA das secoes, por tarefa
_SECOES_REF = ["subject_definitions", "summary", "retention_analysis",
               "detailed_description", "overall_soundscape", "non_diegetic_music"]
_SECOES_BASE = ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"]

_LIMITES = {"picture": 9, "video": 3, "audio": 3}
_TOTAL_MAX = 12

# como o usuario pode rotular cada linha do campo 'referencias'
_ALIAS = {
    "picture": ("imagem", "imagens", "image", "img", "picture", "pic", "foto", "frame"),
    "video":   ("video", "videos", "clipe", "clip", "filmagem"),
    "audio":   ("audio", "audios", "som", "sound", "voz", "voice", "musica", "music"),
}

# ---------------------------------------------------------------------------
# PAPEIS: o atalho que evita escrever o regulamento na mao
# ---------------------------------------------------------------------------
# Em vez de explicar em 'regras_extra' o que cada referencia faz, voce escreve
# o papel entre colchetes:
#       imagem [frame]: ...
#       video [camera]: ...
#       audio [voz]: ...
# Cada papel injeta no system prompt o paragrafo de regra CORRETO segundo o
# guia oficial -- inclusive as pegadinhas (ex.: um blocking do Blender nao pode
# contaminar o resultado com o cinza sem textura dele).
# Limite de tamanho do prompt do H3 (manual de uso, ~2026-08).
_MAX_CHARS = 7000

# CLAUSULA DE INVARIANCIA -- a peca que faltava.
# O manual do H3 exige que TODO prompt de edicao termine declarando o que NAO
# muda. Sem ela, o modelo se sente livre pra alterar/adicionar o que nao foi
# citado -- e o guia oficial reforca isso ao dizer que "conteudo novo adicionado
# nao conta como perda de fidelidade". A frase canonica cobre SEIS eixos:
# identidade, trajetoria de movimento, camera, composicao, ambiente e som.
_INVARIANCIA = (
    "CLOSING INVARIANCE CLAUSE (mandatory): end the description of this asset's "
    "shot with an explicit statement that everything not named as an edit stays "
    "put -- \"Apart from the changes described above, all other character "
    "identity, motion trajectory, camera path, composition, environment and "
    "sound remain unchanged from {TAG}.\" Name the six axes; do not shorten it "
    "to a vague \"everything else stays the same\"."
)

_PAPEIS = {
    # ---- imagens ----
    "frame": ("picture",
        "{TAG} is a CONCRETE FRAME ANCHOR. Give it its own <Picture N> entry and state which "
        "shot and which moment it anchors (first frame, keyframe, or last frame). Use natural "
        "phrasing in the body: \"the shot begins from {TAG}\" / \"the shot ends on {TAG}\"."),
    "sujeito": ("picture",
        "{TAG} DEFINES A CHARACTER. Do NOT give it its own <Picture N> entry -- cite it inside a "
        "<Subject N> definition that fixes identity, age, build, hair, clothing, colours and "
        "distinguishing features, so the same subject can be referred to across every shot."),
    "cenario": ("picture",
        "{TAG} DEFINES AN ENVIRONMENT. Do NOT give it its own <Picture N> entry -- cite it inside "
        "a <Subject N> definition covering materials, palette, surface treatment and lighting of "
        "the place. If a video reference supplies the geometry of this scene, keep the definition "
        "to MATERIAL AND LIGHT only: listing specific structures here makes them appear even where "
        "the source geometry has none."),
    # Troca de fundo / ambiente. Separado do 'cenario' porque aqui a imagem
    # SUBSTITUI o fundo de um video existente -- e o que decide se o resultado
    # parece integrado ou parece recorte colado nao e o fundo em si, e a
    # resposta do SUJEITO a luz nova (spill, halo, borda de cabelo, sombra de
    # contato). Vocabulario de compositing, nao de descricao.
    "fundo": ("picture",
        "{TAG} DEFINES THE REPLACEMENT ENVIRONMENT AND ITS LIGHT: atmosphere, colour, depth and "
        "the behaviour of its light source. Cite it inside a <Subject N> definition; it is not a "
        "frame. Take from it ONLY environment and lighting -- ignore any subject, pose or framing "
        "it happens to contain, because the subject comes from the video. "
        "INTEGRATION IS THE WHOLE JOB: state that the new environment's light acts as the actual "
        "light in the scene on the existing subject -- rim light on the silhouette, key shifted to "
        "the new colour temperature, shadow sides filled by the new ambient, contact shadows and "
        "reflections consistent with it, and any flicker or movement of that light carried onto the "
        "subject in time. Include parallax and background motion answering the camera move. "
        "PRESERVE AT THE EDGE: identity, hair strands and fabric edges, occlusions, gesture timing "
        "and scale are untouched. NEGATIVES, stated explicitly: no colour spill onto the subject "
        "beyond what the new light justifies, no halo or fringe at the matte edge, no altered "
        "performance, no change of framing or duration."),
    "estilo": ("picture",
        "{TAG} DEFINES THE VISUAL STYLE ONLY. Do NOT give it its own <Picture N> entry and do not "
        "reproduce its subject matter -- cite it inside a <Subject N> definition capturing medium, "
        "linework, palette, grain and rendering treatment."),
    # ---- videos ----
    "camera": ("video",
        "{TAG} CONTRIBUTES MOTION AND STAGING ONLY: camera movement, block layout, massing, "
        "framing and shot timing. Its OWN SURFACE APPEARANCE MUST NOT REACH THE TARGET VIDEO -- "
        "untextured grey clay, flat preview lighting, placeholder geometry, missing detail, "
        "viewport artefacts and any previz look are discarded. Every material, colour, texture "
        "and light in the target video comes from the image references instead. Each block or "
        "proxy volume in {TAG} resolves into a finished subject consistent with those images. "
        "If the proxy geometry is resolved into the final subjects, the task type is "
        "\"video editing + reference generation\" and {TAG} is fully_preserved for camera and "
        "layout; if only the camera path is borrowed, the task type is \"reference generation\" "
        "and {TAG} is weak_reference. "
        "NO INVENTED GEOMETRY: do not add buildings, structures or set dressing where {TAG} has "
        "no block. Empty space in {TAG} stays empty. "
        "FULL TIMELINE: describe the shot through to its LAST frame, stating what is on screen "
        "when the video ends."),
    "fonte": ("video",
        "{TAG} IS THE SOURCE VIDEO BEING EDITED. Task type must include \"video editing\", and "
        "the summary must begin, right after the task-type prefix, with \"The target video is an "
        "edited version of {TAG}.\" Preserve its framing, lighting and setting while applying the "
        "requested change. "
        "NO INVENTED GEOMETRY: the target video introduces no structure, object, building or set "
        "dressing that does not already exist in {TAG}. Where {TAG} shows empty space, bare "
        "ground, open sky or flat horizon, the target video KEEPS IT EMPTY -- it does not fill it "
        "in. Every solid form in the target must correspond one-to-one to a form present in "
        "{TAG} at that same moment. "
        "FULL TIMELINE: describe {TAG} through to its LAST frame. The final second must be "
        "specified as precisely as the first, stating what is on screen when the video ends."),
    # O caso do render de previz/blocking usado como FONTE DE EDICAO.
    # Precisa das duas metades: a fidelidade geometrica do 'fonte' E a rejeicao
    # de aparencia do 'camera'. Usar 'fonte' puro faz o modelo PRESERVAR a
    # iluminacao chapada e o material de argila do render -- e os blocos cinza
    # reaparecem no meio do video, onde a descricao afrouxa.
    "previz": ("video",
        "{TAG} IS A PREVIZ / BLOCKING RENDER USED AS THE EDIT SOURCE. Task type must include "
        "\"video editing\", and the summary must begin, right after the task-type prefix, with "
        "\"The target video is an edited version of {TAG}.\" "
        "GEOMETRY AND MOTION ARE LAW: camera path, framing, shot timing, and the position, "
        "height and footprint of every proxy volume are preserved exactly. Each block resolves "
        "one-to-one into a finished subject. Introduce no structure that has no block in {TAG}; "
        "where {TAG} shows empty ground or open sky, the target keeps it empty. "
        "APPEARANCE IS DISCARDED -- this OVERRIDES the usual rule of preserving a source's look: "
        "the untextured grey clay material, the flat uniform preview lighting, the missing "
        "surface detail, the placeholder shading and any viewport artefact of {TAG} MUST NOT "
        "appear anywhere in the target video, at ANY point in the timeline. Every material, "
        "colour, texture, shadow and light comes from the image references instead. "
        "NO REVERSION: state explicitly that the blocks remain fully resolved into finished "
        "surfaces from the first frame to the last -- at no moment does any part of the frame "
        "return to grey untextured proxy geometry. "
        "DESCRIBE ALL THREE PHASES: cover the BEGINNING, the MIDDLE and the END of the shot, so "
        "the resolved look is asserted across the whole timeline. An under-described middle is "
        "where the source's raw look leaks back in. "
        "DESCRIBE APPEARANCE, NEVER GEOMETRY -- this is the single most important rule here. "
        "Material, colour, texture, weathering, wear, shadow and light: describe those in as much "
        "detail as you want, because that is what gets transferred. But do NOT enumerate scene "
        "contents. Never name specific structures or landmarks (no \"a defensive wall\", \"towers "
        "in the distance\", \"a dense cluster of rooftops\", \"narrow streets\"), and never state "
        "which structure the camera reaches at a given moment. EVERY NOUN NAMING A PHYSICAL "
        "STRUCTURE IS AN ORDER TO RENDER IT: if {TAG} has no volume in that spot, the model builds "
        "one to satisfy the sentence. Refer to geometry only relationally -- \"each volume present "
        "in {TAG}\", \"whatever forms the source shows at that moment\", \"the volumes in frame\". "
        "COUNT AND PLACEMENT ARE BOUND TO THE SOURCE: state positively that the number, position, "
        "height, footprint and silhouette of the structures are exactly and only what {TAG} "
        "contains -- a positive binding works, a negative \"do not add\" alone does not."),
    "continuar": ("video",
        "{TAG} IS THE CLIP BEING CONTINUED. Task type must include \"video continuation\". The "
        "target video picks up from its final state; keep subject, lighting and setting "
        "continuous across the join."),
    # ---- audios ----
    "voz": ("audio",
        "{TAG} IS A VOICE-TIMBRE REFERENCE. Marker: reference. Do NOT carry its original words "
        "into the target video -- only timbre, pitch, delivery and pace. Bind it to the target "
        "speaker by reusing that speaker's ID: \"{TAG} is the voice-timbre reference for "
        "<Subject N> (Sx)\", never assigning a new ID here."),
    "trilha": ("audio",
        "{TAG} IS BACKGROUND MUSIC that is reused. Describe the relationship in "
        "non_diegetic_music, not in overall_soundscape. Marker: fully_copy if it is the complete "
        "final score, partially_copy if it is mixed under new dialogue."),
    "copiar": ("audio",
        "{TAG} IS COPIED AS-IS. Marker: fully_copy when it becomes the complete final audio "
        "track, partially_copy when only part of the timeline or some layers are reused."),
    "ambiente": ("audio",
        "{TAG} PROVIDES AMBIENCE / SOUND EFFECTS. Describe the relationship in overall_soundscape, "
        "not in non_diegetic_music."),
}

# Versao da clausula pro PREVIZ. A generica manda preservar "environment"
# do source -- e no previz o "environment" E O CINZA DE ARGILA. Sem esta
# excecao explicita, a clausula de invariancia BRIGA com a regra de descartar
# aparencia, e o modelo resolve o conflito trazendo os blocos de volta.
_INVARIANCIA_PREVIZ = (
    "CLOSING INVARIANCE CLAUSE (mandatory, with an explicit exception): end the "
    "description of this shot with -- \"Apart from the changes described above, "
    "the camera path, motion trajectory, composition and the position and scale "
    "of every element remain unchanged from {TAG}; the surface appearance, "
    "materials, lighting and colour of {TAG} are NOT preserved and are replaced "
    "entirely by those of the image references.\" "
    "The geometry axes are invariant, the appearance axes are NOT. Never write a "
    "blanket \"everything remains unmodified from {TAG}\" -- that sentence orders "
    "the grey proxy look back into the frame."
)

# A clausula de invariancia so faz sentido onde existe um SOURCE a preservar.
for _p in ("fonte", "continuar", "camera"):
    _cat, _txt = _PAPEIS[_p]
    _PAPEIS[_p] = (_cat, _txt + " " + _INVARIANCIA)
_PAPEIS["previz"] = (_PAPEIS["previz"][0], _PAPEIS["previz"][1] + " " + _INVARIANCIA_PREVIZ)

# sinonimos aceitos pra cada papel
_PAPEL_ALIAS = {
    "frame": ("frame", "keyframe", "primeiro", "ultimo", "first", "last", "ancora", "anchor"),
    "sujeito": ("sujeito", "subject", "personagem", "character", "pessoa", "char"),
    "cenario": ("cenario", "scene", "environment", "local", "lugar", "setting"),
    # NAO use 'ambiente' aqui: ja e alias do papel de AUDIO. Alias repetido em
    # dois papeis faz a resolucao depender da ordem do dicionario.
    "fundo": ("fundo", "background", "bg", "trocarfundo", "trocadefundo", "novofundo"),
    "estilo": ("estilo", "style", "look", "arte"),
    # 'blocking'/'previz'/'clay' sairam do 'camera' e foram pro papel 'previz',
    # que e o caso real: blocking usado como FONTE de edicao, nao so como
    # referencia de movimento.
    "camera": ("camera", "movimento", "motion", "layout", "ritmo"),
    "previz": ("previz", "blocking", "blocagem", "clay", "argila", "blockout", "greybox", "proxy"),
    "fonte": ("fonte", "source", "edicao", "editar", "edit", "base"),
    "continuar": ("continuar", "continuacao", "continuation", "extender", "continue"),
    "voz": ("voz", "voice", "timbre", "fala"),
    "trilha": ("trilha", "musica", "music", "bgm", "score"),
    "copiar": ("copiar", "copy", "igual", "asis", "manter"),
    "ambiente": ("ambiente", "ambience", "sfx", "efeitos", "room"),
}

# papel assumido quando nao da pra perguntar pro LLM (motor 'esqueleto')
_PAPEL_PADRAO = {"picture": "sujeito", "video": "camera", "audio": "voz"}

# Catalogo legivel -- e ISTO que aparece no dropdown do node de referencia e
# que o LLM recebe pra decidir sozinho quando voce nao especifica o papel.
_CATALOGO = [
    ("sujeito",   "imagem", "personagem, criatura ou objeto -> vira <Subject N>"),
    ("cenario",   "imagem", "ambiente/lugar quando NAO ha video de origem -> vira <Subject N>"),
    ("fundo",     "imagem", "TROCA o fundo de um video existente: leva luz, sombra, reflexo e borda junto (spill/halo)"),
    ("estilo",    "imagem", "so o look (traco, paleta, material) -> vira <Subject N>"),
    ("frame",     "imagem", "frame inicial / keyframe / ultimo frame -> vira <Picture N>"),
    ("previz",    "video",  "render de blocking/clay do Blender como FONTE: trava geometria e "
                            "camera, e JOGA FORA o cinza sem textura"),
    ("camera",    "video",  "so movimento de camera e ritmo; a geometria nao entra"),
    ("fonte",     "video",  "video REAL de origem que vai ser editado (preserva o look dele)"),
    ("continuar", "video",  "clipe que o video novo CONTINUA"),
    ("voz",       "audio",  "timbre de voz; nao copia as palavras"),
    ("trilha",    "audio",  "musica de fundo reaproveitada"),
    ("copiar",    "audio",  "audio copiado igualzinho"),
    ("ambiente",  "audio",  "ambiencia e efeitos sonoros"),
]
_TIPO_PT = {"picture": "imagem", "video": "video", "audio": "audio"}
_PT_TIPO = {"imagem": "picture", "video": "video", "audio": "audio"}

# opcoes do dropdown: "imagem · frame — frame inicial / keyframe ..."
_OPCOES_PAPEL = [f"{t} · {p} — {d}" for p, t, d in _CATALOGO]


def _catalogo_texto():
    return "\n".join(f"  {t:<6} [{p}] -> {d}" for p, t, d in _CATALOGO)


# ---------------------------------------------------------------------------
# descobre os LLMs que voce JA TEM em ComfyUI/models/LLM (e afins)
# ---------------------------------------------------------------------------
_NENHUM = "(nenhum - usar o widget 'modelo')"
_MAPA_LOCAIS = {}          # rotulo mostrado -> caminho absoluto


def _raizes_llm():
    """Onde procurar. Cobre o models/LLM padrao e a pasta do ComfyUI-QwenVL."""
    import os
    raizes = []
    try:
        import folder_paths
        try:
            if "LLM" in getattr(folder_paths, "folder_names_and_paths", {}):
                raizes += list(folder_paths.get_folder_paths("LLM"))
        except Exception:
            pass
        base = getattr(folder_paths, "models_dir", None)
        if base:
            for sub in ("LLM", "llm", "text_encoders", "prompt_generator"):
                raizes.append(os.path.join(base, sub))
    except Exception:
        pass
    vistas, saida = set(), []
    for r in raizes:
        r = os.path.abspath(r)
        if r not in vistas and os.path.isdir(r):
            vistas.add(r)
            saida.append(r)
    return saida


def _listar_modelos_locais(prof_max=3):
    """Varre as raizes e devolve os diretorios que PARECEM modelo HuggingFace
    (tem config.json). Marca os GGUF, que NAO servem: eu carrego via
    transformers e GGUF precisa de llama.cpp."""
    import os
    _MAPA_LOCAIS.clear()
    achados = []
    for raiz in _raizes_llm():
        base_nome = os.path.basename(raiz)
        for atual, dirs, arqs in os.walk(raiz):
            prof = atual[len(raiz):].count(os.sep)
            if prof >= prof_max:
                dirs[:] = []
                continue
            rel = os.path.relpath(atual, raiz)
            if rel == ".":
                continue
            if "config.json" in arqs and any(
                    a.endswith((".safetensors", ".bin")) for a in arqs) or \
               "model.safetensors.index.json" in arqs:
                rot = f"{base_nome}/{rel}".replace("\\", "/")
                _MAPA_LOCAIS[rot] = atual
                achados.append(rot)
                dirs[:] = []      # nao desce mais dentro de um modelo
            elif any(a.lower().endswith(".gguf") for a in arqs):
                rot = f"{base_nome}/{rel} [GGUF - nao serve]".replace("\\", "/")
                _MAPA_LOCAIS[rot] = None
                achados.append(rot)
                dirs[:] = []
    return [_NENHUM] + sorted(achados)


def _resolver_papel(txt, tipo):
    """'[camera]' -> 'camera'. Devolve (papel, aviso_ou_None)."""
    chave = re.sub(r"[^a-z]", "", _sem_acento((txt or "").lower()))
    if not chave:
        return None, None
    for papel, nomes in _PAPEL_ALIAS.items():
        if chave in nomes:
            if _PAPEIS[papel][0] != tipo:
                return None, (f"papel '[{txt}]' nao vale pra {tipo} (ele e de "
                              f"{_PAPEIS[papel][0]}); vou usar o padrao.")
            return papel, None
    return None, (f"papel '[{txt}]' desconhecido -- papeis validos: " +
                  ", ".join(sorted(_PAPEIS)) + ".")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sem_acento(s):
    tab = str.maketrans("áàâãäéèêëíìîïóòôõöúùûüçÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ",
                        "aaaaaeeeeiiiiooooouuuucAAAAAEEEEIIIIOOOOOUUUUC")
    return s.translate(tab)


def parse_referencias(texto):
    """Le o campo 'referencias' (uma por linha, NA ORDEM DOS SLOTS) e devolve
    (lista, avisos). Cada item: {'tipo','n','tag','desc'}.

    Formato aceito por linha:   tipo: descricao
        imagem: menino super-heroi de capa vermelha
        video: plano do onibus (fonte da edicao)
        audio: voz masculina calma

    A numeracao e POR CATEGORIA e segue a ordem das linhas -- que precisa ser a
    mesma ordem em que voce ligou ref_image_0, ref_image_1, ref_video_0, ...
    """
    itens, avisos = [], []
    contador = {"picture": 0, "video": 0, "audio": 0}
    rotulo = {"picture": "Picture", "video": "Video", "audio": "Audio"}

    for bruto in (texto or "").splitlines():
        linha = bruto.strip()
        if not linha or linha.startswith("#"):
            continue
        if ":" in linha:
            cab, desc = linha.split(":", 1)
        else:
            cab, desc = linha, ""

        # papel opcional entre colchetes:  "imagem [frame]: ..."
        papel_txt = ""
        m = re.search(r"\[([^\]]*)\]", cab)
        if m:
            papel_txt = m.group(1).strip()
            cab = cab[:m.start()] + cab[m.end():]

        chave = re.sub(r"[^a-z]", "", _sem_acento(cab.strip().lower()))

        tipo = None
        for t, nomes in _ALIAS.items():
            if chave in nomes:
                tipo = t
                break
        if tipo is None:
            avisos.append(f"linha ignorada (nao reconheci o tipo {cab.strip()!r}): {linha[:60]!r}. "
                          f"Use 'imagem:', 'video:' ou 'audio:' no comeco da linha.")
            continue

        papel, aviso_papel = _resolver_papel(papel_txt, tipo)
        if aviso_papel:
            avisos.append(aviso_papel)
        # sem papel escrito -> 'auto': o LLM decide lendo a sua frase.
        # (no motor 'esqueleto', que nao tem LLM, cai no _PAPEL_PADRAO)
        explicito = papel is not None
        if papel is None:
            papel = "auto"

        contador[tipo] += 1
        n = contador[tipo]
        itens.append({"tipo": tipo, "n": n, "tag": f"<{rotulo[tipo]} {n}>",
                      "desc": desc.strip(), "papel": papel, "papel_explicito": explicito})

    return itens, avisos


def validar_limites(itens, duracao_s):
    """Checa os limites do H3. Devolve lista de avisos (vazia = tudo certo)."""
    avisos = []
    cont = {"picture": 0, "video": 0, "audio": 0}
    for it in itens:
        cont[it["tipo"]] += 1

    for tipo, teto in _LIMITES.items():
        if cont[tipo] > teto:
            avisos.append(f"LIMITE ESTOURADO: {cont[tipo]} {tipo}(s), o H3 aceita no maximo {teto}.")

    total = sum(cont.values())
    if total > _TOTAL_MAX:
        avisos.append(f"LIMITE ESTOURADO: {total} arquivos de referencia no total, o maximo e {_TOTAL_MAX}.")

    if cont["audio"] > 0 and (cont["picture"] + cont["video"]) == 0:
        avisos.append("AUDIO SOZINHO NAO E PERMITIDO: o H3 exige pelo menos uma imagem ou um video "
                      "junto do audio. Adicione uma referencia visual.")

    d = float(duracao_s)
    if d < 4.0 or d > 15.0:
        avisos.append(f"DURACAO FORA DA FAIXA: {d:.2f}s. O H3 foi treinado pra 4-15s; "
                      f"fora disso a qualidade cai bastante.")

    for it in itens:
        if not it["desc"]:
            avisos.append(f"{it['tag']} esta sem descricao -- o Context-IR vai ter que inventar "
                          f"o que ela e. Escreva o que aparece nessa referencia.")
    return avisos


def _e_modelo_pequeno(nome):
    n = (nome or "").replace("_", "-").replace(".", "-")
    return any(t in n for t in ("-2B", "-1-7B", "-1B", "-0-5B", "-1.7B", "-3B"))


def _vram():
    """(livre_GB, total_GB) da GPU, ou None."""
    try:
        livre, total = torch.cuda.mem_get_info()
        return livre / 2**30, total / 2**30
    except Exception:
        return None


def _despejar(prompt):
    """Imprime o Context-IR inteiro no console, delimitado, pra voce ler sem
    precisar de node de preview nenhum."""
    print("\n" + "=" * 72, flush=True)
    print("  H3 CONTEXT-IR  (copie daqui se quiser editar a mao)", flush=True)
    print("=" * 72, flush=True)
    for linha in (prompt or "(vazio)").splitlines():
        print("  " + linha, flush=True)
    print("=" * 72 + "\n", flush=True)


def calcular_length(duracao_s, fps=24.0):
    """Frames validos pro H3. A template oficial do ComfyUI prende a
    length % 17 == 5 (5, 22, 39, 56, 73, 90, 107, 124, 141...).
    E a mesma conta do node Math Expression da template:
        base = max(5, round(dur*24));  base + (5 - base % 17) % 17
    """
    base = max(5, int(round(float(duracao_s) * float(fps))))
    return base + (5 - base % 17) % 17


# ---------------------------------------------------------------------------
# system prompts (a especificacao oficial, condensada)
# ---------------------------------------------------------------------------
_COMUM = """\
You are H3-Context-IR: the context-processing stage of the MiniMax H3 omni-modal
video+audio model. You convert a user's free-form creative intent into the exact
structured representation H3-Base was trained to consume.

HARD OUTPUT RULES
- Output ONLY the structured fields. No preamble, no markdown, no code fences,
  no bullet points, no commentary before or after.
- Write every section in English. The user may write in Portuguese or any other
  language: translate it. The ONLY things that keep their original language are
  dialogue/lyrics inside <d> and text that is visibly written in the scene.
- Every detail you write must be something visible or audible. Do not write
  abstract mood words, emotional function, or interpretation.

SHOTS AND TIME
- [Shot 1] opens the video and carries NO timestamp.
- Later shots: "[Shot N] At MM:SS.mmm, the camera cuts to ...". Cut times must
  strictly increase and stay inside the video duration.
- A cut must introduce new information (subject, space, state, viewpoint, time).
  If only distance or a slight angle changes, use camera motion instead.

CAMERA MOTION = motion type + amplitude + speed, written as natural English
inside the sentence (never stacked as labels at the end).
- Motion types: Zoom In/Out, Push In, Pull Out, Pan Left/Right, Truck Left/Right,
  Tilt Up/Down, Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot,
  Shake Slightly, Shake Strongly, POV, Roll Clockwise/Counterclockwise.
- Amplitude: "with small amplitude" / "with large amplitude" (omit if medium).
- Speed: "at slow speed" / "at fast speed" (omit if normal).
  Example: "The camera pushes in with small amplitude at slow speed toward her hands."

SPEAKERS AND DIALOGUE
- Anyone who speaks or sings gets a stable ID: (S1), (S2), ... assigned in the
  order of actual vocal events. Two people speaking together: (S1,S2).
  Characters who never vocalize get no ID.
- Dialogue and lyrics go inside <d>[Language] ...</d> and NOWHERE else. `<d>` is
  a special token in the H3 tokenizer; speech written outside it is not treated
  as speech.
- Put the speaker's identity, action and delivery OUTSIDE <d>. Inside <d> put
  only the language tag and the literal spoken words, verbatim, with the user's
  original wording and punctuation preserved.
- On first appearance of a speaker, establish a stable identity: type, age,
  gender, on/off-screen, pitch, timbre, rate, accent.
- Voiceover uses the exact phrase "says in an off-screen voiceover", and right
  after the <d> block you must state that the on-screen character's lips remain
  closed.
- Dialogue crossing a cut uses <scenetrans> at both connecting points, plus an
  explicit statement that the audio continues across the cut. Speech truncated
  by the end of the video uses <cutoff>.

ON-SCREEN TEXT
- Any text actually visible on screen (sign, banner, subtitle, overlay, neon)
  goes in English double quotation marks, verbatim, untranslated.

AUDIO SECTIONS
- overall_soundscape: 1-4 sentences, one paragraph, ambience + physical action
  sounds + non-verbal human sounds across the whole video. Never repeat dialogue
  or lyrics here. Use "N/A" only if the user explicitly wants total silence.
- non_diegetic_music: 1-3 sentences on music only the audience hears. Describe
  instrumentation, tempo, rhythm and dynamic change. No mood adjectives, no
  explanation of emotional function. Use "N/A" if there is none. Music the
  characters can hear is diegetic and belongs in the description body instead.
"""

_SPEC_REF = """\
TASK: ref2va (full-reference mode).

Output EXACTLY these six sections, in this order, each starting at the beginning
of a line with the field name followed by a colon:

subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

--- subject_definitions ---
One item per line. Four label types:
  <Subject N>  reusable VISIBLE content: a person, animal, object, environment,
               costume, prop, interface, effect, style, action or pose. THIS IS
               THE CENTRAL LABEL. One subject may draw on several assets, and one
               asset may provide several subjects.
  <Picture N>  ONLY when the image is literally a frame: first frame, last frame,
               keyframe, edited keyframe, or a composition/storyboard anchor.
               If an image merely defines a character, scene, costume or style,
               do NOT give it its own line -- cite it inside the <Subject N>
               definition that uses it.
  <Video N>    ONLY whole-video relationships: the source video being edited, the
               clip being continued, or a reference for camera movement, cuts,
               rhythm or temporal structure. A person or object taken out of a
               reference video is still a <Subject N>.
  <Audio N>    a standalone audio asset, or the synchronized track of a reference
               video when that track is actually used. A reference video does not
               create an <Audio N> just because the file contains sound.

<Video N> and <Audio N> are numbered independently; the same source file may be
<Video 1> and <Audio 2>.
When an <Audio N> maps to a target speaker, reuse that speaker's global ID:
  "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)."
Never assign a new (Sx) here -- only reuse the one from detailed_description.

Examples:
  <Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue
  cardigan, and a thin silver necklace.
  <Subject 2> is the woman whose appearance comes from <Picture 1> and whose
  walking motion comes from <Video 1>.
  <Picture 2> is the first frame of [Shot 1], showing a woman beside a window.
  <Video 1> is the source video for the target video edit.

--- summary ---
One short English paragraph, opening with a square-bracketed task-type prefix.
Task types (combine with " + ", never repeat one):
  keyframe completion  an image is a concrete frame anchor of the target video
  reference generation an asset guides character/scene/style/action/camera/
                       storyboard WITHOUT being a concrete frame or an edited source
  video editing        an existing source video is directly modified
  video continuation   new content continues or extends an existing source video
  audio reuse          the same audio signal is reused, fully or partly
  audio reference      only style, timbre, content, texture, beat or continuity
                       of the audio is referenced, not the signal itself
A reference video that only provides camera movement, cuts or rhythm is
"reference generation", NOT "video editing".
For video-editing tasks the summary must begin, right after the prefix, with:
  "The target video is an edited version of <Video 1>."
Use only labels already defined above; introduce no new ones.

--- retention_analysis ---
One line per reference label, reusing the meaning fixed in subject_definitions.
Visible content (<Subject>, <Picture>, <Video>) uses exactly one of these fixed
markers: fully_preserved | partially_preserved | attribute_transfer | weak_reference
Audio (<Audio>) uses exactly one of: fully_copy | partially_copy | reference | weak_reference
Format:
  <Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...
  <Picture 2> ([Shot 1] first frame): fully_preserved - ...
  <Video 1> (cut and pacing structure): weak_reference - ...
  <Audio 1>: fully_copy - ...
Never write (Sx) in this section. New actions or backgrounds added to the target
video are NOT losses of reference fidelity.

--- detailed_description ---
Establish the overall style in one or two English sentences BEFORE [Shot 1]
(not inside it). Then describe the video shot by shot in playback order.
Insert each reference label at its first appearance and wherever its role
applies. For concrete frame anchors use natural phrasing: "the shot begins from
<Picture 1>", "the shot ends on <Picture 3>".
When a referenced subject physically speaks, keep BOTH labels: "<Subject 2> (S1)".
Target length 350-500 English words for generation tasks. Dialogue-dense content
prioritizes fitting the full spoken timeline over hitting a word count.
For each shot make explicit: composition, subject appearance and position,
environment and lighting, actions and state changes, camera movement, current
sound, and where referenced content actually takes effect. Never reduce this to
a plot summary or a list of reference relationships.
"""

_SPEC_BASE = """\
TASK: {tarefa}.

{instrucao_bloco}

Then output EXACTLY these three fields, in this order, each starting at the
beginning of a line:

integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...

integrated_multimodal_description carries the whole timeline: visual style and
initial composition stated at the start of [Shot 1], then subject appearance and
position, scene and key props, actions and reactions, cuts, speakers, dialogue,
singing and diegetic sound. Common style words: Cinematic, live-action,
2D-animated, 3D CG, claymation, watercolor, vintage film.
Example opening: "[Shot 1] Live-action, cinematic, a medium-wide shot frames ..."
"""

_INSTR = {
    "t2va": "",
    "i2va": ("The FIRST LINE of your output must be exactly this, followed by one blank line:\n"
             "For the target video, at 0.00 seconds into the target video, <Picture 1> "
             "(from [Shot 1]) is fully referenced.\n"
             "<Picture 1> is the actual first frame at 0.00 s and belongs to [Shot 1]. "
             "Establish its style, subjects, composition and scene anchors first, then "
             "describe the next action. Structure: first-frame anchor -> action onset -> "
             "continuous development -> result or reaction."),
    "fl2va": ("The FIRST LINE of your output must be exactly this, followed by one blank line:\n"
              "How the reference pictures align with the target video — Picture 1 (from Shot 1) "
              "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot {N}) "
              "aligns with the {S}-second mark of the target video.\n"
              "Picture 1 is the opening and Picture 2 is the ending. Do NOT restate two static "
              "image descriptions -- supply the MOTION PATH between them. Prefer a SINGLE shot so "
              "the model can interpolate continuously. Structure: first-frame state -> observable "
              "intermediate changes -> progressively narrowing differences -> last-frame state."),
    "l2va": ("The FIRST LINE of your output must be exactly this, followed by one blank line:\n"
             "How the reference pictures align with the target video — <Picture 1> (from [Shot {N}]) "
             "aligns with the {S}-second mark of the target video.\n"
             "<Picture 1> is the FINAL frame and belongs to the last shot, not Shot 1. Infer a "
             "plausible earlier state, then converge onto the image. Structure: plausible preceding "
             "state -> explicit action and transition path -> gradual convergence -> last-frame landing."),
}


def montar_sistema(tarefa, itens, duracao_s, idioma, estilo, regras_extra):
    if tarefa == "ref2va":
        corpo = _SPEC_REF
    else:
        bloco = _INSTR.get(tarefa, "")
        if bloco:
            bloco = bloco.replace("{N}", "1").replace("{S}", f"{float(duracao_s):.2f}")
        corpo = _SPEC_BASE.format(tarefa=tarefa, instrucao_bloco=bloco or
                                  "This task has no image-alignment instruction line. "
                                  "Begin directly with the three core fields.")

    partes = [_COMUM, corpo]

    ctx = [f"Target video duration: {float(duracao_s):.2f} seconds at 24 fps.",
           f"Default language for dialogue: {idioma}."]
    if (estilo or "").strip():
        ctx.append(f"Requested visual style: {estilo.strip()}")

    if itens:
        linhas = ["",
                  "REFERENCE ASSETS ACTUALLY CONNECTED (labels are already assigned by slot "
                  "order -- use these exact labels and invent no others):"]
        for it in itens:
            linhas.append(f"  {it['tag']} = {it['desc'] or '(no description given)'}")

        # regra especifica do PAPEL de cada referencia: e isto que dispensa o
        # usuario de escrever o regulamento na mao em 'regras_extra'.
        fixos = [i for i in itens if i["papel"] != "auto"]
        autos = [i for i in itens if i["papel"] == "auto"]

        if fixos:
            linhas.append("")
            linhas.append("ROLE OF EACH REFERENCE (follow these rules exactly):")
            for it in fixos:
                linhas.append("  - " + _PAPEIS[it["papel"]][1].replace("{TAG}", it["tag"]))

        if autos:
            # o usuario NAO declarou o papel -> o modelo infere da frase dele.
            linhas.append("")
            linhas.append("ROLE NOT DECLARED for: " + ", ".join(i["tag"] for i in autos) + ".")
            linhas.append("DECIDE the role of each one from the USER INTENT sentence, then apply "
                          "the matching rule below. Phrases like \"the photo is the first frame\", "
                          "\"a foto e o frame inicial\", \"the video carries the camera / o video e "
                          "a ancora de movimento / blocking\" tell you the role directly. Choose "
                          "exactly one role per asset:")
            for papel, (cat, regra) in _PAPEIS.items():
                if any(_TIPO_PT[i["tipo"]] == _TIPO_PT.get(cat, cat) or i["tipo"] == cat
                       for i in autos):
                    linhas.append(f"  * role \"{papel}\" ({cat}): "
                                  + regra.replace("{TAG}", "the asset"))
            linhas.append("State the chosen role implicitly through the structure you write "
                          "(a frame anchor gets its own <Picture N>; a character/environment/style "
                          "goes inside a <Subject N>; a motion reference keeps its <Video N> and "
                          "contributes no surface appearance).")

        linhas.append("Remember: an image that only defines a character, scene, costume or style "
                      "must be cited inside a <Subject N> definition instead of getting its own "
                      "<Picture N> entry.")
        ctx.extend(linhas)
    elif tarefa == "ref2va":
        ctx.append("WARNING: no reference assets were listed. Do not invent reference labels.")

    partes.append("CONTEXT FOR THIS REQUEST\n" + "\n".join(ctx))

    if (regras_extra or "").strip():
        partes.append("EXTRA USER RULES (these override defaults):\n" + regras_extra.strip())

    return "\n\n".join(partes)


# ---------------------------------------------------------------------------
# MOTOR 'esqueleto': monta o Context-IR SEM LLM NENHUM
# ---------------------------------------------------------------------------
# Serve pra quando nao da pra baixar/carregar o Qwen (disco cheio, VRAM ocupada,
# maquina offline). O formato sai 100% correto -- labels numeradas, marcadores
# validos, <d> com tag de idioma, secoes na ordem certa. O que ele NAO faz e
# escrever prosa cinematografica: as descricoes sao as SUAS, coladas nos lugares
# certos. Voce edita o texto, a estrutura ja esta garantida.
_MARCADOR_PADRAO = {
    "frame": "fully_preserved", "sujeito": "fully_preserved",
    "cenario": "fully_preserved", "fundo": "attribute_transfer", "estilo": "weak_reference",
    "camera": "fully_preserved", "fonte": "fully_preserved",
    "previz": "fully_preserved", "continuar": "fully_preserved",
    "voz": "reference", "trilha": "fully_copy",
    "copiar": "fully_copy", "ambiente": "partially_copy",
}
_TAREFA_DO_PAPEL = {
    "frame": "keyframe completion", "sujeito": "reference generation",
    "cenario": "reference generation", "fundo": "reference generation", "estilo": "reference generation",
    "camera": "reference generation", "fonte": "video editing",
    "previz": "video editing", "continuar": "video continuation",
    "voz": "audio reference", "trilha": "audio reuse",
    "copiar": "audio reuse", "ambiente": "audio reuse",
}


def montar_esqueleto(tarefa, itens, ideia, estilo, dialogo, idioma, duracao_s):
    """Context-IR estruturalmente valido, montado por regra -- zero LLM."""
    ideia = (ideia or "").strip() or "TODO: describe what happens in the video."
    estilo_txt = (estilo or "").strip() or "TODO: state the visual style in one sentence."
    falas = [l.strip() for l in (dialogo or "").splitlines() if l.strip()]

    if tarefa != "ref2va":
        corpo = [f"integrated_multimodal_description: [Shot 1] {estilo_txt} {ideia}", "",
                 "overall_soundscape: TODO: ambience and physical action sounds.", "",
                 "non_diegetic_music: N/A"]
        return "\n".join(corpo)

    # ---- subject_definitions ----
    defs, subj_n, mapa_subj = [], 0, {}
    for it in itens:
        # sem LLM nao da pra inferir: cai no padrao da categoria
        p = it["papel"]
        if p == "auto":
            p = _PAPEL_PADRAO[it["tipo"]]
            it = dict(it, papel=p)
        d = it["desc"] or "TODO: describe this reference"
        if p == "frame":
            defs.append(f"{it['tag']} is a frame anchor of [Shot 1], showing {d}.")
        elif p in ("sujeito", "cenario", "estilo"):
            subj_n += 1
            mapa_subj[it["tag"]] = f"<Subject {subj_n}>"
            que = {"sujeito": "the character", "cenario": "the environment",
                   "estilo": "the visual style"}[p]
            defs.append(f"<Subject {subj_n}> is {que} in {it['tag']}: {d}.")
        elif p == "camera":
            defs.append(f"{it['tag']} is the motion and staging reference for the target video "
                        f"(camera movement, layout and shot timing): {d}. Its own untextured "
                        f"preview appearance is not carried into the target video.")
        elif p == "fonte":
            defs.append(f"{it['tag']} is the source video for the target video edit: {d}.")
        elif p == "continuar":
            defs.append(f"{it['tag']} is the clip the target video continues from: {d}.")
        elif p == "voz":
            defs.append(f"{it['tag']} is the voice-timbre reference for the speaker (S1): {d}.")
        else:
            defs.append(f"{it['tag']} is an audio reference: {d}.")

    # ---- summary ----
    tipos, vistos = [], set()
    for it in itens:
        t = _TAREFA_DO_PAPEL[it["papel"] if it["papel"] != "auto" else _PAPEL_PADRAO[it["tipo"]]]
        if t not in vistos:
            vistos.add(t)
            tipos.append(t)
    prefixo = " + ".join(tipos) if tipos else "reference generation"
    linha_edit = ""
    for it in itens:
        if it["papel"] == "fonte":
            linha_edit = f"The target video is an edited version of {it['tag']}. "
            break
    resumo = f"[{prefixo}] {linha_edit}{ideia}"

    # ---- retention_analysis ----
    ret = []
    for it in itens:
        alvo = mapa_subj.get(it["tag"], it["tag"])
        mk = _MARCADOR_PADRAO[it["papel"] if it["papel"] != "auto" else _PAPEL_PADRAO[it["tipo"]]]
        onde = "" if it["tipo"] == "audio" else " (appears in [Shot 1])"
        motivo = it["desc"] or "the referenced characteristics"
        ret.append(f"{alvo}{onde}: {mk} - {motivo}.")

    # ---- detailed_description ----
    corpo = [estilo_txt, f"[Shot 1] {ideia}"]
    if itens:
        usados = ", ".join(mapa_subj.get(i["tag"], i["tag"]) for i in itens)
        corpo.append(f"The shot uses {usados} as established above.")
    for i, fala in enumerate(falas):
        corpo.append(f"The speaker (S{1 if i == 0 else 1}) says, <d>[{idioma}] {fala}</d>")

    partes = [
        "subject_definitions:\n" + ("\n".join(defs) if defs else "TODO: no references listed."),
        "summary:\n" + resumo,
        "retention_analysis:\n" + ("\n".join(ret) if ret else "TODO: no references listed."),
        "detailed_description:\n" + "\n".join(corpo),
        "overall_soundscape:\nTODO: ambience and physical action sounds across the whole video.",
        "non_diegetic_music:\nN/A",
    ]
    return "\n\n".join(partes)


# ---------------------------------------------------------------------------
# limpeza e verificacao da saida
# ---------------------------------------------------------------------------
def limpar_saida(txt):
    """Tira cercas de markdown e preambulo antes da primeira secao."""
    t = (txt or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t).strip()
    cabecas = _SECOES_REF + _SECOES_BASE + ["For the target video", "How the reference pictures"]
    melhor = None
    for c in cabecas:
        m = re.search(r"^\s*" + re.escape(c), t, flags=re.M)
        if m and (melhor is None or m.start() < melhor):
            melhor = m.start()
    if melhor:
        t = t[melhor:].strip()
    return t


def detectar_loop(txt, jan=12, min_rep=3):
    """Acha repeticao degenerada: a mesma sequencia de `jan` palavras aparecendo
    `min_rep`+ vezes. Modelo pequeno faz isso e enche o max_tokens com lixo."""
    pal = re.findall(r"\w+", (txt or "").lower())
    if len(pal) < jan * min_rep:
        return None
    cont = {}
    for i in range(len(pal) - jan + 1):
        ch = " ".join(pal[i:i + jan])
        cont[ch] = cont.get(ch, 0) + 1
    if not cont:
        return None
    frase, n = max(cont.items(), key=lambda kv: kv[1])
    return (frase, n) if n >= min_rep else None


def conferir_labels(txt, itens):
    """O modelo so pode citar labels que EXISTEM no grafo. Se ele inventar um
    <Audio 1> sem audio conectado, o H3 recebe uma instrucao que aponta pro
    vazio -- e nao reclama."""
    avisos = []
    disp = {"Picture": 0, "Video": 0, "Audio": 0}
    for it in itens:
        disp[{"picture": "Picture", "video": "Video", "audio": "Audio"}[it["tipo"]]] += 1
    for rot in ("Picture", "Video", "Audio"):
        usados = {int(n) for n in re.findall(r"<" + rot + r"\s+(\d+)>", txt)}
        extras = sorted(n for n in usados if n > disp[rot])
        if extras:
            quais = ", ".join(f"<{rot} {n}>" for n in extras)
            avisos.append(
                f"LABEL INVENTADA: o texto cita {quais}, mas so existem {disp[rot]} "
                f"referencia(s) desse tipo conectadas. O H3 vai receber uma instrucao "
                f"apontando pro vazio. Apague essas mencoes, ou conecte o arquivo e "
                f"acrescente a linha em 'referencias'.")
    return avisos


def conferir_formato(txt, tarefa):
    """Confere se as secoes obrigatorias existem e estao na ordem certa."""
    avisos = []
    esperadas = _SECOES_REF if tarefa == "ref2va" else _SECOES_BASE
    posicoes = []
    for s in esperadas:
        m = re.search(r"^\s*" + re.escape(s) + r"\s*:", txt, flags=re.M)
        if m is None:
            avisos.append(f"FORMATO: a secao obrigatoria '{s}:' nao apareceu na saida.")
        else:
            posicoes.append((s, m.start()))
    ordenadas = [s for s, _ in sorted(posicoes, key=lambda x: x[1])]
    presentes = [s for s in esperadas if s in ordenadas]
    if ordenadas != presentes:
        avisos.append(f"FORMATO: as secoes sairam fora de ordem ({' < '.join(ordenadas)}). "
                      f"A ordem correta e: {' < '.join(esperadas)}.")

    if tarefa == "ref2va":
        if not re.search(r"<Subject\s+\d+>", txt):
            avisos.append("FORMATO: nenhum <Subject N> foi definido. Em ref2va o <Subject> e a "
                          "peca central -- so <Picture> costuma indicar que o modelo tratou as "
                          "imagens como frames em vez de personagens.")
        # Procura os marcadores SO dentro do bloco retention_analysis: a palavra
        # "reference" tambem aparece em "[reference generation]" no summary, e
        # buscar no texto inteiro daria falso negativo.
        bloco = re.search(r"^\s*retention_analysis\s*:(.*?)(?=^\s*detailed_description\s*:|\Z)",
                          txt, flags=re.M | re.S)
        trecho = bloco.group(1) if bloco else ""
        marcadores = ("fully_preserved", "partially_preserved", "attribute_transfer",
                      "weak_reference", "fully_copy", "partially_copy", "reference")
        if trecho and not any(mk in trecho for mk in marcadores):
            avisos.append("FORMATO: retention_analysis saiu sem nenhum marcador fixo "
                          "(fully_preserved / partially_copy / reference / ...).")

    if len(txt) > _MAX_CHARS:
        avisos.append(f"TAMANHO: o prompt tem {len(txt)} caracteres e o H3 aceita no maximo "
                      f"{_MAX_CHARS}. O excesso e cortado silenciosamente -- e o que se perde e "
                      f"o FIM (overall_soundscape e non_diegetic_music). Reduza o "
                      f"detailed_description ou baixe o max_tokens.")

    abre, fecha = txt.count("<d>"), txt.count("</d>")
    if abre != fecha:
        avisos.append(f"FORMATO: <d> abertos ({abre}) != fechados ({fecha}). O H3 trata <d> como "
                      f"token especial; desbalanceado, a fala nao e reconhecida.")
    for m in re.finditer(r"<d>(.{0,40})", txt, flags=re.S):
        if not re.match(r"\s*\[[A-Za-z]", m.group(1)):
            avisos.append("FORMATO: achei um <d> sem a tag de idioma logo em seguida "
                          "(o certo e '<d>[English] ...</d>').")
            break
    return avisos


# ---------------------------------------------------------------------------
# NODE
# ---------------------------------------------------------------------------
class BruxosH3ContextIR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ideia": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "Sua ideia, PODE SER EM PORTUGUES e do jeito que vier na cabeca. Ex.: 'o menino de capa "
                    "vermelha no telhado fala olhando pra camera, ai corta pro mecha gigante rugindo'.\n"
                    "O node transforma isso na representacao estruturada que o H3-Base espera."}),
                "tarefa": (_TAREFAS, {"default": "ref2va", "tooltip":
                    "Qual node do H3 vai receber o prompt:\n"
                    "ref2va -> MiniMax H3 Reference to Video (imagens/videos/audios de referencia). SEIS secoes.\n"
                    "t2va   -> so texto, sem imagem.\n"
                    "i2va   -> primeira imagem = primeiro frame.\n"
                    "fl2va  -> duas imagens = primeiro e ultimo frame.\n"
                    "l2va   -> uma imagem = ultimo frame."}),
                "modelo": (_MODELOS, {"default": _MODELOS[0], "tooltip":
                    "Qual Qwen3 escreve o Context-IR. O 4B ja da conta; 8B/14B melhoram a coerencia entre as "
                    "secoes. Os '-VL' enxergam imagem -- ligue 'imagem_ref' pra ele DESCREVER a referencia em "
                    "vez de confiar so no seu texto."}),
                "modo_qwen3": (["direto", "pensar", "personalizado"], {"default": "direto", "tooltip":
                    "Valores oficiais do Qwen3. 'direto' (temp 0.7/top_p 0.80) basta e e rapido. "
                    "'pensar' (temp 0.6/top_p 0.95) ajuda em roteiro com varios cortes e falas, mas exige "
                    "subir bastante o max_tokens porque o raciocinio consome antes da resposta."}),
            },
            "optional": {
                "referencias": ("STRING", {"multiline": True, "default":
                    "imagem: descreva aqui o que aparece na primeira imagem\n"
                    "imagem: descreva a segunda\n"
                    "# video: plano de origem, se houver\n"
                    "# audio: voz ou trilha, se houver\n", "tooltip":
                    "UMA REFERENCIA POR LINHA, NA MESMA ORDEM EM QUE VOCE LIGOU OS SLOTS no node do H3 "
                    "(ref_image_0, ref_image_1, ..., ref_video_0, ..., ref_audio_0).\n"
                    "Formato:  tipo: descricao      -- tipos aceitos: imagem / video / audio\n"
                    "A numeracao <Picture 1>, <Picture 2>, <Video 1>, <Audio 1> sai daqui automaticamente, "
                    "por categoria. Linhas comecando com # sao ignoradas.\n"
                    "Descreva de verdade o que aparece: e isso que vira o <Subject N>."}),
                "duracao_s": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip":
                    "Duracao alvo em segundos. O H3 foi treinado pra 4-15s -- fora disso o node avisa. "
                    "A saida 'length' ja converte pra frames validos."}),
                "idioma_dialogo": (_LINGUAS, {"default": "English", "tooltip":
                    "Idioma que vai dentro das tags <d>[Idioma] fala</d>. Sao os 11 com suporte estavel. "
                    "So a FALA fica nesse idioma; o resto do prompt e sempre em ingles."}),
                "dialogo": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "[opcional] As falas EXATAS, uma por linha. Vao ser preservadas ao pe da letra dentro de "
                    "<d></d>, sem traducao e sem reescrita. Deixe vazio se nao ha fala."}),
                "estilo": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "[opcional] Estilo visual em uma frase. Ex.: 'comic-book ink, linha grossa, paleta vermelho "
                    "e azul-preto, cidade noturna'."}),
                "imagem_ref": ("IMAGE", {"tooltip":
                    "[so nos modelos -VL] Um frame pro Qwen OLHAR. Deixa as definicoes de <Subject> fieis ao que "
                    "existe na imagem em vez de genericas."}),
                "regras_extra": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "[opcional] Regras suas somadas ao final do system prompt (tem prioridade). Ex.: 'plano unico, "
                    "sem cortes', 'sem trilha, so som ambiente'."}),
                "max_tokens": ("INT", {"default": 1600, "min": 256, "max": 8192, "step": 64, "tooltip":
                    "Teto de tokens da resposta. O Context-IR de ref2va e LONGO (seis secoes, "
                    "detailed_description sozinho pede 350-500 palavras): 1600 e o minimo confortavel. "
                    "No modo 'pensar' suba pra 3000+."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip":
                    "[so em 'personalizado'] Oficial: 0.7 no direto, 0.6 no pensar."}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                    "[so em 'personalizado'] Oficial: 0.80 no direto, 0.95 no pensar."}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1, "tooltip":
                    "[so em 'personalizado'] Oficial: 20 nos dois modos."}),
                "pensar_personalizado": ("BOOLEAN", {"default": False, "tooltip":
                    "[so em 'personalizado'] Liga o enable_thinking do chat template."}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "manter_carregado": ("BOOLEAN", {"default": False, "tooltip":
                    "Mantem o Qwen3 na VRAM entre execucoes. DESLIGADO por padrao aqui: o H3 e um modelo de 33B "
                    "e voce vai precisar de cada MB da 4090 pro sampler."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # -----------------------------------------------------------
                # APPEND-ONLY: widgets NOVOS vao SEMPRE no FIM desta lista.
                # O ComfyUI casa os `widgets_values` salvos por ORDEM, nao por
                # nome. Inserir um widget no meio desloca TODOS os valores dos
                # workflows ja salvos. Acrescente ABAIXO desta nota.
                # -----------------------------------------------------------
                "motor": (["qwen3", "esqueleto"], {"default": "qwen3", "tooltip":
                    "qwen3     -> um LLM escreve o Context-IR inteiro (melhor qualidade, precisa do modelo "
                    "baixado em disco).\n"
                    "esqueleto -> monta o Context-IR POR REGRA, sem LLM nenhum: zero download, zero VRAM, "
                    "instantaneo. A ESTRUTURA sai 100% correta (labels numeradas, marcadores validos, <d> com "
                    "idioma, secoes na ordem); a PROSA e a sua, colada nos lugares certos, com TODO: onde falta. "
                    "Use quando o disco estiver cheio, a VRAM ocupada ou a maquina offline."}),
                "modelo_local": ("STRING", {"default": "", "tooltip":
                    "[motor qwen3] Caminho de uma pasta de modelo que voce JA TEM no disco, tipo "
                    "D:\\\\modelos\\\\Qwen3-4B. Se preenchido, IGNORA o widget 'modelo' e nao baixa nada.\n"
                    "Serve pra apontar pra outro drive quando o C: esta sem espaco, ou pra reaproveitar um "
                    "Qwen que outro custom node ja baixou."}),
                "modelo_em_disco": (_listar_modelos_locais(), {"default": _NENHUM, "tooltip":
                    "Os modelos que voce JA TEM em ComfyUI/models/LLM (e nas pastas onde o ComfyUI-QwenVL "
                    "baixa). Escolher aqui NAO baixa nada -- e o jeito mais facil de rodar com o disco cheio.\n"
                    "Tem prioridade sobre o 'modelo' e sobre o 'modelo_local'.\n"
                    "Os marcados [GGUF - nao serve] existem no disco mas NAO carregam aqui: eu uso "
                    "transformers e GGUF precisa de llama.cpp.\n"
                    "A lista e montada quando o ComfyUI inicia -- se voce baixar um modelo novo, reinicie "
                    "(ou de refresh no navegador) pra ele aparecer."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "length", "avisos", "raciocinio", "info")
    OUTPUT_TOOLTIPS = (
        "O Context-IR estruturado -> ligue no 'prompt' do MiniMax H3 Reference to Video (ou do t2v/i2v).",
        "Frames validos pro H3 (respeita length %% 17 == 5) -> ligue no 'length' do node do H3.",
        "Avisos de limite e de formato. VAZIO = tudo certo. LEIA se nao estiver vazio.",
        "O bloco <think> do modo pensar (vazio no modo direto).",
        "Modelo, parametros e contagem de referencias.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "H3 Context-IR local (Bruxos): converte sua ideia solta em portugues na REPRESENTACAO ESTRUTURADA que o "
        "MiniMax H3-Base espera -- as seis secoes oficiais do modo referencia (subject_definitions, summary, "
        "retention_analysis, detailed_description, overall_soundscape, non_diegetic_music) ou os tres campos do "
        "modo base. O Context-IR oficial da MiniMax e fechado; esta e a implementacao local dele seguindo os dois "
        "guias de prompt publicados. Numera as tags <Picture N>/<Video N>/<Audio N> sozinho pela ordem dos slots, "
        "valida os limites do modelo (<=9 imagens, <=3 videos, <=3 audios nunca sozinhos, <=12 arquivos, 4-15s) e "
        "confere o formato da saida antes de devolver."
    )

    def run(self, ideia, tarefa, modelo, modo_qwen3="direto", referencias="", duracao_s=5.0,
            idioma_dialogo="English", dialogo="", estilo="", imagem_ref=None, regras_extra="",
            max_tokens=1600, temperature=0.7, top_p=0.8, top_k=20, pensar_personalizado=False,
            dtype="bf16", device="auto", manter_carregado=False, seed=0,
            motor="qwen3", modelo_local="", modelo_em_disco=_NENHUM):
        if not _OK:
            raise RuntimeError("[H3 Context-IR] torch indisponivel.")
        texto_ideia = (ideia or "").strip()
        if not texto_ideia:
            raise ValueError("[H3 Context-IR] 'ideia' esta vazia -- escreva o que voce quer que aconteca no video.")

        # ---- referencias e validacao -------------------------------------
        itens, avisos = parse_referencias(referencias)
        avisos += validar_limites(itens, duracao_s)
        length = calcular_length(duracao_s)

        if tarefa == "ref2va" and not itens:
            avisos.append("MODO ref2va SEM REFERENCIA: voce escolheu o modo de referencia mas nao listou "
                          "nenhuma. Preencha 'referencias' ou troque a tarefa pra t2va.")
        n_pic = sum(1 for i in itens if i["tipo"] == "picture")
        esperado = {"i2va": 1, "l2va": 1, "fl2va": 2}.get(tarefa)
        if esperado is not None and n_pic != esperado:
            avisos.append(f"MODO {tarefa} espera exatamente {esperado} imagem(ns), mas voce listou {n_pic}.")

        # =================================================================
        # MOTOR 'esqueleto': sem LLM, sem download, sem VRAM
        # =================================================================
        if motor == "esqueleto":
            prompt = montar_esqueleto(tarefa, itens, texto_ideia, estilo,
                                      dialogo, idioma_dialogo, duracao_s)
            avisos += conferir_formato(prompt, tarefa)
            if "TODO:" in prompt:
                avisos.append("ESQUELETO: sobrou 'TODO:' no texto -- esses trechos voce precisa "
                              "escrever a mao (ou preencher 'estilo' e as descricoes das "
                              "referencias e rodar de novo).")
            avisos.append("MOTOR 'esqueleto': a ESTRUTURA esta correta, mas a prosa e a sua. "
                          "O H3 espera INGLES em tudo, menos dentro de <d>. Se voce escreveu a "
                          "ideia em portugues, traduza antes de mandar pro sampler.")
            cont = {t: sum(1 for i in itens if i["tipo"] == t) for t in ("picture", "video", "audio")}
            info = (f"{tarefa} | motor=esqueleto (sem LLM) | refs: {cont['picture']} img / "
                    f"{cont['video']} vid / {cont['audio']} aud | "
                    f"{float(duracao_s):.2f}s -> length={length} | {len(prompt)}ch")
            print(f"[H3 Context-IR] {info}", flush=True)
            for it in itens:
                print(f"[H3 Context-IR]   {it['tag']} papel={it['papel']}"
                      f"{'' if it['papel_explicito'] else ' (padrao)'}", flush=True)
            for a in avisos:
                print(f"[H3 Context-IR]   - {a}", flush=True)
            _despejar(prompt)
            return (prompt, int(length), "\n".join(f"- {a}" for a in avisos), "", info)

        # ---- monta a conversa --------------------------------------------
        sistema = montar_sistema(tarefa, itens, duracao_s, idioma_dialogo, estilo, regras_extra)

        pedido = [f"USER INTENT (translate to English, keep the intent exactly):\n{texto_ideia}"]
        falas = [l.strip() for l in (dialogo or "").splitlines() if l.strip()]
        if falas:
            pedido.append("EXACT DIALOGUE LINES — reproduce these verbatim inside <d>[" +
                          idioma_dialogo + "] ...</d>, word for word, without translating, "
                          "rewriting, shortening or reordering them:\n" +
                          "\n".join(f"  {i+1}. {l}" for i, l in enumerate(falas)))
        pedido.append(f"The target video is {float(duracao_s):.2f} seconds long. "
                      f"Fit the whole timeline inside that duration.")
        instr = "\n\n".join(pedido)

        # ---- parametros de amostragem -------------------------------------
        if modo_qwen3 in _PRESETS:
            p = dict(_PRESETS[modo_qwen3])
            origem = f"preset oficial '{modo_qwen3}'"
        else:
            p = dict(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k),
                     min_p=0.0, enable_thinking=bool(pensar_personalizado))
            origem = "personalizado"
            if p["enable_thinking"] and p["temperature"] <= 0.0:
                p["temperature"] = 0.6
                print("[H3 Context-IR] AVISO: temperatura 0 com 'pensar' faz o Qwen3 entrar em LOOP. "
                      "Corrigi pra 0.6 (valor oficial).", flush=True)

        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        # prioridade: dropdown do disco > caminho digitado > repo do HuggingFace
        alvo_modelo, origem_mdl = modelo, "repo HuggingFace (pode baixar)"
        if (modelo_local or "").strip():
            alvo_modelo, origem_mdl = modelo_local.strip(), "modelo_local"
        if modelo_em_disco and modelo_em_disco != _NENHUM:
            if not _MAPA_LOCAIS:
                _listar_modelos_locais()
            caminho = _MAPA_LOCAIS.get(modelo_em_disco)
            if caminho is None:
                raise ValueError(
                    f"[H3 Context-IR] '{modelo_em_disco}' nao serve.\n"
                    f"Se esta marcado [GGUF - nao serve]: eu carrego via transformers e GGUF "
                    f"precisa de llama.cpp. Escolha uma pasta com config.json + .safetensors, "
                    f"ou use o motor 'esqueleto'.\n"
                    f"Se a pasta sumiu, reinicie o ComfyUI pra atualizar a lista."
                )
            alvo_modelo, origem_mdl = caminho, "modelo_em_disco"
        print(f"[H3 Context-IR] modelo: {alvo_modelo}  ({origem_mdl})", flush=True)
        import time
        _t0 = time.time()
        try:
            mdl, tok, proc = _carregar(alvo_modelo, dtype, dev)
        except Exception as e:
            if "espa" in str(e).lower() or "disk space" in str(e).lower() or "error 112" in str(e):
                raise RuntimeError(
                    f"[H3 Context-IR] SEM ESPACO EM DISCO pra baixar '{alvo_modelo}'.\n"
                    f"O cache do HuggingFace fica em C:\\Users\\<voce>\\.cache\\huggingface e o "
                    f"modelo pede varios GB.\n"
                    f"Tres saidas, da mais rapida pra mais definitiva:\n"
                    f"  1) Troque o widget 'motor' pra 'esqueleto': ele monta o Context-IR por "
                    f"regra, sem LLM e sem baixar nada.\n"
                    f"  2) Se voce ja tem um Qwen em outro drive, ponha o caminho da pasta no "
                    f"widget 'modelo_local' -- ele ignora o download.\n"
                    f"  3) Mande o cache pra outro disco antes de subir o ComfyUI:\n"
                    f"       set HF_HOME=D:\\hf_cache\n"
                    f"     (ou a variavel de ambiente do Windows, pra valer sempre)\n"
                    f"Erro original: {e}"
                ) from e
            raise
        _t_carga = time.time() - _t0

        # ONDE o modelo realmente ficou? Se caiu na CPU (ou foi offloadado pela
        # accelerate porque a VRAM estava ocupada pelo H3), a geracao fica
        # ordens de grandeza mais lenta -- e essa e a causa nº1 de "demorou 6 min".
        _devs = set()
        try:
            for _p in mdl.parameters():
                _devs.add(str(_p.device))
                if len(_devs) > 3:
                    break
        except Exception:
            pass
        _na_cpu = any(d.startswith("cpu") for d in _devs)
        _vm = _vram()
        _vram_txt = f" | VRAM livre: {_vm[0]:.1f}/{_vm[1]:.1f} GB" if _vm else ""
        print(f"[H3 Context-IR] carga do modelo: {_t_carga:.1f}s | devices: "
              f"{sorted(_devs) or '?'}{_vram_txt}", flush=True)
        if _vm and _vm[0] < 3.0 and not _na_cpu:
            print(f"[H3 Context-IR] *** VRAM APERTADA: so {_vm[0]:.1f} GB livres. ***\n"
                  f"[H3 Context-IR] No Windows, quando a VRAM acaba o driver NAO da erro -- ele\n"
                  f"[H3 Context-IR] pagina pra memoria do sistema em silencio. O modelo continua\n"
                  f"[H3 Context-IR] reportando 'cuda:0' e roda 10-20x mais devagar. E a causa mais\n"
                  f"[H3 Context-IR] comum de tok/s baixo com tudo aparentemente certo.\n"
                  f"[H3 Context-IR] Conserto: rode o Context-IR ANTES do H3 carregar (VAEs + text\n"
                  f"[H3 Context-IR] encoder do H3 ja ocupam ~30 GB staged), ou num prompt separado.",
                  flush=True)
        if _na_cpu:
            print("[H3 Context-IR] *** ATENCAO: parte (ou todo) o modelo esta na CPU. ***\n"
                  "[H3 Context-IR] Na CPU a geracao fica 20-50x mais lenta. Causas comuns:\n"
                  "[H3 Context-IR]   - a VRAM ja estava tomada pelo H3/VAE -> rode o Context-IR "
                  "ANTES de carregar o H3, ou num prompt separado;\n"
                  "[H3 Context-IR]   - modelo grande demais pra 4090 -> use um 4B em vez de 8B/14B;\n"
                  "[H3 Context-IR]   - device='cpu' no widget.", flush=True)

        ehVL = proc is not None and imagem_ref is not None

        if ehVL:
            from PIL import Image
            fr = imagem_ref[0]
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
            texto = alvo.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            print("[H3 Context-IR] aviso: seu 'transformers' nao aceita enable_thinking -- "
                  "usando o template padrao.", flush=True)

        inputs = (proc(text=[texto], images=[pil], return_tensors="pt")
                  if ehVL else tok([texto], return_tensors="pt")).to(dev)

        if seed:
            torch.manual_seed(int(seed))

        # ANTI-LOOP. Modelo pequeno (2B/1.7B) gerando texto longo e estruturado
        # entra em loop: repete a mesma frase ate estourar o max_tokens. Os
        # presets oficiais do Qwen3 nao trazem penalidade nenhuma, entao a gente
        # acrescenta -- valores conservadores, pra nao estragar as repeticoes
        # LEGITIMAS do formato ("fully_preserved", "[Shot N] At ...").
        #   repetition_penalty 1.05 -> desencoraja de leve token ja usado
        #   no_repeat_ngram_size 24 -> proibe repetir 24 tokens SEGUIDOS iguais,
        #     que nunca acontece por acaso: e assinatura de loop.
        ger = dict(max_new_tokens=int(max_tokens), do_sample=p["temperature"] > 0,
                   temperature=p["temperature"], top_p=p["top_p"], top_k=int(p["top_k"]),
                   min_p=p["min_p"],
                   # barato: roda na GPU junto com os outros logits processors
                   repetition_penalty=1.05)
        # CARO: o NoRepeatNGramLogitsProcessor do transformers roda na CPU e
        # forca sincronizacao GPU->CPU a CADA token, derrubando o tok/s. So
        # vale nos modelos pequenos, que sao os que realmente entram em loop.
        # De 4B pra cima o repetition_penalty sozinho ja segura.
        if _e_modelo_pequeno(alvo_modelo):
            ger["no_repeat_ngram_size"] = 24
            print("[H3 Context-IR] modelo pequeno -> ligando no_repeat_ngram_size=24 "
                  "(anti-loop). Custa velocidade, mas evita o texto repetido.", flush=True)
        _t1 = time.time()
        with torch.inference_mode():
            try:
                out = mdl.generate(**inputs, **ger)
            except TypeError:
                ger.pop("min_p", None)
                out = mdl.generate(**inputs, **ger)
        _t_ger = time.time() - _t1

        corte = out[:, inputs["input_ids"].shape[1]:]
        _n_novos = int(corte.shape[1])
        _tps = _n_novos / max(_t_ger, 1e-6)
        print(f"[H3 Context-IR] geracao: {_t_ger:.1f}s | {_n_novos} tokens | "
              f"{_tps:.1f} tok/s | entrada: {int(inputs['input_ids'].shape[1])} tokens",
              flush=True)
        if _tps < 5.0:
            print(f"[H3 Context-IR] *** {_tps:.1f} tok/s e MUITO lento pra uma 4090 "
                  f"(o esperado e 30-80 tok/s num 4B). Veja o aviso de device acima. ***",
                  flush=True)
        dec = (proc if ehVL else tok).batch_decode(corte, skip_special_tokens=True)[0]
        bruto, pensamento = _separa_think(dec)
        prompt = limpar_saida(bruto)

        # ---- confere o formato da saida ----------------------------------
        avisos += conferir_formato(prompt, tarefa)
        avisos += conferir_labels(prompt, itens)

        _loop = detectar_loop(prompt)
        if _loop:
            _frase, _n = _loop
            avisos.append(
                f"LOOP DO MODELO: a mesma frase se repete {_n}x -- \"{_frase[:70]}...\". "
                f"O texto e lixo, nao use. Causa quase sempre e MODELO PEQUENO demais pra "
                f"gerar as 6 secoes: um 2B ou 1.7B nao aguenta. Use um 4B ou maior. "
                f"Se so tem o pequeno, tente modo_qwen3='pensar' ou baixe a temperature "
                f"pra 0.5.")

        _mini = any(t in alvo_modelo.replace("_", "-") for t in ("-2B", "-1.7B", "-1B", "-0.5B"))
        if _mini and tarefa == "ref2va":
            avisos.append(
                f"MODELO PEQUENO: '{os.path.basename(alvo_modelo.rstrip(chr(92) + '/'))}' tem "
                f"poucos bilhoes de parametros. O Context-IR de ref2va sao 6 secoes com "
                f"marcadores fixos e 350-500 palavras coerentes entre si -- e a tarefa mais "
                f"pesada que este node pede. Abaixo de 4B a taxa de loop e de secao faltando "
                f"e alta. Recomendo Qwen3-4B ou Qwen3-VL-4B pra cima.")
        if not prompt:
            avisos.append("SAIDA VAZIA: no modo 'pensar' isso quase sempre e max_tokens curto -- "
                          "o raciocinio consumiu tudo antes da resposta. Suba pra 3000+.")
        elif len(corte[0]) >= int(max_tokens) - 2:
            avisos.append(f"SAIDA TRUNCADA: bateu no teto de {max_tokens} tokens, entao as ultimas "
                          f"secoes provavelmente ficaram pela metade. Suba o max_tokens.")

        if not manter_carregado:
            if _t_carga > 20.0:
                print(f"[H3 Context-IR] DICA: a carga do modelo levou {_t_carga:.0f}s e "
                      f"'manter_carregado' esta DESLIGADO -- entao esse tempo se repete a CADA "
                      f"execucao. Se voce vai iterar no prompt, LIGUE 'manter_carregado' e so "
                      f"desligue quando for rodar o sampler do H3.", flush=True)
            from .bruxos_qwen3_enhancer import _CACHE
            _CACHE.update({"chave": None, "modelo": None, "tok": None, "proc": None})
            try:
                del mdl, tok, proc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        cont = {t: sum(1 for i in itens if i["tipo"] == t) for t in ("picture", "video", "audio")}
        # BUG CORRIGIDO: aqui imprimia `modelo` (o widget) em vez de `alvo_modelo`
        # (o que realmente rodou). Quem escolhia pelo dropdown 'modelo_em_disco'
        # via o nome errado no log.
        info = (f"{tarefa} | {alvo_modelo} | {origem} | temp={p['temperature']} top_p={p['top_p']} "
                f"thinking={p['enable_thinking']}{' | +imagem(VL)' if ehVL else ''} | "
                f"refs: {cont['picture']} img / {cont['video']} vid / {cont['audio']} aud | "
                f"{float(duracao_s):.2f}s -> length={length} | {len(prompt)}ch")
        texto_avisos = "\n".join(f"- {a}" for a in avisos)

        print(f"[H3 Context-IR] {info}", flush=True)
        for it in itens:
            print(f"[H3 Context-IR]   {it['tag']} papel={it['papel']}"
                  f"{'' if it['papel_explicito'] else ' (padrao)'} = {it['desc'][:60]!r}", flush=True)
        _despejar(prompt)
        if avisos:
            print("[H3 Context-IR] AVISOS:", flush=True)
            for a in avisos:
                print(f"[H3 Context-IR]   - {a}", flush=True)
        else:
            print("[H3 Context-IR] sem avisos: formato e limites OK.", flush=True)

        return (prompt, int(length), texto_avisos, pensamento, info)


# ---------------------------------------------------------------------------
# NODE COMPANHEIRO: monta a linha de referencia com o papel em DROPDOWN
# ---------------------------------------------------------------------------
class BruxosH3Referencia:
    """Uma referencia = um node. Encadeie na ordem dos slots do H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "papel": (_OPCOES_PAPEL, {"default": _OPCOES_PAPEL[0], "tooltip":
                    "O QUE esta referencia faz no video. A lista inteira esta aqui -- nao precisa "
                    "decorar colchete nenhum.\n\n"
                    "IMAGEM:\n"
                    "  sujeito  personagem/criatura/objeto -> vira <Subject N>\n"
                    "  cenario  ambiente, lugar, arquitetura -> vira <Subject N>\n"
                    "  estilo   so o look (traco, paleta) -> vira <Subject N>\n"
                    "  frame    frame inicial / keyframe -> vira <Picture N>\n\n"
                    "VIDEO:\n"
                    "  camera     movimento + blocking. O look do previz (cinza sem textura, luz "
                    "chapada) NAO e herdado -- material e cor vem das imagens.\n"
                    "  fonte      video de origem que vai ser EDITADO\n"
                    "  continuar  clipe que o video novo CONTINUA\n\n"
                    "AUDIO:\n"
                    "  voz       timbre de voz; NAO copia as palavras\n"
                    "  trilha    musica de fundo reaproveitada\n"
                    "  copiar    audio copiado igualzinho\n"
                    "  ambiente  ambiencia e efeitos sonoros"}),
                "descricao": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "O que aparece nesta referencia. Quanto mais concreto, mais estavel o video: "
                    "e daqui que sai a definicao de <Subject N>.\n"
                    "Pode deixar vazio SE voce ligar a imagem no 'imagem_ref' do Context-IR e usar "
                    "um modelo -VL -- ai ele olha e descreve sozinho."}),
            },
            "optional": {
                "anterior": ("STRING", {"forceInput": True, "tooltip":
                    "Encadeie: ligue aqui a saida do node da referencia ANTERIOR. A ordem da "
                    "corrente tem que ser a mesma ordem dos slots no node do H3 "
                    "(ref_image_0, ref_image_1, ..., ref_video_0, ..., ref_audio_0)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("referencias",)
    OUTPUT_TOOLTIPS = ("Ligue no proximo node de referencia, ou no campo 'referencias' do Context-IR.",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "H3 Referencia (Bruxos): monta UMA linha do campo 'referencias' do Context-IR, com o papel "
        "escolhido num dropdown em vez de digitado entre colchetes. Encadeie um node por referencia, "
        "na mesma ordem dos slots do node do H3. Papeis disponiveis:\n" + _catalogo_texto()
    )

    def run(self, papel, descricao="", anterior=""):
        # "imagem · frame — ..." -> ("imagem", "frame")
        cabeca = papel.split("—")[0]
        tipo_pt, nome = [x.strip() for x in cabeca.split("·")[:2]]
        linha = f"{tipo_pt} [{nome}]: {(descricao or '').strip()}"
        antes = (anterior or "").rstrip()
        return ((antes + "\n" + linha) if antes else linha,)


NODE_CLASS_MAPPINGS = {
    "BruxosH3ContextIR": BruxosH3ContextIR,
    "BruxosH3Referencia": BruxosH3Referencia,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3ContextIR": "MiniMax H3 · Context-IR local (Bruxos)",
    "BruxosH3Referencia": "MiniMax H3 · Referência (Bruxos)",
}
