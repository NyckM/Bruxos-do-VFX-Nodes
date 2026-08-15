# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — MiniMax H3: Prompt Rapido (macros @ e #)
========================================================
POR QUE ESTE NODE EXISTE
    O BruxosH3ContextIR escreve o Context-IR completo com um Qwen3 local. E
    bom, mas carrega um modelo e leva minutos. Na maior parte das vezes voce
    nao precisa de um LLM -- precisa parar de decorar as tags do H3.

    Este node e substituicao de texto pura: nenhum modelo, nenhum download,
    tempo de execucao na casa dos milissegundos.

A IDEIA E DO ComfyUI-MiniMaxH3-Easy (nkxx188, MIT)
    O '@' para referencias e o '#' para bloco vieram de la:
    https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy
    Aqui a execucao e outra -- la e um editor visual em JavaScript com popup
    de preview; aqui e macro em texto puro, e o '#' expande os PAPEIS do
    pacote Bruxos (previz, camera, fundo, ...) em vez de abrir bloco de fala.
    Credito da ideia e dele.

CUIDADO QUE ESTE ARQUIVO TOMA
    Os textos de _PAPEIS no bruxos_h3_context_ir.py sao ORDENS PARA O LLM que
    escreve o prompt ("Do NOT give it its own <Picture N> entry", "the summary
    must begin with..."). Colar aquilo aqui produziria um prompt que manda o
    H3 formatar um documento em vez de gerar video.
    Por isso _BLOCOS abaixo e um catalogo SEPARADO, com o mesmo conhecimento
    porem redigido como CONTEUDO que o H3 le. Se voce mexer num, releia o
    outro -- eles descrevem a mesma coisa para leitores diferentes.

SINTAXE
    @img  @img2  @1        -> <Picture 1>, <Picture 2>, <Picture 1>
    @video  @v2            -> <Video 1>, <Video 2>
    @audio  @a3            -> <Audio 1>, <Audio 3>
    #previz  #camera2      -> o paragrafo do papel, ja amarrado na tag certa
    #d:texto#              -> <d>texto</d>   (fala; o H3 usa <d> para dialogo)

    Escreve-se '@@' e '##' para um @ ou # literal.
"""

import logging
import re

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/MiniMax H3"

try:  # aliases: fonte unica de verdade, para os dois nodes aceitarem as mesmas palavras
    from .bruxos_h3_context_ir import _PAPEL_ALIAS as _ALIAS_IMPORTADO
except Exception:  # pragma: no cover
    _ALIAS_IMPORTADO = None

_TAG = {"picture": "Picture", "video": "Video", "audio": "Audio"}


# ---------------------------------------------------------------------------
# CATALOGO VOLTADO AO H3
#   Cada entrada: papel -> (tipo, paragrafo). {T} vira a tag (<Video 1>, ...).
#   Regras que segui ao redigir:
#     - descreve APARENCIA e INTENCAO, nunca geometria nova (substantivo que
#       nomeia estrutura e ordem de renderizar aquilo);
#     - toda restricao vira frase afirmativa quando da, porque negativa solta
#       ("sem predios") tende a invocar o que nega;
#     - fecha com o que preservar, que e onde o H3 costuma derivar.
# ---------------------------------------------------------------------------
_BLOCOS = {
    # ------------------------------- imagens -------------------------------
    "frame": ("picture",
        "The shot begins from {T}: its framing, subject placement and lighting are the "
        "first frame of the target video, and the motion continues from there."),
    "sujeito": ("picture",
        "The character in {T} keeps the identity shown there -- facial features and "
        "proportions, age, build, hair, clothing and colours stay the same in every shot. "
        # Retrato fechado usado como referencia de identidade EMPURRA o video pro
        # close: o modelo le o enquadramento da referencia como intencao de plano.
        # Por isso a segunda frase separa explicitamente identidade de enquadramento.
        "{T} is an identity reference only: its own crop, shot size and framing carry no "
        "instruction about the target video, which keeps the framing it already has. A "
        "tight portrait in {T} is there so the face can be read in detail, not because "
        "the camera should move closer, and {T} is not a frame of the target video."),
    "cenario": ("picture",
        "The environment takes its materials, palette, surface treatment and lighting "
        "from {T}."),
    "fundo": ("picture",
        "The environment of {T} replaces the original background, and its light acts as "
        "the real light in the scene: the key shifts to its colour temperature, the "
        "silhouette picks up its rim light, shadow sides fill with its ambient, and "
        "contact shadows and reflections follow it. Any flicker or movement of that light "
        "shows on the subject over time. The subject's identity, hair strands, fabric "
        "edges, gesture timing, scale and framing are untouched, and the matte edge stays "
        "clean."),
    "estilo": ("picture",
        "The visual treatment of {T} -- medium, linework, palette, grain and rendering -- "
        "applies to the whole video. Its subject matter is not reproduced."),
    # ------------------------------- videos --------------------------------
    "camera": ("video",
        "The camera movement, framing, staging and shot timing follow {T}. Every material, "
        "colour, texture and light comes from the image references instead, so the flat "
        "preview look of {T} does not appear. Each block or proxy volume in {T} resolves "
        "into a finished subject consistent with those images, and the areas {T} leaves "
        "open stay open."),
    "previz": ("video",
        "The camera path, staging and layout follow {T} exactly, and every solid form in "
        "the target corresponds to a form present in {T} at that same moment -- where {T} "
        "is open ground, sky or empty space, the target stays open as well. The appearance "
        "of {T} does not carry over: its untextured grey surfaces, flat preview lighting "
        "and placeholder detail are replaced by the materials, colour and lighting of the "
        "image references. The description runs through to the last frame of {T}."),
    "fonte": ("video",
        "The target video is an edited version of {T}. Its framing, lighting and setting "
        "are preserved while the described change is applied, and every solid form in the "
        "target matches a form already present in {T} -- open ground, sky and empty space "
        "in {T} stay that way. The description runs through to the last frame."),
    "continuar": ("video",
        "The target video continues directly from the end of {T}, picking up the same "
        "subjects, lighting, camera motion and pace at the moment {T} stops."),
    # O caso que o 'fonte' NAO cobre: preservar movimento e identidade mas TROCAR
    # luz e ambiente. O 'fonte' diz "lighting ... preserved", que e o oposto.
    # Aqui a fidelidade e de MOVIMENTO e IDENTIDADE; luz e cenario ficam livres.
    # A parte de integracao e a mesma do papel 'fundo' -- o dificil nunca e o
    # ambiente novo, e o sujeito responder a luz nova.
    "relight": ("video",
        "The target video is a relit version of {T}. The camera path, framing and "
        "composition are identical to {T} frame for frame, and the subject's motion, "
        "pose, gesture timing, expression and gaze are carried over exactly. What "
        "changes is the light and the environment. The new light acts as the actual "
        "light in the scene on the existing subject: the key shifts to its colour "
        "temperature, the silhouette picks up its rim light, shadow sides fill with its "
        "ambient, and contact shadows and reflections follow it. Any flicker or movement "
        "of that light shows on the subject over time, in sync with the source. "
        "Identity, hair strands, fabric edges, occlusions and scale are untouched, and "
        "the matte edge stays clean. "
        "It is one continuous take from the first frame to the last: no cut anywhere, "
        "and the shot size and framing at the final frame are the same as at the start. "
        "The new lighting holds all the way through that final frame."),
    # Mapa de profundidade. O erro classico e o modelo tratar o cinza como
    # APARENCIA e devolver um video cinzento -- por isso a primeira frase diz
    # o que o video E, antes de dizer o que fazer com ele.
    "depth": ("video",
        "{T} is a depth map, not footage: its brightness encodes distance from camera, "
        "near being bright and far being dark. Read it for spatial layout only -- the "
        "volume and placement of the subject, the depth of the environment behind it, "
        "the occlusion order between them, and how all of that shifts as the camera "
        "moves. The greyscale itself is not a look and does not appear in the target "
        "video: every material, colour, texture and light comes from the image "
        "references and the description instead. Solid form in the target follows the "
        "solid form in {T}, for both the subject and the background. "
        "It is one continuous take from the first frame to the last: no cut anywhere, "
        "and the framing follows {T} through to its final frame."),
    # Style transfer: trava TUDO da estrutura, libera so o tratamento visual.
    # Diferente do 'relight', que libera luz e ambiente mas mantem o meio.
    "st": ("video",
        "The target video is a restyled version of {T}. Camera path, framing, "
        "composition, subject identity, motion, gesture timing and expression are "
        "carried over from {T} exactly, frame for frame, and nothing is added to or "
        "removed from the scene. What changes is the rendering treatment alone -- "
        "medium, linework, palette, shading, texture and grain -- applied evenly across "
        "subject and environment so both belong to the same image. Edges, occlusions "
        "and scale stay where they are. "
        # SEM ESTA FRASE o H3 troca de plano no fim. Ele perde a redea nos
        # ultimos frames e tem tendencia forte a CORTAR -- inventa um close que
        # nao existe na fonte. O 'fonte' e o 'previz' ja tinham a clausula de
        # timeline; eu escrevi 'st', 'relight' e 'depth' sem ela, e o corte
        # apareceu na primeira geracao longa. Nao remova.
        "It is one continuous take from the first frame to the last: no cut anywhere, "
        "and the shot size, lens and framing at the final frame are the same as at the "
        "start. The described treatment holds all the way through that final frame."),
    # ------------------------------- audios --------------------------------
    "voz": ("audio",
        "The speaking voice matches the timbre, age and accent heard in {T}. The words "
        "spoken are the ones written in this prompt, not the words in {T}."),
    "trilha": ("audio",
        "The background music of {T} plays under the shot, at its original tempo and mood."),
    "copiar": ("audio",
        "The audio of {T} is reproduced as it is."),
    "ambiente": ("audio",
        "The ambience and sound effects of {T} fill the scene."),
    # ------------------------- macros que nao sao papel --------------------
    "invariancia": (None,
        "Apart from the changes described above, character identity, motion trajectory, "
        "camera path, composition, environment and sound all remain unchanged."),
    "semlegenda": (None,
        "The frame carries no subtitles, captions, watermark, logo or on-screen text."),
}

# quantos itens de cada tipo o H3 aceita (README oficial)
_LIMITE = {"picture": 9, "video": 3, "audio": 3}

# ---- aliases -> papel canonico --------------------------------------------
_ALIAS = {}
if _ALIAS_IMPORTADO:
    for canon, palavras in _ALIAS_IMPORTADO.items():
        if canon in _BLOCOS:
            for p in palavras:
                _ALIAS[p] = canon
for canon in _BLOCOS:
    _ALIAS.setdefault(canon, canon)
# atalhos proprios deste node
_ALIAS.update({
    "inv": "invariancia", "invar": "invariancia", "fecho": "invariancia",
    "limpo": "semlegenda", "semtexto": "semlegenda", "notext": "semlegenda",
    # papeis proprios deste node (nao existem no _PAPEL_ALIAS do Context-IR)
    "reluz": "relight", "relighting": "relight", "iluminar": "relight",
    "profundidade": "depth", "depthmap": "depth", "zdepth": "depth",
    "styletransfer": "st", "restyle": "st", "estilizar": "st",
    "transferencia": "st", "restilizar": "st",
})
# Erros de digitacao que valem aceitar em silencio. Nao e preguica: uma macro
# nao reconhecida vai INTEIRA pro prompt como texto literal, entao o custo de
# um typo aqui e alto e invisivel -- o H3 recebe "#previs" e tenta interpretar.
# Nao adicione qualquer palavra: so grafias que a pessoa escreve sem perceber.
_ALIAS.update({
    "previs": "previz",        # 'previs' com s e o mais comum em portugues
    "prevys": "previz",
    "previsualizacao": "previz",
    "cenairo": "cenario",
    "sujeto": "sujeito",
    "camara": "camera",
})
# NAO ponha alias com '_' ou '-' aqui: o _RE_MACRO so casa [A-Za-zÀ-ſ]+ e
# digito, entao "#sem_legenda" quebraria em "#sem" + "_legenda" e o alias
# nunca seria alcancado.

# ---- aliases de tipo para o '@' -------------------------------------------
_TIPO_ALIAS = {
    "picture": ("img", "imagem", "image", "picture", "pic", "foto", "p", "i"),
    "video":   ("video", "vid", "v", "clipe", "clip"),
    "audio":   ("audio", "som", "sound", "a", "aud"),
}
_ALIAS_TIPO = {a: t for t, lista in _TIPO_ALIAS.items() for a in lista}

_RE_DIALOGO = re.compile(r"#d\s*:\s*(.+?)\s*#", re.S | re.I)
# A macro aceita DUAS formas de dizer em qual referencia ela pega:
#     #camera2            numero colado
#     #camera @video2     apontando logo depois, na MESMA linha
# A segunda existe porque e como as pessoas escrevem naturalmente. Uso
# [ \t]+ e nao \s+ de proposito: com \s+ um '#invariancia' no fim de uma linha
# engoliria o '@img1' da linha seguinte, que e outra frase.
_RE_MACRO = re.compile(
    r"#([A-Za-zÀ-ſ]+)(\d*)(?:[ \t]+@([A-Za-zÀ-ſ]*)(\d*))?(?:[ \t]*\((\d{1,2})\))?")

# ---------------------------------------------------------------------------
# FORCA POR REFERENCIA:  #sujeito2(9)
#
# O encoder do H3 e o Qwen3-VL lendo PROSA -- ver o docstring de
# comfy/text_encoders/minimax.py: "raw prose (no special tokens)". Nao existe
# parser de peso como o (palavra:1.2) do CLIP. Escrever "(9)" no prompt seria
# lido como o numero nove.
#
# Entao o numero e sintaxe NOSSA e vira FRASE. E por isso que sao faixas e nao
# dez niveis: linguagem nao tem dez degraus distintos de "siga isto". Fingir
# precisao de 1 a 10 seria mentira de interface.
#
# A frase de 7+ resolve CONFLITO explicitamente ("onde discordarem, a
# referencia vence") porque e o que um LLM sabe seguir -- muito mais eficaz
# que adjetivo solto tipo "muito importante".
# ---------------------------------------------------------------------------
_FORCA = [
    (1, 3, "solta",  "{T} is a loose reference here: take it as a starting point and "
                     "reinterpret it freely where the description suggests otherwise."),
    (4, 6, "padrao", ""),          # o bloco como escrito, sem acrescimo
    (7, 8, "firme",  "Follow {T} closely: where the description and {T} disagree about "
                     "anything {T} defines, {T} wins."),
    (9, 10, "maxima", "Follow {T} exactly. Everything {T} defines is fixed and is not open "
                      "to reinterpretation; where the description and {T} disagree about "
                      "anything {T} defines, {T} wins."),
]


def _frase_forca(n):
    """Devolve (rotulo, frase) para a intensidade pedida."""
    for lo, hi, rotulo, frase in _FORCA:
        if lo <= n <= hi:
            return rotulo, frase
    return ("padrao", "")
_RE_REF = re.compile(r"@([A-Za-zÀ-ſ]*)(\d*)")


def _sem_acento(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s or "")
                   if unicodedata.category(c) != "Mn").lower()


def expandir(texto, avisos, usados, vistos, fortes):
    """Faz as tres passadas na ordem que importa.

    #d: PRIMEIRO -- se o '@' rodasse antes, um '@' dentro da fala viraria tag.
    #papel DEPOIS -- ele ja injeta '<Video 1>' pronto.
    @ POR ULTIMO -- so sobra o que o usuario escreveu a mao.
    """
    protegido = {}

    def guardar(txt):
        chave = "\x00%d\x00" % len(protegido)
        protegido[chave] = txt
        return chave

    # escapes: '@@' e '##' viram literais e saem do caminho
    texto = texto.replace("@@", guardar("@")).replace("##", guardar("#"))

    # ---- 1) falas ----
    def _dial(m):
        fala = " ".join(m.group(1).split())
        if not fala:
            avisos.append("um '#d:...#' estava vazio e foi removido.")
            return ""
        return guardar("<d>%s</d>" % fala)
    texto = _RE_DIALOGO.sub(_dial, texto)

    # ---- 2) papeis ----
    def _macro(m):
        bruto, num = m.group(1), m.group(2)
        ref_tipo, ref_num = m.group(3), m.group(4)   # forma '#papel @video2'
        forca = m.group(5)                            # forma '#papel(9)'
        sobra = ""                                    # '@...' que nao coube na macro
        papel = _ALIAS.get(_sem_acento(bruto))
        if papel:
            vistos.add(papel)
        if not papel:
            avisos.append("'#%s' nao e um papel conhecido -- deixei o texto como estava. "
                          "Veja a saida 'ajuda' para a lista." % bruto)
            return m.group(0)
        tipo, corpo = _BLOCOS[papel]
        if tipo is None:                      # macro sem referencia (invariancia, semlegenda)
            if forca:
                avisos.append("'#%s' nao aponta pra referencia nenhuma, entao '(%s)' nao "
                              "significa nada aqui -- ignorei." % (bruto, forca))
            # Se havia um '@...' logo depois, ele NAO era desta macro: devolve
            # intacto para a passada do '@' resolver.
            if ref_tipo is not None or ref_num:
                return guardar(corpo) + " @%s%s" % (ref_tipo or "", ref_num or "")
            return guardar(corpo)
        # ---- de onde sai o numero: '#papel2' ou '#papel @video2' ----
        if ref_tipo is not None or ref_num:
            if ref_tipo:
                t_apontado = _ALIAS_TIPO.get(_sem_acento(ref_tipo))
                if t_apontado is None:
                    avisos.append("'@%s' depois de '#%s' nao e um tipo conhecido "
                                  "(img / video / audio) -- ignorei o apontamento."
                                  % (ref_tipo, bruto))
                elif t_apontado != tipo:
                    avisos.append("'#%s' e papel de %s, mas voce apontou '@%s%s', que e %s. "
                                  "Usei %s %s -- se queria outra coisa, troque a macro."
                                  % (bruto, _TAG[tipo], ref_tipo, ref_num, _TAG[t_apontado],
                                     _TAG[tipo], ref_num or "1"))
            if num and ref_num and num != ref_num:
                avisos.append("'#%s%s @%s%s' diz dois numeros diferentes. Usei o do '@'."
                              % (bruto, num, ref_tipo or "", ref_num))
            num = ref_num or num
        n = int(num) if num else 1
        if n < 1 or n > _LIMITE[tipo]:
            avisos.append("'#%s%s' pede %s %d, mas o H3 aceita no maximo %d %s(s)."
                          % (bruto, num, _TAG[tipo], n, _LIMITE[tipo], _TAG[tipo]))
            n = max(1, min(n, _LIMITE[tipo]))
        tag = "<%s %d>" % (_TAG[tipo], n)
        usados.setdefault(tipo, set()).add(n)
        texto_bloco = corpo
        if forca:
            f = int(forca)
            if f < 1 or f > 10:
                avisos.append("'#%s(%s)' -- a forca vai de 1 a 10. Usei %d."
                              % (bruto, forca, max(1, min(f, 10))))
                f = max(1, min(f, 10))
            rotulo, frase = _frase_forca(f)
            if frase:
                texto_bloco = texto_bloco + " " + frase
            fortes.append("%s(%d)=%s" % (papel, f, rotulo))
        return guardar(texto_bloco.replace("{T}", tag))
    texto = _RE_MACRO.sub(_macro, texto)

    # ---- 3) referencias soltas ----
    def _ref(m):
        bruto, num = m.group(1), m.group(2)
        if not bruto and not num:
            return m.group(0)                 # '@' sozinho: nao mexe
        if not bruto:                          # '@2' -> imagem, o caso comum
            tipo = "picture"
        else:
            tipo = _ALIAS_TIPO.get(_sem_acento(bruto))
            if not tipo:
                avisos.append("'@%s' nao e um tipo conhecido (use img / video / audio)."
                              % bruto)
                return m.group(0)
        n = int(num) if num else 1
        if n < 1 or n > _LIMITE[tipo]:
            avisos.append("'@%s%s' pede %s %d, mas o H3 aceita no maximo %d."
                          % (bruto, num, _TAG[tipo], n, _LIMITE[tipo]))
            n = max(1, min(n, _LIMITE[tipo]))
        usados.setdefault(tipo, set()).add(n)
        return guardar("<%s %d>" % (_TAG[tipo], n))
    texto = _RE_REF.sub(_ref, texto)

    for chave, valor in protegido.items():
        texto = texto.replace(chave, valor)
    return texto


def macros_payload():
    """Catalogo servido em /bruxos/h3_macros para o popup do '#' e do '@'.

    O JS NAO guarda copia destes textos -- se guardasse, editar _BLOCOS aqui
    deixaria a lista do popup mentindo sobre o que a macro faz.
    """
    # descricao curta em PT, so para o popup. O paragrafo em ingles vai junto
    # como 'preview' para quem quiser conferir antes de inserir.
    curto = {
        "frame": "a imagem é o primeiro quadro; o movimento parte dela",
        "sujeito": "fixa identidade — rosto, corpo, cabelo, roupa, cores",
        "cenario": "material, paleta e luz do lugar",
        "fundo": "troca o fundo e faz a luz nova agir no sujeito",
        "estilo": "só traço/paleta/grão; o assunto da imagem não vaza",
        "camera": "só movimento e enquadramento; aparência descartada",
        "previz": "blocking do Blender: geometria é lei, cinza é descartado",
        "fonte": "filmagem real sendo editada; preserva o look",
        "continuar": "o novo clipe começa onde o outro parou",
        "relight": "trava movimento e identidade, libera luz e ambiente",
        "depth": "mapa de profundidade: só geometria, o cinza não vira look",
        "st": "style transfer: trava a estrutura, troca só o tratamento visual",
        "voz": "copia o timbre, não as palavras",
        "trilha": "música de fundo no tempo e clima originais",
        "copiar": "áudio reproduzido igual",
        "ambiente": "ambiência e efeitos sonoros",
        "invariancia": "cláusula de fecho nomeando os seis eixos",
        "semlegenda": "sem legenda, marca d'água ou texto na tela",
    }
    # sinonimos, para o popup filtrar por eles tambem ("#clay" acha previz)
    sinonimos = {}
    for palavra, canon in _ALIAS.items():
        if palavra != canon:
            sinonimos.setdefault(canon, []).append(palavra)

    macros = []
    for papel, (tipo, corpo) in _BLOCOS.items():
        macros.append({
            "nome": papel,
            "tipo": tipo or "",                       # "" = nao usa referencia
            "tag": _TAG[tipo] if tipo else "",
            "limite": _LIMITE.get(tipo, 0) if tipo else 0,
            "curto": curto.get(papel, ""),
            "texto": corpo,
            "sinonimos": sorted(sinonimos.get(papel, [])),
        })
    refs = []
    for tipo, aliases in _TIPO_ALIAS.items():
        refs.append({
            "tipo": tipo,
            "tag": _TAG[tipo],
            "limite": _LIMITE[tipo],
            "principal": aliases[0],
            "sinonimos": list(aliases),
        })
    return {"macros": macros, "refs": refs}


def _ajuda():
    linhas = ["REFERENCIAS", "  @img  @img2  @1        -> <Picture 1>, <Picture 2>, <Picture 1>",
              "  @video  @v2             -> <Video 1>, <Video 2>",
              "  @audio  @a3             -> <Audio 1>, <Audio 3>",
              "", "FALA", "  #d:oi, tudo bem?#       -> <d>oi, tudo bem?</d>",
              "", "FORCA  (entre parenteses, 1 a 10)",
              "  #sujeito2(9)            segue com rigor",
              "  #camera(2)              so ponto de partida",
              "  1-3 solta | 4-6 padrao | 7-8 firme | 9-10 maxima",
              "  Sao FAIXAS: o encoder do H3 le prosa, nao numero -- o (N) vira",
              "  frase. Dez degraus distintos de 'siga isto' nao existem em texto.",
              "", "PAPEIS  (aceita numero: #camera2 amarra no <Video 2>)"]
    for papel, (tipo, _c) in _BLOCOS.items():
        if tipo is None:
            continue
        linhas.append("  #%-11s [%s]" % (papel, _TAG[tipo]))
    linhas += ["", "EXTRAS"]
    for papel, (tipo, _c) in _BLOCOS.items():
        if tipo is None:
            linhas.append("  #%-11s" % papel)
    linhas += ["", "Escreva '@@' ou '##' para um @ ou # literal."]
    return "\n".join(linhas)


class BruxosH3PromptRapido:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texto": ("STRING", {"multiline": True, "default":
                    "#previz  #sujeito2  #estilo3\n"
                    "Wide shot of an ancient city gate at dusk, warm torchlight on stone.\n"
                    "#invariancia",
                    "tooltip":
                    "Escreva normal e use as macros. A saida 'ajuda' lista todas.\n\n"
                    "@img @video @audio (com numero opcional) viram <Picture N> <Video N> <Audio N>.\n"
                    "#previz #camera #fundo #estilo ... injetam o paragrafo daquele papel ja "
                    "amarrado na tag certa -- #camera2 amarra no <Video 2>.\n"
                    "#d:fala aqui# vira <d>fala aqui</d>.\n\n"
                    "Nada aqui carrega modelo: e substituicao de texto, roda em milissegundos."}),
            },
            "optional": {
                "duracao_s": ("FLOAT", {"default": 8.0, "min": 4.0, "max": 15.0, "step": 0.1,
                    "tooltip":
                    "So para calcular o 'length' na grade do H3 (length % 17 == 5).\n"
                    "8.0s = 192 frames e a UNICA duracao de segundo inteiro entre 4s e 15s -- "
                    "use ela quando precisar casar 1:1 com render de Blender."}),
                "prefixo": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "[opcional] Texto colado ANTES do seu (tambem aceita macros). Bom para guardar "
                    "um cabecalho fixo de projeto."}),
                "sufixo": ("STRING", {"multiline": True, "default": "", "tooltip":
                    "[opcional] Texto colado DEPOIS do seu (tambem aceita macros). Bom para regras "
                    "que voce sempre repete, tipo '#semlegenda'."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "length", "info", "ajuda")
    OUTPUT_TOOLTIPS = (
        "O prompt expandido -> ligue no 'prompt' do MiniMaxH3ReferenceToVideo.",
        "Frames na grade do H3 para a 'duracao_s' pedida.",
        "Contagem de caracteres, referencias usadas e avisos.",
        "A lista completa de macros -- ligue num Preview/Display Any para consultar.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "MiniMax H3 · Prompt Rapido (Bruxos): expande macros '@' e '#' em tags do H3, sem carregar "
        "modelo nenhum. '@video' vira <Video 1>; '#previz' injeta o paragrafo inteiro daquele papel "
        "ja amarrado na tag certa. Ideia do '@'/'#' vinda do ComfyUI-MiniMaxH3-Easy (nkxx188, MIT). "
        "Para escrever o Context-IR completo com LLM, use o node 'Context-IR' -- este aqui e o "
        "caminho rapido."
    )

    def run(self, texto, duracao_s=8.0, prefixo="", sufixo=""):
        avisos, usados, vistos, fortes = [], {}, set(), []
        partes = [p for p in (prefixo, texto, sufixo) if p and p.strip()]
        bruto = "\n\n".join(partes)
        prompt = expandir(bruto, avisos, usados, vistos, fortes)

        # A cláusula de invariância diz "environment ... remain unchanged".
        # Com #previz e #fundo o objetivo E trocar a aparencia do ambiente,
        # entao as duas se contradizem -- foi assim que o clay cinza do Blender
        # voltou numa geracao. Os dois papeis ja trazem a propria preservacao.
        if "invariancia" in vistos:
            briga = sorted(vistos & {"previz", "fundo", "camera", "estilo"})
            if briga:
                avisos.append(
                    "'#invariancia' junto com '#%s': a cláusula diz que o AMBIENTE fica "
                    "inalterado, e é justamente a aparência do ambiente que esses papéis "
                    "trocam. Eles já trazem a própria frase de preservação -- tire o "
                    "'#invariancia' se o cenário voltar com o look da referência de origem."
                    % "', '#".join(briga))

        # normaliza espaco em branco sem destruir paragrafos
        prompt = re.sub(r"[ \t]+", " ", prompt)
        prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()

        # --- length na grade do H3: base + (5 - base % 17) % 17 ---
        base = max(5, int(round(float(duracao_s) * 24.0)))
        length = base + (5 - base % 17) % 17
        dur_real = length / 24.0

        n = len(prompt)
        if n > 7000:
            avisos.append("o prompt tem %d caracteres e o limite do H3 e 7000. O excedente e "
                          "cortado pelo modelo, entao o final do seu texto some." % n)
        if not prompt:
            avisos.append("o prompt saiu VAZIO.")

        ref = " | ".join(
            "%s: %s" % (_TAG[t], ", ".join(str(x) for x in sorted(v)))
            for t, v in sorted(usados.items()) if v) or "nenhuma"
        for t, v in usados.items():
            faltando = [x for x in range(1, max(v)) if x not in v]
            if faltando:
                avisos.append("voce usou %s %d mas nao usou %s -- confira se as referencias "
                              "estao ligadas na ordem que voce imagina, porque a numeracao segue "
                              "a ORDEM DOS SLOTS, nao o que voce escreveu."
                              % (_TAG[t], max(v), ", ".join("%s %d" % (_TAG[t], f) for f in faltando)))

        if abs(dur_real - round(dur_real)) > 1e-6:
            avisos.append("%d frames = %.4fs, que nao e segundo redondo. A unica duracao inteira "
                          "entre 4s e 15s e 8s (192 frames)." % (length, dur_real))

        info = ("%d caracteres | length %d (%.4fs @24fps) | referencias -> %s%s"
                % (n, length, dur_real, ref,
                   " | forca -> " + ", ".join(fortes) if fortes else ""))
        print("[Bruxos H3 Prompt Rapido] %s" % info, flush=True)
        for a in avisos:
            print("[Bruxos H3 Prompt Rapido]   AVISO: %s" % a, flush=True)
        if avisos:
            info += "\n" + "\n".join("- %s" % a for a in avisos)
        return (prompt, int(length), info, _ajuda())


NODE_CLASS_MAPPINGS = {"BruxosH3PromptRapido": BruxosH3PromptRapido}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3PromptRapido": "MiniMax H3 · Prompt Rápido @ # (Bruxos)"
}
