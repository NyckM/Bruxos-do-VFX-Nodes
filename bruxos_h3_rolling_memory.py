"""
MiniMax H3 - memoria rolante entre geracoes independentes de 124 frames.

Nao e KV-cache e nao finge ser memoria interna do Transformer. O estado salvo
e multimodal e explicito: tail temporal (Video de referencia), primeiro frame
permanente e um marco recente (Pictures de referencia). A execucao N salva o
estado que a execucao N+1 le, evitando ciclos no grafo do ComfyUI.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time

import torch

try:
    import folder_paths
except Exception:
    folder_paths = None


CAT = "Bruxos do VFX/MiniMax H3/Memoria Rolante"
MARCA = "BRUXOS_H3_ROLLING_V1"


def _seguro(nome):
    nome = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(nome or "h3_rolling")).strip("._")
    return nome[:96] or "h3_rolling"


def _raiz():
    if folder_paths is not None:
        base = folder_paths.get_temp_directory()
    else:
        base = os.path.join(os.getcwd(), ".bruxos_temp")
    path = os.path.abspath(os.path.join(base, "bruxos_h3_rolling"))
    os.makedirs(path, exist_ok=True)
    return path


def _job(nome):
    return os.path.join(_raiz(), _seguro(nome))


def _save_tensor(tensor, path):
    tmp = path + ".tmp"
    torch.save(tensor.detach().to(device="cpu", dtype=torch.float16).contiguous(), tmp)
    os.replace(tmp, path)


def _load_tensor(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True).float()
    except TypeError:
        return torch.load(path, map_location="cpu").float()


def _save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _validar_image(x, nome="imagens"):
    if not torch.is_tensor(x) or x.ndim != 4 or int(x.shape[0]) < 1:
        raise ValueError(f"[Bruxos H3 Rolling] {nome} precisa ser IMAGE [T,H,W,C], nao vazio.")
    return x[..., :3]


class BruxosH3Bloco124:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "indice_bloco": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                      "tooltip": "0 no primeiro render; aumente para 1, 2, 3..."}),
            "sobreposicao_fonte": ("INT", {"default": 0, "min": 0, "max": 123,
                                            "tooltip": "Sobreposicao apenas ao RECORTAR o video-fonte. "
                                                       "A geracao continua tendo 124 frames."}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("inicio_fonte", "length", "indice_bloco", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = "Controlador de blocos H3 fixos de 124 frames (grade 17k+5)."

    def run(self, indice_bloco, sobreposicao_fonte):
        indice = max(0, int(indice_bloco))
        passo = 124 - min(123, max(0, int(sobreposicao_fonte)))
        inicio = indice * passo
        info = f"bloco {indice:05d} | fonte {inicio}:{inicio + 124} | length=124 | passo={passo}"
        return inicio, 124, indice, info


class BruxosH3BlocoFlex:
    """Controlador temporal H3 sem expressoes, para qualquer grade 17k+5."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "indice_bloco": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                      "tooltip": "0 no primeiro render; aumente para 1, 2, 3..."}),
            "frames_por_bloco": ("INT", {"default": 124, "min": 56, "max": 1025, "step": 17,
                                           "tooltip": "Precisa obedecer 17k+5. Exemplos: 56, 73, 90, 107, 124, 192."}),
            "sobreposicao_fonte": ("INT", {"default": 17, "min": 0, "max": 1024,
                                              "tooltip": "Frames reaproveitados entre blocos. 17 = um passo temporal H3."}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("inicio_fonte", "length", "indice_bloco", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = "Controlador flexivel de blocos MiniMax H3 na grade temporal 17k+5."

    def run(self, indice_bloco, frames_por_bloco, sobreposicao_fonte):
        indice = max(0, int(indice_bloco))
        length = int(frames_por_bloco)
        if length < 5 or (length - 5) % 17 != 0:
            anterior = max(5, ((length - 5) // 17) * 17 + 5)
            proximo = anterior + 17
            raise ValueError(
                f"[Bruxos H3 Bloco Flex] {length} frames nao pertence a grade 17k+5. "
                f"Use {anterior} ou {proximo}; 124 e 192 sao validos."
            )
        overlap = min(length - 1, max(0, int(sobreposicao_fonte)))
        passo = length - overlap
        inicio = indice * passo
        info = (f"bloco {indice:05d} | fonte {inicio}:{inicio + length} | "
                f"length={length} | overlap={overlap} | passo={passo}")
        return inicio, length, indice, info


class BruxosH3MemoriaLer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "fallback_atual": ("IMAGE", {"tooltip": "Bloco-fonte atual. So e usado para inicializar o bloco 0."}),
            "nome_job": ("STRING", {"default": "lyonir_rolling_01"}),
            "indice_bloco": ("INT", {"forceInput": True}),
            "tail_frames": ("INT", {"default": 56, "min": 56, "max": 124, "step": 17,
                                   "tooltip": "56 = 2,33 s e respeita a grade 17k+5/minimo oficial de 2 s do Ref2VA."}),
            "exigir_anterior": ("BOOLEAN", {"default": True,
                                             "tooltip": "No bloco >0, para com erro se o bloco anterior nao foi salvo."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("tail_video", "memoria_inicial", "memoria_marco", "ultimo_frame", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Antes do sampler, le o estado salvo pelo bloco anterior. Tail e referencia temporal; "
        "as duas imagens sao memoria visual, nao keyframes posicionais."
    )

    def run(self, fallback_atual, nome_job, indice_bloco, tail_frames, exigir_anterior):
        fallback = _validar_image(fallback_atual, "fallback_atual").detach().cpu().float()
        indice = max(0, int(indice_bloco))
        tail_n = max(56, int(tail_frames))
        pasta = _job(nome_job)
        anterior = indice - 1
        tail_path = os.path.join(pasta, f"tail_{anterior:05d}.pt")
        first_path = os.path.join(pasta, "memoria_inicial.pt")
        marco_path = os.path.join(pasta, f"marco_{anterior:05d}.pt")

        usando_estado = indice > 0 and os.path.isfile(tail_path)
        if indice > 0 and not usando_estado and exigir_anterior:
            raise FileNotFoundError(
                f"[Bruxos H3 Rolling] falta a memoria do bloco {anterior}: {tail_path}.\n"
                f"Rode primeiro o bloco {anterior}, com o mesmo nome_job '{nome_job}'."
            )

        if usando_estado:
            tail = _load_tensor(tail_path)
            primeiro = _load_tensor(first_path) if os.path.isfile(first_path) else tail[:1]
            marco = _load_tensor(marco_path) if os.path.isfile(marco_path) else tail[tail.shape[0] // 2:tail.shape[0] // 2 + 1]
            origem = f"estado do bloco {anterior}"
        else:
            # O bloco zero precisa produzir sockets validos para o Autogrow do
            # Ref2VA. Ele recebe amostras fracas do proprio video-fonte; o prompt
            # dinamico deixa claro que elas NAO sao um passado gerado.
            if int(fallback.shape[0]) >= tail_n:
                tail = fallback[:tail_n]
            else:
                tail = torch.cat([fallback, fallback[-1:].repeat(tail_n - int(fallback.shape[0]), 1, 1, 1)], 0)
            primeiro = fallback[:1]
            meio = int(fallback.shape[0]) // 2
            marco = fallback[meio:meio + 1]
            origem = "inicializacao pelo bloco-fonte atual"

        ultimo = tail[-1:]
        info = (f"bloco {indice} | {origem} | tail={int(tail.shape[0])} frames | "
                f"memoria inicial=1 | marco=1 | job={pasta}")
        print(f"[Bruxos H3 Rolling] {info}", flush=True)
        return tail, primeiro, marco, ultimo, info

    @classmethod
    def IS_CHANGED(cls, fallback_atual, nome_job, indice_bloco, tail_frames, exigir_anterior):
        indice = max(0, int(indice_bloco))
        if indice <= 0:
            return (0, nome_job, tail_frames, exigir_anterior)
        path = os.path.join(_job(nome_job), f"tail_{indice - 1:05d}.pt")
        try:
            stat = os.stat(path)
            return (stat.st_mtime_ns, stat.st_size, indice, tail_frames, exigir_anterior)
        except OSError:
            return float("nan")


class BruxosH3MemoriaPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt_base": ("STRING", {"forceInput": True}),
            "indice_bloco": ("INT", {"forceInput": True}),
            "tail_frames": ("INT", {"default": 56, "min": 56, "max": 124, "step": 17}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Declara os papeis das referencias da memoria. Picture 2/3 nunca sao tratados como "
        "composicao obrigatoria; Video 2 recebe prioridade de continuidade local."
    )

    def run(self, prompt_base, indice_bloco, tail_frames):
        indice = max(0, int(indice_bloco))
        if indice == 0:
            memoria = f"""ROLLING_BLOCK_CONTEXT:
This is generation block 0, exactly 124 frames at 24 fps. <Video 1> is the current source/previz block and controls its camera path, geometry, staging and timing. <Video 2> is only a {int(tail_frames)}-frame initialization sample taken from the same current source; it is not an earlier generated event. <Picture 1> may establish the opening appearance requested by the user. <Picture 2> and <Picture 3> are weak visual memory references for stable identity, materials, palette and lighting only. They do not require the target to reproduce their pose, camera position, framing or moment in time.
"""
        else:
            memoria = f"""ROLLING_BLOCK_CONTEXT:
This is generation block {indice}, exactly 124 frames at 24 fps. [video continuation + video editing + reference generation] <Video 2> is the {int(tail_frames)}-frame generated tail immediately preceding this block. Continue directly from its final state, preserving instantaneous subject identity, pose trajectory, direction and speed of motion, camera velocity, lighting phase and spatial relationships. <Video 1> is the current source/previz block and controls the new block's camera path, geometry, staging and timing. <Picture 2> is long-term visual memory from the beginning of the generated sequence: use it only for stable identity, clothing, materials and palette, never as a command to return to its pose, framing, background arrangement or time. <Picture 3> is a medium-term memory from the preceding block: retain only changes that remain logically true now. Do not recreate either memory picture as a target frame. Local temporal continuity from <Video 2> has priority over old composition.
"""
        prompt = memoria + "\nUSER_AND_SCENE_DIRECTION:\n" + str(prompt_base).strip()
        info = f"prompt de memoria para bloco {indice} | tail {int(tail_frames)} | 124 frames"
        return prompt, info


class BruxosH3MemoriaSalvar:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "imagens": ("IMAGE",),
            "nome_job": ("STRING", {"default": "lyonir_rolling_01"}),
            "indice_bloco": ("INT", {"forceInput": True}),
            "tail_frames": ("INT", {"default": 56, "min": 56, "max": 124, "step": 17}),
            "sobrescrever_bloco": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("bloco", "tail", "memoria_inicial", "memoria_marco", "job", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Depois do decode, salva o bloco em float16 no SSD e publica o estado para a proxima execucao. "
        "Nao mantem os blocos anteriores em RAM/VRAM."
    )

    def run(self, imagens, nome_job, indice_bloco, tail_frames, sobrescrever_bloco):
        x = _validar_image(imagens).detach().cpu().float()
        indice = max(0, int(indice_bloco))
        tail_n = min(int(x.shape[0]), max(56, int(tail_frames)))
        pasta = _job(nome_job)
        os.makedirs(pasta, exist_ok=True)
        bloco_path = os.path.join(pasta, f"bloco_{indice:05d}.pt")
        if os.path.exists(bloco_path) and not sobrescrever_bloco:
            raise FileExistsError(
                f"[Bruxos H3 Rolling] bloco {indice} ja existe: {bloco_path}. "
                "Ligue sobrescrever_bloco somente se deseja substituir essa geracao."
            )

        inicio_t = time.perf_counter()
        tail = x[-tail_n:]
        primeiro = x[:1]
        meio = int(x.shape[0]) // 2
        marco = x[meio:meio + 1]
        _save_tensor(x, bloco_path)
        _save_tensor(tail, os.path.join(pasta, f"tail_{indice:05d}.pt"))
        if indice == 0 or not os.path.isfile(os.path.join(pasta, "memoria_inicial.pt")):
            _save_tensor(primeiro, os.path.join(pasta, "memoria_inicial.pt"))
        else:
            primeiro = _load_tensor(os.path.join(pasta, "memoria_inicial.pt"))
        _save_tensor(marco, os.path.join(pasta, f"marco_{indice:05d}.pt"))
        manifesto = {
            "marca": MARCA, "nome_job": _seguro(nome_job), "ultimo_bloco": indice,
            "frames_bloco": int(x.shape[0]), "tail_frames": tail_n,
            "altura": int(x.shape[1]), "largura": int(x.shape[2]), "dtype_disco": "float16",
            "atualizado": time.time(),
        }
        _save_json(os.path.join(pasta, "manifesto.json"), manifesto)
        dur = time.perf_counter() - inicio_t
        info = (f"bloco {indice:05d} salvo ({int(x.shape[0])} frames) | tail {tail_n} | "
                f"memorias inicial+marco | {dur:.2f}s | {pasta}")
        print(f"[Bruxos H3 Rolling] {info}", flush=True)
        return x, tail, primeiro, marco, pasta, info


class BruxosH3MemoriaLimpar:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "nome_job": ("STRING", {"default": "lyonir_rolling_01"}),
            "confirmar": ("BOOLEAN", {"default": False,
                                      "tooltip": "Apaga todos os blocos e memorias deste job. Nao e reversivel."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    CATEGORY = CAT
    OUTPUT_NODE = True

    def run(self, nome_job, confirmar):
        pasta = _job(nome_job)
        if not confirmar:
            return (f"nao limpei; confirme para apagar {pasta}",)
        raiz = os.path.abspath(_raiz())
        alvo = os.path.abspath(pasta)
        if os.path.commonpath([raiz, alvo]) != raiz or alvo == raiz:
            raise ValueError("[Bruxos H3 Rolling] alvo de limpeza inseguro")
        if os.path.isdir(alvo):
            shutil.rmtree(alvo)
            return (f"job removido: {alvo}",)
        return (f"job nao existia: {alvo}",)


NODE_CLASS_MAPPINGS = {
    "BruxosH3Bloco124": BruxosH3Bloco124,
    "BruxosH3BlocoFlex": BruxosH3BlocoFlex,
    "BruxosH3MemoriaLer": BruxosH3MemoriaLer,
    "BruxosH3MemoriaPrompt": BruxosH3MemoriaPrompt,
    "BruxosH3MemoriaSalvar": BruxosH3MemoriaSalvar,
    "BruxosH3MemoriaLimpar": BruxosH3MemoriaLimpar,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3Bloco124": "H3 Bloco 124 (Bruxos)",
    "BruxosH3BlocoFlex": "H3 Bloco Flex 17k+5 (Bruxos)",
    "BruxosH3MemoriaLer": "H3 Memoria Rolante - Ler (Bruxos)",
    "BruxosH3MemoriaPrompt": "H3 Memoria Rolante - Prompt (Bruxos)",
    "BruxosH3MemoriaSalvar": "H3 Memoria Rolante - Salvar (Bruxos)",
    "BruxosH3MemoriaLimpar": "H3 Memoria Rolante - Limpar Job (Bruxos)",
}
