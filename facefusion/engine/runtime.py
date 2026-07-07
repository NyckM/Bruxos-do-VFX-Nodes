# -*- coding: utf-8 -*-
"""Selecao de ExecutionProvider do onnxruntime pro FaceFusion (Bruxos).

O device e escolhido no node (auto/cuda/cpu). Se o usuario pedir CUDA mas o
onnxruntime instalado nao tiver o CUDAExecutionProvider (conflito de pacote:
onnxruntime-openvino/directml/cpu ocupando o lugar do onnxruntime-gpu), a gente
NAO cai mudo na CPU: imprime um aviso claro com o conserto.
"""

import onnxruntime as ort

DEVICE = "auto"          # auto | cuda | cpu
_warned = False


def set_device(device):
    global DEVICE
    DEVICE = (device or "auto").strip().lower()


def _warn_no_cuda(avail):
    global _warned
    if _warned:
        return
    _warned = True
    print("[Bruxos FaceFusion] AVISO: CUDAExecutionProvider NAO esta disponivel no "
          "onnxruntime -> rodando na CPU (bem mais lento).", flush=True)
    print(f"[Bruxos FaceFusion] Providers disponiveis: {avail}", flush=True)
    if any(("openvino" in p.lower() or "dml" in p.lower() or "directml" in p.lower()) for p in avail):
        print("[Bruxos FaceFusion] Detectei um onnxruntime NAO-CUDA (openvino/directml). "
              "Ter mais de uma variante do onnxruntime instalada tira o CUDA do ar.", flush=True)
    print("[Bruxos FaceFusion] Conserto (no python_embeded do ComfyUI):", flush=True)
    print("   python.exe -m pip uninstall -y onnxruntime onnxruntime-gpu "
          "onnxruntime-openvino onnxruntime-directml", flush=True)
    print("   python.exe -m pip install onnxruntime-gpu", flush=True)
    print("[Bruxos FaceFusion] (mantenha SO o onnxruntime-gpu; reinicie o ComfyUI depois.)", flush=True)


def resolve_providers():
    """Retorna a lista de providers pra passar no InferenceSession, conforme DEVICE."""
    try:
        avail = list(ort.get_available_providers())
    except Exception:
        avail = ["CPUExecutionProvider"]

    if DEVICE == "cpu":
        return ["CPUExecutionProvider"]

    # auto ou cuda
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # CUDA pedido/auto, mas indisponivel
    _warn_no_cuda(avail)
    if DEVICE == "cuda":
        print("[Bruxos FaceFusion] device=cuda foi pedido, mas nao ha CUDA no onnxruntime; "
              "usando CPU por ora.", flush=True)
    return ["CPUExecutionProvider"]
