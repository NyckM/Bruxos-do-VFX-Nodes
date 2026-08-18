# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — resizer neural pro latente do Wan/Bernini (16ch)
=====================================================================
PRA QUE SERVE
    O 'Bernini Latent Upscale (Bruxos)' upscala o latente por INTERPOLACAO
    pura (bicubic/bilinear/...). Isso funciona, mas e a causa raiz do
    fantasma/imagem-duplicada que aparece quando voce baixa o denoise do
    passe 2 pra tentar segurar a identidade: interpolacao pura NAO inventa
    detalhe nenhum, so "borra" o latente pro tamanho novo, e um denoise
    baixo nao da tempo do sampler resolver essa borra em algo coerente.

    Este modulo troca a interpolacao por uma rede neural pequena treinada
    especificamente pra "adivinhar" como o latente do Wan2.2 deveria ficar
    numa resolucao maior -- em vez de so espalhar os mesmos valores.

DE ONDE VEIO
    Arquitetura + pesos do projeto DenRakEiw/WAN_NN_Latent_Upscale
    (https://github.com/DenRakEiw/WAN_NN_Latent_Upscale), que por sua vez
    se baseia no Ttl/ComfyUi_NNLatentUpscale. E a MESMA ideia por tras do
    upscaler neural do MiniMax H3 (LBH-123-AI/Comfyui_Minimax_h3_latent_
    Upscaler) que resolve o mesmo fantasma la -- so que pro H3 (24ch) e nao
    existe uma rede assim madura pro Bernini/Wan2.2, entao usamos essa.

AVISO IMPORTANTE (honestidade, nao propaganda)
    O proprio autor marca o modelo do Wan2.2 como "WIP"/"Improving" -- o
    ganho medido por ele e modesto (SSIM ~20% melhor que bilinear, mas em
    numero absoluto ainda baixo: 0.3247 vs 0.2690). Ou seja: deve ajudar
    com o fantasma (que e um problema estrutural da interpolacao pura),
    mas NAO e uma rede tao madura quanto a do H3. Trate como experimental:
    compare lado a lado com o metodo bicubic antes de confiar de olhos
    fechados.

DOWNLOAD DO CHECKPOINT
    O arquivo (~3.9MB) NAO vem junto do pack -- e baixado sozinho na
    primeira vez que voce usar metodo="neural" e fica salvo em
    'ComfyUI-Bruxos-do-VFX/models/wan22_neural_resizer_best.pt'. Se sua
    maquina do ComfyUI nao tem internet liberada pra isso, baixe manualmente
    em:
        https://raw.githubusercontent.com/DenRakEiw/WAN_NN_Latent_Upscale/main/models/wan2.2_resizer_best.pt
    e coloque nesse caminho (crie a pasta 'models' se nao existir).
"""

import logging
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)

_LOCAL_DIR = os.path.dirname(os.path.realpath(__file__))
_MODELS_DIR = os.path.join(_LOCAL_DIR, "models")
_CKPT_PATH = os.path.join(_MODELS_DIR, "wan22_neural_resizer_best.pt")
_CKPT_URL = (
    "https://raw.githubusercontent.com/DenRakEiw/WAN_NN_Latent_Upscale/"
    "main/models/wan2.2_resizer_best.pt"
)

# Empiricamente determinado pelo autor do checkpoint pro latente do Wan2.2
# (16 canais). Normaliza a magnitude do latente antes de entrar na rede.
SCALE_FACTOR = 0.3604

_modelo_cache = {"model": None, "device": None, "dtype": None}


if _OK:
    class _WanLatentResizer(nn.Module):
        """
        Mesma arquitetura do checkpoint 'wan2.2_resizer_best.pt' (16ch),
        copiada do DenRakEiw/WAN_NN_Latent_Upscale::latent_resizer.py --
        os nomes das camadas precisam bater EXATAMENTE com o state_dict.
        """

        def __init__(self, in_channels: int = 16, out_channels: int = 16):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1),   # encoder.0
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1),           # encoder.2
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, padding=1),          # encoder.4
                nn.ReLU(inplace=True),
            )
            self.upsampler = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # upsampler.0
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),                     # upsampler.2
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels, 3, padding=1),            # upsampler.4
            )
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)

        def forward(self, x, scale: float = 2.0):
            skip = self.skip_conv(x)
            features = self.encoder(x)
            upsampled = self.upsampler(features)  # sempre 2x nativo (stride=2)

            if upsampled.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=upsampled.shape[2:], mode="bilinear", align_corners=False)
            out = upsampled + skip

            if abs(float(scale) - 2.0) > 0.01:
                target_h = int(round(x.shape[2] * float(scale)))
                target_w = int(round(x.shape[3] * float(scale)))
                out = F.interpolate(out, size=(target_h, target_w), mode="bilinear", align_corners=False)
            return out


def _garantir_checkpoint():
    """Baixa o checkpoint (uma vez so) se ainda nao existir localmente."""
    if os.path.isfile(_CKPT_PATH) and os.path.getsize(_CKPT_PATH) > 0:
        return True
    os.makedirs(_MODELS_DIR, exist_ok=True)
    print(f"[Bruxos Bernini Neural Upscale] checkpoint nao encontrado, baixando de "
          f"{_CKPT_URL} ...", flush=True)
    try:
        torch.hub.download_url_to_file(_CKPT_URL, _CKPT_PATH, progress=True)
        print(f"[Bruxos Bernini Neural Upscale] checkpoint salvo em {_CKPT_PATH}.", flush=True)
        return True
    except Exception as e:
        print(f"[Bruxos Bernini Neural Upscale] FALHOU baixar o checkpoint automaticamente ({e}). "
              f"Baixe manualmente de {_CKPT_URL} e salve em {_CKPT_PATH}.", flush=True)
        return False


def _carregar_modelo(device, dtype):
    cached = _modelo_cache["model"]
    if cached is not None and _modelo_cache["device"] == device and _modelo_cache["dtype"] == dtype:
        return cached

    if not _garantir_checkpoint():
        return None

    try:
        model = _WanLatentResizer(in_channels=16, out_channels=16)
        state_dict = torch.load(_CKPT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device=device, dtype=dtype)
        model.eval()
    except Exception as e:
        print(f"[Bruxos Bernini Neural Upscale] FALHOU carregar o checkpoint ({e}). "
              f"Arquivo pode estar corrompido -- apague {_CKPT_PATH} e tente de novo.", flush=True)
        return None

    _modelo_cache["model"] = model
    _modelo_cache["device"] = device
    _modelo_cache["dtype"] = dtype
    print("[Bruxos Bernini Neural Upscale] modelo carregado (experimental/WIP, ver docstring do modulo).",
          flush=True)
    return model


def disponivel():
    """True se torch existe (nao garante que o download vai funcionar)."""
    return _OK


def neural_upscale(samples, nh, nw, chunk=16):
    """
    samples: tensor [B,C,T,H,W] (latente do Wan/Bernini, 16 canais).
    nh, nw : altura/largura NOVAS do latente (ja arredondadas/pares).
    Retorna tensor [B,C,T,nh,nw] no mesmo dtype/device de entrada, ou
    None se o modelo nao pode ser carregado (quem chama deve cair pro
    metodo bicubic nesse caso).
    """
    if not _OK:
        return None

    import comfy.model_management as model_management
    device = model_management.get_torch_device()
    dtype = torch.float16 if model_management.should_use_fp16() else torch.float32

    model = _carregar_modelo(device, dtype)
    if model is None:
        return None

    B, C, T, H, W = (int(v) for v in samples.shape)
    escala_h = float(nh) / float(H)
    escala_w = float(nw) / float(W)
    # a rede so aprendeu escala uniforme -- se h/w pedirem escalas
    # diferentes (raro, so acontece com arredondamento de video bem
    # estreito), usa a media e deixa o F.interpolate final acertar o resto.
    escala = (escala_h + escala_w) / 2.0

    x = samples.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).to(device=device, dtype=dtype)

    saidas = []
    model.to(device=device)
    try:
        with torch.no_grad():
            for i in range(0, x.shape[0], max(1, int(chunk))):
                bloco = x[i:i + chunk] * SCALE_FACTOR
                out = model(bloco, scale=escala)
                out = out / SCALE_FACTOR
                if out.shape[2] != nh or out.shape[3] != nw:
                    out = F.interpolate(out, size=(nh, nw), mode="bilinear", align_corners=False)
                saidas.append(out)
    except Exception as e:
        print(f"[Bruxos Bernini Neural Upscale] FALHOU rodar a rede ({e}). Caindo pro metodo bicubic.",
              flush=True)
        return None
    finally:
        model.to(device=model_management.vae_offload_device())

    y = torch.cat(saidas, dim=0)
    y = y.reshape(B, T, C, nh, nw).permute(0, 2, 1, 3, 4).to(dtype=samples.dtype, device=samples.device).contiguous()
    return y
