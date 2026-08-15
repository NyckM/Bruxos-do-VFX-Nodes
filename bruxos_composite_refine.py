# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Composite & Refine: colar referencia no video + mascara pronta
==============================================================================
Serve pra modelos que NAO tem entrada de imagem de referencia (MiniMax H3, Wan
sem VACE, etc). A ideia e a tecnica classica de VFX "composite-and-refine":

    voce fornece o CONTEUDO pelo pixel (colagem tosca),
    o modelo fornece a INTEGRACAO (luz, borda, grao, motion blur).

Fluxo:
    1. Cola o elemento de referencia nos frames, na posicao/escala que voce quer
       -- pode ficar feio, com borda dura e luz errada, nao importa.
    2. Este node ja devolve a MASCARA daquela regiao (com feather e uma margem
       pra fora, que e onde o modelo costura a colagem na cena).
    3. Encode -> noise_mask -> sampler com denoise 0.5-0.7.
       O modelo redesenha aquela regiao PARTINDO da sua colagem, entao ele
       mantem o que voce colou e conserta a integracao.

MODOS DE MOVIMENTO:
    travado   -> a colagem fica parada no quadro (bom pra camera parada, ou
                 quando o elemento e um objeto fixo no enquadramento).
    trajetoria-> a colagem anda em linha reta do ponto inicial ao final, com
                 escala e rotacao interpoladas. Serve pra acompanhar um
                 movimento simples de camera ou do objeto.
    mascara   -> a colagem SEGUE uma mascara de rastreio (ex.: a saida do
                 'AutoEdit Mask (SAM3)'): a cada frame ela e centralizada e
                 escalada pra caber na caixa daquela mascara. E o modo que
                 acompanha movimento de verdade.

Saidas: frames com a colagem + mask -> ligue direto no encode e no node de
mascara do seu modelo.
"""

import logging

try:
    import torch
    import torch.nn.functional as F
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Composite"


# ---------------------------------------------------------------------------
def _norm_mask(m):
    """MASK [H,W]/[T,H,W] ou IMAGE [T,H,W,C] -> [T,H,W] em 0..1."""
    if m is None:
        return None
    if m.dim() == 4:
        m = m[..., :3].amax(dim=-1)
    elif m.dim() == 2:
        m = m.unsqueeze(0)
    return m.float().clamp(0, 1)


def _feather(m, grow=0, blur=0):
    """m [T,H,W] -> dilata(+)/contrai(-) e suaviza, em pixels."""
    x = m.unsqueeze(1)
    g = int(grow)
    if g > 0:
        x = F.max_pool2d(x, kernel_size=g * 2 + 1, stride=1, padding=g)
    elif g < 0:
        a = -g
        x = -F.max_pool2d(-x, kernel_size=a * 2 + 1, stride=1, padding=a)
    b = int(blur)
    if b > 0:
        k = b * 2 + 1
        co = torch.arange(k, dtype=torch.float32, device=x.device) - b
        sig = b * 0.5 + 1e-6
        g1 = torch.exp(-(co ** 2) / (2 * sig * sig))
        g1 = g1 / g1.sum()
        x = F.conv2d(x, g1.view(1, 1, 1, k), padding=(0, b))
        x = F.conv2d(x, g1.view(1, 1, k, 1), padding=(b, 0))
    return x.squeeze(1).clamp(0, 1)


def _bbox(m1, minimo=8):
    """Caixa (x0,y0,x1,y1) da parte acesa de uma mascara [H,W]. None se vazia."""
    ys = torch.nonzero(m1.amax(dim=1) > 0.02).flatten()
    xs = torch.nonzero(m1.amax(dim=0) > 0.02).flatten()
    if ys.numel() == 0 or xs.numel() == 0:
        return None
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    if (x1 - x0) < minimo or (y1 - y0) < minimo:
        return None
    return x0, y0, x1, y1


def _transformar(ref_rgb, ref_a, larg, alt, ang):
    """Redimensiona (e opcionalmente gira) o elemento. ref_* sao [1,C,h,w]."""
    larg, alt = max(1, int(larg)), max(1, int(alt))
    rgb = F.interpolate(ref_rgb, size=(alt, larg), mode="bilinear", align_corners=False)
    a = F.interpolate(ref_a, size=(alt, larg), mode="bilinear", align_corners=False)
    if abs(float(ang)) > 1e-3:
        th = torch.tensor(float(ang) * 3.14159265 / 180.0)
        cos, sin = torch.cos(th), torch.sin(th)
        # grid_sample gira em torno do centro; o alpha gira junto
        theta = torch.tensor([[[cos, -sin, 0.0], [sin, cos, 0.0]]], dtype=rgb.dtype, device=rgb.device)
        grid = F.affine_grid(theta, rgb.shape, align_corners=False)
        rgb = F.grid_sample(rgb, grid, align_corners=False, padding_mode="zeros")
        a = F.grid_sample(a, grid, align_corners=False, padding_mode="zeros")
    return rgb, a


class BruxosCompositeRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Frames do video de destino (onde o elemento vai ser colado)."}),
                "referencia": ("IMAGE", {"tooltip":
                    "A imagem do elemento a inserir. Se ela tiver canal alpha (RGBA, ex.: saida de um RMBG/remove-fundo), "
                    "o alpha e usado como recorte automatico. Senao, ligue 'referencia_mask'."}),
            },
            "optional": {
                "referencia_mask": ("MASK", {"tooltip":
                    "[opcional] Recorte do elemento (1 = elemento, 0 = fundo). Use quando a referencia nao tiver alpha. "
                    "Pode vir de um RMBG, do SAM3 na imagem, ou desenhada na mao."}),
                "modo": (["travado", "trajetoria", "mascara"], {"default": "travado", "tooltip":
                    "travado = colagem parada no quadro (camera parada / objeto fixo no enquadramento).\n"
                    "trajetoria = anda em linha reta do ponto inicial ao final, com escala/rotacao interpoladas.\n"
                    "mascara = SEGUE a 'mask_rastreio' (ex.: saida do AutoEdit Mask/SAM3) -- e o modo que acompanha "
                    "movimento de verdade."}),
                "mask_rastreio": ("MASK", {"tooltip":
                    "[modo=mascara] Mascara do objeto RASTREADA no video ([T,H,W]). A cada frame a colagem e "
                    "centralizada e escalada pra caber na caixa dessa mascara. Ligue aqui a saida 'mask' do "
                    "'AutoEdit Mask · SAM3 rastreado'."}),
                "x": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip":
                    "Posicao X do CENTRO da colagem, em pixels (0 = centro do quadro). Positivo vai pra direita.\n"
                    "[modo=trajetoria] este e o ponto INICIAL."}),
                "y": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip":
                    "Posicao Y do CENTRO da colagem (0 = centro do quadro). Positivo desce.\n"
                    "[modo=trajetoria] ponto INICIAL."}),
                "escala": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01, "tooltip":
                    "Tamanho do elemento (1.0 = tamanho original da imagem de referencia).\n"
                    "[modo=trajetoria] escala INICIAL. [modo=mascara] multiplica o tamanho da caixa rastreada."}),
                "rotacao": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "tooltip":
                    "Rotacao em graus. [modo=trajetoria] rotacao INICIAL."}),
                "x_final": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip":
                    "[modo=trajetoria] Posicao X no ULTIMO frame. A colagem interpola de x ate x_final."}),
                "y_final": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip":
                    "[modo=trajetoria] Posicao Y no ULTIMO frame."}),
                "escala_final": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01, "tooltip":
                    "[modo=trajetoria] Escala no ULTIMO frame (da pra simular aproximacao/afastamento)."}),
                "rotacao_final": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5, "tooltip":
                    "[modo=trajetoria] Rotacao no ULTIMO frame."}),
                "margem_mask": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1, "tooltip":
                    "Quanto a MASCARA cresce PARA FORA do elemento colado, em pixels. E AQUI que o modelo costura a "
                    "colagem na cena (sombra, contato, reflexo, borda). Margem 0 = ele so pode mexer dentro do recorte "
                    "e a colagem fica com cara de adesivo. 20-40 costuma ir bem."}),
                "feather_mask": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1, "tooltip":
                    "Suaviza a borda da mascara. Evita emenda dura entre o que foi gerado e o original."}),
                "opacidade": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "Opacidade da colagem. 1.0 = cola opaco. Menor deixa a cena original transparecer -- as vezes "
                    "ajuda o modelo a integrar melhor, porque a colagem fica menos 'dura'."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("frames", "mask", "preview", "info")
    OUTPUT_TOOLTIPS = (
        "Frames com o elemento colado -> ligue no 'frames' do encode (H3 Encode, VAE Encode, etc).",
        "Mascara da regiao a refinar (colagem + margem) -> ligue no node de mascara do seu modelo.",
        "Frames com a mascara pintada de rosa por cima -- CONFIRA AQUI antes de rodar o render.",
        "Resumo do que foi colado e onde.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Composite & Refine (Bruxos): cola uma imagem de referencia nos frames do video e devolve a mascara daquela "
        "regiao ja pronta pro inpaint. Serve pra dar 'imagem de referencia' a modelos que NAO tem essa entrada "
        "(MiniMax H3, Wan sem VACE): voce fornece o CONTEUDO pelo pixel e o modelo, com denoise 0.5-0.7, fornece a "
        "INTEGRACAO (luz, borda, grao, motion blur). A colagem pode ficar travada, seguir uma trajetoria reta, ou "
        "SEGUIR uma mascara rastreada do SAM3."
    )

    def run(self, frames, referencia, referencia_mask=None, modo="travado", mask_rastreio=None,
            x=0, y=0, escala=1.0, rotacao=0.0,
            x_final=0, y_final=0, escala_final=1.0, rotacao_final=0.0,
            margem_mask=24, feather_mask=12, opacidade=1.0):
        if not _OK:
            raise RuntimeError("[Bruxos Composite] torch indisponivel.")
        if frames.ndim != 4:
            raise ValueError(f"[Bruxos Composite] 'frames' precisa ser IMAGE [T,H,W,C]; veio {tuple(frames.shape)}.")

        T, H, W, _ = (int(v) for v in frames.shape)
        dev = frames.device

        # ---- referencia + alpha ------------------------------------------
        ref = referencia[0] if referencia.ndim == 4 else referencia          # [h,w,C]
        rh, rw = int(ref.shape[0]), int(ref.shape[1])
        ref_rgb = ref[..., :3].permute(2, 0, 1).unsqueeze(0).float()          # [1,3,h,w]
        if referencia.shape[-1] == 4:
            alpha = ref[..., 3:4].permute(2, 0, 1).unsqueeze(0).float()
            fonte_alpha = "alpha da imagem (RGBA)"
        elif referencia_mask is not None:
            am = _norm_mask(referencia_mask)[0]
            if am.shape != (rh, rw):
                am = F.interpolate(am[None, None], size=(rh, rw), mode="bilinear", align_corners=False)[0, 0]
            alpha = am[None, None].float()
            fonte_alpha = "referencia_mask"
        else:
            alpha = torch.ones((1, 1, rh, rw), dtype=torch.float32, device=ref.device)
            fonte_alpha = "sem recorte (retangulo inteiro)"
        ref_rgb, alpha = ref_rgb.to(dev), alpha.to(dev)

        # ---- rastreio (modo=mascara) --------------------------------------
        trilha = None
        if modo == "mascara":
            if mask_rastreio is None:
                raise ValueError(
                    "[Bruxos Composite] modo='mascara' exige a entrada 'mask_rastreio'. "
                    "Ligue a saida 'mask' do 'AutoEdit Mask · SAM3 rastreado' -- ou troque o modo pra "
                    "'travado'/'trajetoria'."
                )
            trilha = _norm_mask(mask_rastreio).to(dev)
            if trilha.shape[0] == 1 and T > 1:
                trilha = trilha.repeat(T, 1, 1)
            if trilha.shape[-2:] != (H, W):
                trilha = F.interpolate(trilha.unsqueeze(1), size=(H, W), mode="bilinear",
                                       align_corners=False).squeeze(1)

        saida = frames.clone()
        mask_out = torch.zeros((T, H, W), dtype=torch.float32, device=dev)
        pulados = 0

        for i in range(T):
            f = 0.0 if T <= 1 else i / (T - 1)

            if modo == "trajetoria":
                cx = W * 0.5 + (x + (x_final - x) * f)
                cy = H * 0.5 + (y + (y_final - y) * f)
                sc = escala + (escala_final - escala) * f
                ang = rotacao + (rotacao_final - rotacao) * f
                lw, lh = rw * sc, rh * sc
            elif modo == "mascara":
                bb = _bbox(trilha[i])
                if bb is None:
                    pulados += 1
                    continue          # objeto sumiu neste frame -> nao cola nada
                x0, y0, x1, y1 = bb
                cx, cy = (x0 + x1) * 0.5 + x, (y0 + y1) * 0.5 + y
                bw, bh = (x1 - x0), (y1 - y0)
                # encaixa mantendo a proporcao da referencia
                sc = min(bw / max(rw, 1), bh / max(rh, 1)) * float(escala)
                lw, lh = rw * sc, rh * sc
                ang = rotacao
            else:  # travado
                cx, cy = W * 0.5 + x, H * 0.5 + y
                sc = float(escala)
                lw, lh = rw * sc, rh * sc
                ang = rotacao

            lw, lh = max(1, int(round(lw))), max(1, int(round(lh)))
            rgb_i, a_i = _transformar(ref_rgb, alpha, lw, lh, ang)
            a_i = a_i * float(opacidade)

            # destino no quadro (canto superior esquerdo)
            px, py = int(round(cx - lw * 0.5)), int(round(cy - lh * 0.5))
            dx0, dy0 = max(0, px), max(0, py)
            dx1, dy1 = min(W, px + lw), min(H, py + lh)
            if dx1 <= dx0 or dy1 <= dy0:
                pulados += 1
                continue              # totalmente fora do quadro
            sx0, sy0 = dx0 - px, dy0 - py
            sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)

            src = rgb_i[0, :, sy0:sy1, sx0:sx1].permute(1, 2, 0)     # [h,w,3]
            av = a_i[0, 0, sy0:sy1, sx0:sx1].unsqueeze(-1)           # [h,w,1]
            base = saida[i, dy0:dy1, dx0:dx1, :3]
            saida[i, dy0:dy1, dx0:dx1, :3] = base * (1 - av) + src * av
            mask_out[i, dy0:dy1, dx0:dx1] = torch.maximum(
                mask_out[i, dy0:dy1, dx0:dx1], av[..., 0]
            )

        # ---- mascara final: cresce pra fora + feather ---------------------
        # binariza antes de crescer: a margem tem que envolver o elemento
        # inteiro, nao so a parte mais opaca dele.
        mb = (mask_out > 0.02).float()
        mask_final = _feather(mb, grow=int(margem_mask), blur=int(feather_mask))

        # preview: mascara em rosa por cima
        tint = torch.tensor([1.0, 0.15, 0.45], dtype=saida.dtype, device=dev).view(1, 1, 1, 3)
        mv = mask_final.unsqueeze(-1).to(saida.dtype)
        preview = (saida[..., :3] * (1 - mv * 0.5) + tint * (mv * 0.5)).clamp(0, 1)

        cob = float(mask_final.mean()) * 100.0
        avisos = ""
        if pulados:
            avisos += f" | {pulados}/{T} frame(s) sem colagem (fora do quadro ou objeto ausente no rastreio)"
        if cob < 0.05:
            avisos += " | ATENCAO: mascara vazia -- confira posicao/escala."
        elif cob > 90.0:
            avisos += " | ATENCAO: mascara cobre quase tudo -- reduza escala ou margem_mask."

        info = (f"modo={modo} | ref {rw}x{rh} ({fonte_alpha}) | quadro {W}x{H} x{T}f | "
                f"margem {margem_mask} feather {feather_mask} | cobertura {cob:.1f}%{avisos}")
        print(f"[Bruxos Composite] {info}", flush=True)
        return (saida, mask_final, preview, info)


NODE_CLASS_MAPPINGS = {"BruxosCompositeRefine": BruxosCompositeRefine}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosCompositeRefine": "Composite & Refine · referencia -> video + mascara (Bruxos)"}
