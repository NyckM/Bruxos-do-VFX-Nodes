# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — MiniMax H3: mascara pra INPAINT e OUTPAINT
===========================================================
Dois nodes que fecham o ciclo de edicao no H3:

  1. `H3 Outpaint · Expandir Quadro`  -> aumenta a moldura e ja devolve a
     mascara da area nova (por frame). Saida: frames + mask.
  2. `H3 Mascara -> Latente (inpaint)` -> pega uma MASK em pixels e escreve
     ela como `noise_mask` no LATENTE do H3, no shape latente correto.

COMO O INPAINT FUNCIONA NO COMFYUI (e por que este node e necessario):
    O sampler nao recebe "mascara" em pixels. Ele le a chave `noise_mask`
    dentro do dict do LATENT e, a cada passo, mistura o resultado com o
    latente original:
        area com mascara=1 -> o modelo GERA (muda)
        area com mascara=0 -> fica o ORIGINAL (preserva)
    Mas o latente e MUITO menor que a imagem (o VAE comprime no espaco e no
    tempo). Entao a mascara precisa ser reamostrada pro tamanho do latente.
    Este node faz isso lendo o shape REAL do latente que voce passou -- sem
    chutar o fator de compressao do H3.

FLUXOS:

  INPAINT (trocar/remover algo, com a mascara seguindo o objeto):
      AutoEdit Mask (SAM3)  ─ mask ──┐
      Load Video ─ frames ───────────┼─> H3 Encode ─ latent ─> H3 Mascara -> Latente ─> Sampler
      VAEs (video + audio) ──────────┘                                                  (denoise 0.6-1.0)

  OUTPAINT (expandir o quadro):
      Load Video ─> H3 Outpaint ─┬─ frames ─> H3 Encode ─ latent ─> H3 Mascara -> Latente ─> Sampler
                                 └─ mask ────────────────────────────^            (denoise 1.0)
"""

import logging

try:
    import torch
    import torch.nn.functional as F
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/MiniMax H3"


# ---------------------------------------------------------------------------
def _norm_mask(m):
    """Aceita MASK [H,W] / [T,H,W] ou IMAGE [T,H,W,C] -> [T,H,W] em 0..1."""
    if m is None:
        return None
    if m.dim() == 4:                      # IMAGE colorida
        m = m[..., :3].amax(dim=-1)
    elif m.dim() == 2:                    # [H,W]
        m = m.unsqueeze(0)
    return m.float().clamp(0, 1)


def _grow_blur(m, grow=0, blur=0):
    """m [T,H,W] -> dilata(+)/contrai(-) e suaviza a borda, em pixels."""
    x = m.unsqueeze(1)                    # [T,1,H,W]
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


def _video_de(samples):
    """Pega o componente de VIDEO do container do H3 (NestedTensor/lista/tensor)."""
    if torch.is_tensor(samples):
        return samples
    t = getattr(samples, "tensors", None)
    if isinstance(t, (list, tuple)) and t:
        return t[0]
    if isinstance(samples, (list, tuple)) and samples:
        return samples[0]
    return None


# ===========================================================================
# 1) OUTPAINT — expande a moldura e devolve a mascara da area nova
# ===========================================================================
class BruxosH3Outpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Frames do video original (todos os frames de uma vez)."}),
                "esquerda": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "Pixels a acrescentar A ESQUERDA."}),
                "direita": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "Pixels a acrescentar A DIREITA."}),
                "cima": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "Pixels a acrescentar EM CIMA."}),
                "baixo": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "Pixels a acrescentar EMBAIXO."}),
            },
            "optional": {
                "feather": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1, "tooltip":
                    "Suaviza a borda ENTRE o quadro original e a area nova, em pixels. Evita emenda dura e ajuda o "
                    "modelo a costurar a continuacao. 16-40 costuma ficar bom. 0 = borda seca."}),
                "invadir_original": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1, "tooltip":
                    "Quantos pixels da imagem ORIGINAL entram na mascara (a mascara avanca pra dentro). "
                    "Uma faixinha ajuda o modelo a casar textura e luz na junta. 0 = mascara so na area nova."}),
                "alinhar": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1, "tooltip":
                    "Arredonda o tamanho FINAL pra multiplo deste valor (o VAE precisa disso). 16 e seguro pro H3. "
                    "O ajuste e feito acrescentando alguns pixels a direita/embaixo."}),
                "preenchimento": (["cinza", "replicar_borda", "espelhado"], {"default": "cinza", "tooltip":
                    "O que colocar na area nova ANTES de gerar. cinza = 0.5 neutro (padrao do ComfyUI, previsivel). "
                    "replicar_borda = estica os pixels da beirada (da mais pista de cor/luz ao modelo). "
                    "espelhado = reflete a imagem pra fora."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("frames", "mask", "info")
    OUTPUT_TOOLTIPS = (
        "Frames JA expandidos -> ligue no 'frames' do H3 Encode.",
        "Mascara da area a gerar (1 = gera, 0 = preserva) -> ligue no 'H3 Mascara -> Latente'.",
        "Tamanho antes/depois e avisos de alinhamento.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "H3 Outpaint · Expandir Quadro (Bruxos): aumenta a moldura do video e devolve JUNTO a mascara da area nova, "
        "ja com feather na junta e alinhamento de resolucao pro VAE. Funciona no batch inteiro de frames. "
        "Saida: frames expandidos + mask -> H3 Encode -> H3 Mascara -> Latente -> Sampler (denoise 1.0)."
    )

    def run(self, frames, esquerda, direita, cima, baixo,
            feather=24, invadir_original=8, alinhar=16, preenchimento="cinza"):
        if not _OK:
            raise RuntimeError("[Bruxos H3 Outpaint] torch indisponivel.")
        if frames.ndim != 4:
            raise ValueError(f"[Bruxos H3 Outpaint] 'frames' precisa ser IMAGE [T,H,W,C]; veio {tuple(frames.shape)}.")

        T, H, W, C = (int(v) for v in frames.shape)
        l, r, t_, b = (max(0, int(v)) for v in (esquerda, direita, cima, baixo))
        if l == r == t_ == b == 0:
            raise ValueError(
                "[Bruxos H3 Outpaint] todos os lados estao em 0 -- nao ha nada pra expandir. "
                "Coloque pixels em pelo menos um lado (esquerda/direita/cima/baixo)."
            )

        nH, nW = H + t_ + b, W + l + r
        # alinhamento: acrescenta o resto a direita/embaixo
        a = max(1, int(alinhar))
        extra_w = (-nW) % a
        extra_h = (-nH) % a
        r += extra_w
        b += extra_h
        nH, nW = H + t_ + b, W + l + r

        # ---- canvas -------------------------------------------------------
        x = frames.permute(0, 3, 1, 2)                       # [T,C,H,W]
        if preenchimento == "replicar_borda":
            out = F.pad(x, (l, r, t_, b), mode="replicate")
        elif preenchimento == "espelhado":
            # reflect nao aceita padding >= dimensao; cai pra replicate se passar
            if l < W and r < W and t_ < H and b < H:
                out = F.pad(x, (l, r, t_, b), mode="reflect")
            else:
                print("[Bruxos H3 Outpaint] padding maior que a imagem -> 'espelhado' virou 'replicar_borda'.", flush=True)
                out = F.pad(x, (l, r, t_, b), mode="replicate")
        else:  # cinza
            out = F.pad(x, (l, r, t_, b), mode="constant", value=0.5)
        novos_frames = out.permute(0, 2, 3, 1).clamp(0, 1)    # [T,nH,nW,C]

        # ---- mascara: 1 na area nova, 0 no original -----------------------
        m = torch.ones((T, nH, nW), dtype=torch.float32, device=frames.device)
        inv = max(0, int(invadir_original))
        y0, y1 = t_ + inv, t_ + H - inv
        x0, x1 = l + inv, l + W - inv
        if y1 > y0 and x1 > x0:
            m[:, y0:y1, x0:x1] = 0.0
        else:
            print("[Bruxos H3 Outpaint] AVISO: 'invadir_original' grande demais -- a mascara cobriu o quadro inteiro.", flush=True)

        if int(feather) > 0:
            m = _grow_blur(m, grow=0, blur=int(feather))

        avisos = ""
        if extra_w or extra_h:
            avisos = f" | alinhado p/ multiplo de {a}: +{extra_w}px direita, +{extra_h}px embaixo"
        info = (f"{W}x{H} -> {nW}x{nH} ({T} frames) | bordas L{l} R{r} T{t_} B{b} | "
                f"feather {feather} | invade {inv} | {preenchimento}{avisos}")
        print(f"[Bruxos H3 Outpaint] {info}", flush=True)
        return (novos_frames, m, info)


# ===========================================================================
# 2) INPAINT — mascara de pixels -> noise_mask no latente do H3
# ===========================================================================
class BruxosH3MaskToLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "O latente vindo do 'H3 Encode Video -> Latente'."}),
                "mask": ("MASK", {"tooltip":
                    "Mascara em PIXELS. 1 (branco) = o modelo GERA ali; 0 (preto) = PRESERVA o original.\n"
                    "Aceita uma mascara por frame [T,H,W] -- pode ligar direto a saida do 'AutoEdit Mask (SAM3)', "
                    "que ja rastreia o objeto ao longo do video. Se vier uma mascara so, ela e repetida em todos os frames."}),
            },
            "optional": {
                "inverter": ("BOOLEAN", {"default": False, "tooltip":
                    "Troca o que gera pelo que preserva. Ligue se a sua mascara veio ao contrario "
                    "(objeto preto e fundo branco)."}),
                "mask_grow": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1, "tooltip":
                    "Dilata (+) ou contrai (-) a mascara, em pixels. SUBA quando o objeto novo for maior que o antigo, "
                    "ou pra pegar sombra/reflexo em volta."}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1, "tooltip":
                    "Suaviza a borda (feather), em pixels. Evita emenda dura entre gerado e original. 8-24 costuma ir bem."}),
                "forca": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "Multiplica a mascara inteira. 1.0 = dentro da mascara o modelo tem liberdade total. "
                    "Menor (0.6-0.8) deixa o original 'vazar' um pouco e o resultado fica mais preso a cena -- "
                    "util quando o inpaint esta inventando demais."}),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK", "STRING")
    RETURN_NAMES = ("latent", "mask_latente", "info")
    OUTPUT_TOOLTIPS = (
        "Latente com o 'noise_mask' escrito -> ligue no 'latent_image' do SamplerCustomAdvanced.",
        "A mascara JA no tamanho do latente (pra voce conferir num Preview o que o modelo vai realmente ver).",
        "Shapes de entrada/saida e avisos.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "H3 Mascara -> Latente (Bruxos): converte uma MASK em pixels para o 'noise_mask' do LATENTE do H3, "
        "reamostrando pro shape latente REAL (le o shape do proprio latente -- nao chuta o fator de compressao do VAE). "
        "1 = gera, 0 = preserva. Aceita mascara por frame (SAM3) ou unica. Serve tanto pra inpaint quanto pra outpaint."
    )

    def run(self, latent, mask, inverter=False, mask_grow=0, mask_blur=8, forca=1.0):
        if not _OK:
            raise RuntimeError("[Bruxos H3 Mask] torch indisponivel.")

        s = latent.get("samples", latent) if isinstance(latent, dict) else latent
        video = _video_de(s)
        if video is None:
            raise ValueError(
                "[Bruxos H3 Mask] nao achei o componente de video no latente. "
                "Ponha o 'Inspetor de LATENT (Bruxos)' no fio e me mande o que ele imprime."
            )

        # shape latente: [B,C,T,H,W] (5D) ou [B,C,H,W] (4D)
        if video.ndim == 5:
            _B, _C, Tl, Hl, Wl = (int(v) for v in video.shape)
        elif video.ndim == 4:
            _B, _C, Hl, Wl = (int(v) for v in video.shape)
            Tl = 1
        else:
            raise ValueError(f"[Bruxos H3 Mask] shape de latente inesperado: {tuple(video.shape)}. "
                             f"Use o Inspetor de LATENT e me mande o resultado.")

        m = _norm_mask(mask)
        Tm, Hm, Wm = (int(v) for v in m.shape)
        print(f"[Bruxos H3 Mask] mascara {Tm}x{Hm}x{Wm} (pixels) -> latente T={Tl} {Hl}x{Wl}", flush=True)

        if inverter:
            m = 1.0 - m
        if int(mask_grow) != 0 or int(mask_blur) > 0:
            m = _grow_blur(m, int(mask_grow), int(mask_blur))

        # ---- reamostra ESPACIALMENTE pro tamanho do latente ---------------
        if (Hm, Wm) != (Hl, Wl):
            m = F.interpolate(m.unsqueeze(1), size=(Hl, Wl), mode="bilinear", align_corners=False).squeeze(1)

        # ---- alinha TEMPORALMENTE (o VAE comprime o tempo) ----------------
        # Mapeia cada slot latente pro frame de pixel correspondente e pega o
        # MAXIMO da janela: se o objeto aparece em qualquer frame daquele
        # bloco, o slot latente entra na mascara (melhor sobrar que faltar).
        if Tm == 1 and Tl > 1:
            m = m.repeat(Tl, 1, 1)
        elif Tm != Tl:
            blocos = []
            for i in range(Tl):
                a = int(round(i * Tm / Tl))
                b = max(a + 1, int(round((i + 1) * Tm / Tl)))
                blocos.append(m[a:min(b, Tm)].amax(dim=0))
            m = torch.stack(blocos, dim=0)
            print(f"[Bruxos H3 Mask] tempo comprimido {Tm} -> {Tl} slots (maximo por bloco)", flush=True)

        f = float(forca)
        if f != 1.0:
            m = m * f

        m = m.clamp(0, 1)

        # noise_mask no formato que o sampler espera: [T,1,H,W]
        noise_mask = m.unsqueeze(1).to(dtype=torch.float32)

        cob = float(m.mean()) * 100.0
        avisos = ""
        if cob < 0.05:
            avisos = " | ATENCAO: mascara praticamente VAZIA -- nada vai mudar. Confira o 'inverter'."
        elif cob > 97.0:
            avisos = " | ATENCAO: mascara cobre quase TUDO -- vira geracao normal, sem preservar nada."

        out = dict(latent) if isinstance(latent, dict) else {"samples": s}
        out["noise_mask"] = noise_mask

        # ---- AVISO IMPORTANTE sobre o container do H3 ---------------------
        # O inpaint do ComfyUI funciona misturando, a CADA passo:
        #     x = x * noise_mask + latente_original * (1 - noise_mask)
        # Isso pressupoe que `x` seja um TENSOR comum. O latente do H3 nao e:
        # e um container com DOIS componentes (video + audio). Se a versao do
        # ComfyUI nao souber multiplicar esse container por uma mascara, o
        # inpaint ou explode com erro, ou -- pior -- e ignorado em silencio.
        if not torch.is_tensor(s):
            print(
                f"[Bruxos H3 Mask] NOTA: o latente e um container "
                f"'{type(s).__name__}' (video+audio), nao um tensor simples.\n"
                f"[Bruxos H3 Mask] O caminho de inpaint do ComfyUI mistura "
                f"'x * noise_mask' a cada passo, e isso pode nao funcionar com "
                f"container. COMO SABER SE PEGOU:\n"
                f"[Bruxos H3 Mask]   - deu erro de shape/tipo no sampler -> nao pegou;\n"
                f"[Bruxos H3 Mask]   - rodou, mas a area FORA da mascara tambem "
                f"mudou -> foi ignorado em silencio (o caso traicoeiro);\n"
                f"[Bruxos H3 Mask]   - rodou e so a area da mascara mudou -> pegou.\n"
                f"[Bruxos H3 Mask] Se nao pegar, use o plano B: gere normal e "
                f"componha com 'ImageCompositeMasked' depois do VAE Decode.",
                flush=True)

        info = (f"noise_mask {tuple(noise_mask.shape)} | latente T={Tl} {Hl}x{Wl} | "
                f"cobertura {cob:.1f}% | forca {f:.2f}{avisos}")
        print(f"[Bruxos H3 Mask] {info}", flush=True)
        return (out, m, info)


NODE_CLASS_MAPPINGS = {
    "BruxosH3Outpaint": BruxosH3Outpaint,
    "BruxosH3MaskToLatent": BruxosH3MaskToLatent,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3Outpaint": "H3 Outpaint · Expandir Quadro (Bruxos)",
    "BruxosH3MaskToLatent": "H3 Mascara -> Latente · inpaint (Bruxos)",
}
