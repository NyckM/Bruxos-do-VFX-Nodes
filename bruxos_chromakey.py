# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Chromakey: fundo -> mascara + referencia de fundo
=================================================================
Feito pro caso: "tenho video em chroma, quero recortar SO o fundo, mandar uma
imagem de fundo de referencia, e o modelo junta tudo".

O node faz, em uma passada:
  1. TIRA a chave do croma (verde/azul) e gera a mascara do FUNDO
     (1 = fundo -> o modelo pode gerar ali; 0 = sujeito -> preserva).
  2. REMOVE O SPILL -- o esverdeado que contamina a borda e as partes claras
     do sujeito. Este e o passo que quase todo mundo pula e que faz a
     composicao parecer falsa: sem ele, sobra uma franja verde no sujeito que
     o modelo NAO vai consertar (ela esta fora da mascara, entao e preservada).
  3. COMPOE a sua imagem de referencia atras do sujeito.

O que sai daqui vai pro modelo assim:
    frames (sujeito + fundo de referencia ja colado)  -> encode
    mask   (so o fundo)                               -> noise_mask
    denoise 0.3-0.5  -> o modelo HARMONIZA a composicao (luz, borda, grao,
                        perspectiva, sombra de contato) em vez de inventar
                        um fundo do zero.

Por que compor o fundo em vez de deixar verde: se voce mandar o verde chapado
pro latente, ele vaza pro resultado -- o modelo enxerga verde e tende a puxar
a cor pra la. Com a referencia ja colada, ele so precisa integrar.
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


def _feather(m, grow=0, blur=0):
    """m [T,H,W] -> dilata(+)/contrai(-) e suaviza a borda, em pixels."""
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


def _ajustar_fundo(bg, H, W, modo):
    """bg [1,h,w,3] -> [1,H,W,3] no enquadramento pedido."""
    x = bg.permute(0, 3, 1, 2)                     # [1,3,h,w]
    h, w = int(x.shape[2]), int(x.shape[3])
    if modo == "esticar":
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    else:
        # cobrir = preenche o quadro e corta a sobra; caber = cabe inteiro (com barras)
        s = max(W / w, H / h) if modo == "cobrir" else min(W / w, H / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        if modo == "cobrir":
            y0, x0 = max(0, (nh - H) // 2), max(0, (nw - W) // 2)
            x = x[:, :, y0:y0 + H, x0:x0 + W]
            if x.shape[2] != H or x.shape[3] != W:
                x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        else:
            cvs = torch.zeros((1, 3, H, W), dtype=x.dtype, device=x.device)
            y0, x0 = (H - nh) // 2, (W - nw) // 2
            cvs[:, :, y0:y0 + nh, x0:x0 + nw] = x
            x = cvs
    return x.permute(0, 2, 3, 1).clamp(0, 1)


class BruxosChromaKeyBG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Video em chroma (fundo verde/azul), todos os frames."}),
            },
            "optional": {
                "fundo": ("IMAGE", {"tooltip":
                    "Sua imagem de REFERENCIA de fundo. Ela e composta ATRAS do sujeito. Se deixar vazio, a area do "
                    "fundo e preenchida com cinza neutro (0.5) e o modelo gera o fundo so pelo prompt/ref_images."}),
                "matte": ("MASK", {"tooltip":
                    "[modo=matte_externo] Seu matte pronto, em tons de cinza. Aceita uma por frame ([T,H,W]) ou uma so "
                    "(repetida). Tambem aceita IMAGE em preto e branco.\n"
                    "(Entrada de FIO, nao widget -- nao afeta a ordem dos valores salvos.)"}),
                "cor_chave": (["verde", "azul", "personalizada"], {"default": "verde", "tooltip":
                    "[modo=chroma] Cor do fundo do croma. 'personalizada' usa os campos R/G/B abaixo."}),
                "chave_r": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1, "tooltip": "[personalizada] R da cor do fundo (0-255)."}),
                "chave_g": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "tooltip": "[personalizada] G da cor do fundo."}),
                "chave_b": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1, "tooltip": "[personalizada] B da cor do fundo."}),
                "tolerancia": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.02, "tooltip":
                    "Quanto de dominancia da cor-chave ja conta como FUNDO. SUBA se sobrar verde no fundo (buracos, "
                    "cantos mal iluminados). DESCA se o sujeito estiver sendo comido (cabelo, roupa esverdeada)."}),
                "suavidade": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.02, "tooltip":
                    "Largura da transicao entre sujeito e fundo. Maior = borda mais macia (bom pra cabelo e "
                    "semitransparencia). Menor = recorte duro."}),
                "despill": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip":
                    "Remove o esverdeado que contamina a BORDA e as partes claras do sujeito. DEIXE EM 1.0. "
                    "Sem isso sobra uma franja verde no sujeito -- e ela esta FORA da mascara, entao o modelo nao "
                    "conserta: fica verde no resultado final."}),
                "contrair": ("INT", {"default": 2, "min": -64, "max": 64, "step": 1, "tooltip":
                    "Encolhe (+) a mascara do fundo pra dentro do fundo, protegendo a borda do sujeito. 1-3 costuma "
                    "limpar a franja. Negativo faz o contrario (a mascara avanca sobre o sujeito)."}),
                "feather": ("INT", {"default": 4, "min": 0, "max": 128, "step": 1, "tooltip":
                    "Suaviza a borda da mascara final, em pixels. Evita recorte de tesoura."}),
                "fundo_fit": (["cobrir", "caber", "esticar"], {"default": "cobrir", "tooltip":
                    "Como encaixar a imagem de fundo no quadro. cobrir = preenche e corta a sobra (recomendado). "
                    "caber = cabe inteira (pode sobrar barra). esticar = distorce pra preencher."}),

                # ---------------------------------------------------------
                # APPEND-ONLY: widgets NOVOS vao SEMPRE no FIM desta lista.
                # O ComfyUI casa os `widgets_values` salvos por ORDEM, nao por
                # nome. Inserir um widget no meio desloca TODOS os valores dos
                # workflows ja salvos -- foi exatamente o que quebrou aqui
                # ("cor_chave: 0 not in [...]", "tolerancia: 2.0 > max 1.0").
                # Se precisar de um widget novo, acrescente ABAIXO desta nota.
                # ---------------------------------------------------------
                "modo": (["chroma", "luma", "matte_externo"], {"default": "chroma", "tooltip":
                    "COMO extrair o recorte:\n"
                    "chroma = tira a chave pela COR do fundo (verde/azul). Usa cor_chave/tolerancia/suavidade.\n"
                    "luma = tira pela LUMINANCIA -- pra material filmado contra fundo preto ou branco. Usa "
                    "luma_fundo/tolerancia/suavidade.\n"
                    "matte_externo = voce JA TEM o matte pronto (renderizado no Blender/AE, ou saida de um RMBG/SAM3): "
                    "ligue em 'matte' e diga em 'matte_representa' se o branco e o sujeito ou o fundo. Aqui a "
                    "tolerancia/suavidade nao sao usadas -- o matte entra como veio."}),
                "matte_representa": (["sujeito", "fundo"], {"default": "sujeito", "tooltip":
                    "[modo=matte_externo] O que o BRANCO do seu matte significa. 'sujeito' e a convencao usual de VFX "
                    "(branco = o que fica). 'fundo' se o seu matte ja veio invertido."}),
                "luma_fundo": (["escuro", "claro"], {"default": "escuro", "tooltip":
                    "[modo=luma] O fundo e PRETO (escuro) ou BRANCO (claro)? Define de que lado da luminancia esta o "
                    "que deve virar fundo."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("frames", "mask_fundo", "mask_sujeito", "preview", "info")
    OUTPUT_TOOLTIPS = (
        "Sujeito com spill removido + fundo de referencia composto -> ligue no 'frames' do encode.",
        "Mascara do FUNDO (1 = o modelo pode gerar) -> ligue no 'H3 Mascara -> Latente' / noise_mask.",
        "Mascara do SUJEITO (o inverso) -- util pra outros usos.",
        "Composicao com a mascara pintada de rosa -- CONFIRA a borda aqui antes de renderizar.",
        "Cobertura do fundo e avisos de chave.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Chromakey · fundo -> mascara + referencia (Bruxos): tira a chave do croma, REMOVE O SPILL verde da borda do "
        "sujeito e compoe a sua imagem de fundo atras dele -- devolvendo tambem a mascara do fundo pronta pro inpaint. "
        "Mande a saida pro encode com denoise 0.3-0.5: o modelo harmoniza a composicao em vez de inventar fundo do zero."
    )

    def run(self, frames, fundo=None, modo="chroma", matte=None, matte_representa="sujeito",
            luma_fundo="escuro", cor_chave="verde", chave_r=0, chave_g=255, chave_b=0,
            tolerancia=0.5, suavidade=0.3, despill=1.0, contrair=2, feather=4, fundo_fit="cobrir"):
        if not _OK:
            raise RuntimeError("[Bruxos Chromakey] torch indisponivel.")
        if frames.ndim != 4:
            raise ValueError(f"[Bruxos Chromakey] 'frames' precisa ser IMAGE [T,H,W,C]; veio {tuple(frames.shape)}.")

        T, H, W, _ = (int(v) for v in frames.shape)
        dev = frames.device
        x = frames[..., :3].float()
        R, G, B = x[..., 0], x[..., 1], x[..., 2]
        d = float(despill)

        # =================================================================
        # MODO 3: matte pronto -- entra como veio, sem chave nem despill
        # =================================================================
        if modo == "matte_externo":
            if matte is None:
                raise ValueError(
                    "[Bruxos Chromakey] modo='matte_externo' exige a entrada 'matte'. "
                    "Ligue ali o seu matte (grayscale) -- ou troque o modo pra 'chroma'/'luma'."
                )
            m = matte
            if m.dim() == 4:                       # veio como IMAGE
                m = m[..., :3].amax(dim=-1)
            elif m.dim() == 2:
                m = m.unsqueeze(0)
            m = m.float().clamp(0, 1).to(dev)
            if m.shape[0] == 1 and T > 1:
                m = m.repeat(T, 1, 1)
            elif m.shape[0] != T:
                print(f"[Bruxos Chromakey] matte com {m.shape[0]} frames e video com {T} -- reamostrando no tempo.",
                      flush=True)
                idx = torch.linspace(0, m.shape[0] - 1, T).round().long().clamp(0, m.shape[0] - 1)
                m = m[idx]
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
            # convencao: a nossa mascara e do FUNDO
            key = (1.0 - m) if matte_representa == "sujeito" else m
            nome = f"matte externo (branco = {matte_representa})"
            d = 0.0                                 # nao ha spill a remover

        else:
            # =============================================================
            # MODO 2: luma -- fundo preto ou branco
            # =============================================================
            if modo == "luma":
                # luminancia perceptual (Rec.709)
                lum = 0.2126 * R + 0.7152 * G + 0.0722 * B
                # dominancia: >0 onde e FUNDO
                dom = (0.5 - lum) if luma_fundo == "escuro" else (lum - 0.5)
                t_lo = 0.5 * (1.0 - float(tolerancia))
                larg = 0.005 + 0.245 * float(suavidade)
                t_hi = t_lo + larg
                key = ((dom - t_lo) / max(t_hi - t_lo, 1e-6)).clamp(0, 1)
                nome = f"luma (fundo {luma_fundo})"
                d = 0.0                             # despill nao se aplica a luma

            # =============================================================
            # MODO 1: chroma -- pela cor
            # =============================================================
            else:
                if cor_chave == "azul":
                    canal, outros = B, torch.maximum(R, G)
                    nome = "chroma azul"
                elif cor_chave == "personalizada":
                    c = torch.tensor([chave_r, chave_g, chave_b], dtype=torch.float32) / 255.0
                    i = int(torch.argmax(c))
                    canal = x[..., i]
                    outros = torch.maximum(x[..., (i + 1) % 3], x[..., (i + 2) % 3])
                    nome = f"chroma personalizada(canal {'RGB'[i]})"
                else:
                    canal, outros = G, torch.maximum(R, B)
                    nome = "chroma verde"

                # dominancia: >0 onde a cor-chave manda. Num croma bem
                # iluminado da ~0.25-0.5; no sujeito costuma ser <=0.
                dom = canal - outros
                t_lo = 0.30 * (1.0 - float(tolerancia))
                larg = 0.005 + 0.245 * float(suavidade)
                t_hi = t_lo + larg
                key = ((dom - t_lo) / max(t_hi - t_lo, 1e-6)).clamp(0, 1)   # 1 = fundo

                # ---- despill: puxa o canal da chave pro nivel dos outros --
                # limpa a franja do sujeito (que fica FORA da mascara e nao
                # seria corrigida pelo modelo).
                if d > 0:
                    excesso = (canal - outros).clamp(min=0.0)
                    novo = canal - d * excesso
                    x = x.clone()
                    if cor_chave == "azul":
                        x[..., 2] = novo
                    elif cor_chave == "personalizada":
                        c = torch.tensor([chave_r, chave_g, chave_b], dtype=torch.float32)
                        x[..., int(torch.argmax(c))] = novo
                    else:
                        x[..., 1] = novo
                    x = x.clamp(0, 1)

        # ---- mascara do fundo: contrai + feather -------------------------
        mask_bg = _feather(key, grow=-int(contrair), blur=int(feather))
        mask_fg = (1.0 - mask_bg).clamp(0, 1)

        # ---- compoe o fundo ----------------------------------------------
        if fundo is not None:
            bg = fundo[:1, ..., :3].float().to(dev)
            bg = _ajustar_fundo(bg, H, W, fundo_fit)          # [1,H,W,3]
            fonte_bg = f"referencia ({fundo_fit})"
        else:
            bg = torch.full((1, H, W, 3), 0.5, dtype=torch.float32, device=dev)
            fonte_bg = "cinza neutro (sem referencia)"

        a = mask_bg.unsqueeze(-1)                              # [T,H,W,1]
        saida = (x * (1.0 - a) + bg * a).clamp(0, 1)

        # ---- preview com a mascara em rosa -------------------------------
        tint = torch.tensor([1.0, 0.15, 0.45], dtype=saida.dtype, device=dev).view(1, 1, 1, 3)
        preview = (saida * (1 - a * 0.45) + tint * (a * 0.45)).clamp(0, 1)

        cob = float(mask_bg.mean()) * 100.0
        avisos = ""
        if cob < 5.0:
            avisos = (" | ATENCAO: quase nada virou fundo. " +
                      ("Confira o 'matte_representa' (talvez esteja invertido)." if modo == "matte_externo"
                       else "SUBA a tolerancia ou confira a cor/luma."))
        elif cob > 95.0:
            avisos = (" | ATENCAO: quase tudo virou fundo -- o sujeito foi comido. " +
                      ("Inverta o 'matte_representa'." if modo == "matte_externo" else "DESCA a tolerancia."))
        if modo == "chroma" and d <= 0:
            avisos += " | despill=0: vai sobrar franja colorida na borda do sujeito."

        if modo == "matte_externo":
            det = "matte entra como veio (sem chave/despill)"
        else:
            det = f"tol={tolerancia:.2f} suav={suavidade:.2f} (limiar {t_lo:.3f}->{t_hi:.3f}) despill={d:.2f}"
        info = (f"modo={nome} | {det} | contrai={contrair} feather={feather} | fundo={fonte_bg} | "
                f"{cob:.1f}% do quadro e fundo{avisos}")
        print(f"[Bruxos Chromakey] {info}", flush=True)
        return (saida, mask_bg, mask_fg, preview, info)


NODE_CLASS_MAPPINGS = {"BruxosChromaKeyBG": BruxosChromaKeyBG}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosChromaKeyBG": "Chromakey · fundo -> mascara + referencia (Bruxos)"}
