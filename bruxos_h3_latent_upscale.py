# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — MiniMax H3: upscale de LATENTE entre dois samplers (2-pass)
===========================================================================
PRA QUE SERVE
    Rodar a maior parte do schedule numa resolucao e TERMINAR noutra maior.
    E o substituto local do H3-Regenerate-2K -- o modulo que a MiniMax usa pra
    entregar 2K e que NAO foi aberto.

    passe 1 (sigmas altos, barato)  ->  upscale do latente  ->  passe 2 (sigmas
    baixos, resolucao final)

POR QUE OS NODES PADRAO NAO SERVEM
    `LatentUpscaleBy` e `AddNoise` assumem que o latente e um TENSOR. O do H3 e
    um container com dois componentes:
        video [B, 24, T, H/16, W/16]   +   audio [B, 32, 2, T_audio]
    Passar isso pelos nodes padrao quebra ou corrompe.

A PARTE QUE QUASE NINGUEM LEMBRA (e que este node faz)
    Nao basta escalar o latente de video. No modo ref2va, o CONDICIONAMENTO
    carrega os latentes das suas referencias junto com os metadados
    `latent_h` / `latent_w`. Se o canvas cresce e as referencias nao, elas ficam
    na escala relativa errada e o RoPE le as linhas deslocadas -- a identidade
    da referencia DEFORMA, e voce vai jurar que o problema e o prompt.
    Aqui as referencias sao escaladas junto, latente e metadado.

LIGACAO
    BasicScheduler ─ SIGMAS ─> SplitSigmas
                                 ├ high ─> SamplerCustomAdvanced #1
                                 │           (pega `denoised_output`)
                                 │              │
                                 │              v
                                 │        [este node]  escala 1.5
                                 │         ^      ^  \
                                 │      NOISE   MODEL  \_ positive/negative
                                 │                        (os MESMOS do passe 1)
                                 └ low ──> SamplerCustomAdvanced #2
                                            noise = DisableNoise
                                            guider = NOVO BasicGuider feito com
                                                     o positive que sai daqui

CUIDADOS
  * Use a saida `denoised_output` do passe 1, NAO a `output`.
  * O passe 2 vai com DisableNoise -- o ruido ja foi somado aqui.
  * Nao ponha Empty Cache / force-unload entre os passes: com o MiniMax
    quantizado + SageAttention isso costuma derrubar o processo.
"""

import logging

try:
    import torch
    import torch.nn.functional as F
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

try:
    from .minimax_h3_bruxos import _componentes, _reconstruir, _fmt
except Exception:  # pragma: no cover
    from minimax_h3_bruxos import _componentes, _reconstruir, _fmt

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/MiniMax H3"

# O VAE do H3 comprime 16x no espaco e o transformer ainda faz patch 2x2.
# Logo a resolucao final precisa ser multipla de 32 -> o LATENTE precisa ser
# multiplo de 2. Nao e detalhe: latente impar desalinha o patchify.
MULT_LATENTE = 2


def _par(n):
    n = int(round(n))
    return max(MULT_LATENTE, n - (n % MULT_LATENTE))


def _escalar_espacial(t, nh, nw, metodo):
    """[B,C,T,H,W] ou [B,C,H,W] -> mesma coisa com H,W novos."""
    if t.ndim == 5:
        B, C, T, H, W = t.shape
        x = t.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.interpolate(x.float(), size=(nh, nw), mode=metodo,
                          **({"align_corners": False} if metodo in ("bilinear", "bicubic") else {}))
        return x.reshape(B, T, C, nh, nw).permute(0, 2, 1, 3, 4).to(t.dtype).contiguous()
    if t.ndim == 4:
        x = F.interpolate(t.float(), size=(nh, nw), mode=metodo,
                          **({"align_corners": False} if metodo in ("bilinear", "bicubic") else {}))
        return x.to(t.dtype).contiguous()
    raise ValueError(f"[Bruxos H3 Upscale] nao sei escalar tensor {tuple(t.shape)}")


# ---------------------------------------------------------------------------
# condicionamento: escala os latentes das REFERENCIAS junto
# ---------------------------------------------------------------------------
def _escalar_refs(obj, fator, metodo, contador, prof=0):
    """Percorre a estrutura do condicionamento e escala todo bloco que tenha
    um latente visual + os metadados latent_h/latent_w.

    Escrito por DUCK TYPING de proposito: a chave exata muda entre versoes do
    ComfyUI ('minimax_refs', 'minimax_keyframes', ...). Em vez de casar nome,
    procuramos a ASSINATURA: um tensor sob alguma chave com 'latent' no nome,
    acompanhado de latent_h/latent_w numericos."""
    if prof > 6:
        return obj
    if isinstance(obj, dict):
        chaves_lat = [k for k, v in obj.items()
                      if "latent" in str(k).lower() and torch.is_tensor(v) and v.ndim in (4, 5)]
        tem_meta = any(str(k).lower().endswith(("latent_h", "latent_w")) for k in obj)
        novo = dict(obj)
        if chaves_lat and tem_meta:
            for k in chaves_lat:
                t = obj[k]
                h, w = int(t.shape[-2]), int(t.shape[-1])
                nh, nw = _par(h * fator), _par(w * fator)
                novo[k] = _escalar_espacial(t, nh, nw, metodo)
                contador["refs"] += 1
                contador["detalhe"].append(f"{k}: {h}x{w} -> {nh}x{nw}")
            for k in list(novo.keys()):
                kl = str(k).lower()
                if kl.endswith("latent_h") and isinstance(novo[k], (int, float)):
                    novo[k] = _par(novo[k] * fator)
                elif kl.endswith("latent_w") and isinstance(novo[k], (int, float)):
                    novo[k] = _par(novo[k] * fator)
            return novo
        return {k: _escalar_refs(v, fator, metodo, contador, prof + 1) for k, v in novo.items()}
    if isinstance(obj, (list, tuple)):
        saida = [_escalar_refs(v, fator, metodo, contador, prof + 1) for v in obj]
        return type(obj)(saida) if isinstance(obj, tuple) else saida
    return obj


def _escalar_cond(cond, fator, metodo, contador):
    """CONDITIONING = lista de [tensor, dict]. So o dict tem as referencias."""
    if cond is None:
        return None
    saida = []
    for item in cond:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            saida.append([item[0], _escalar_refs(item[1], fator, metodo, contador)])
        else:
            saida.append(item)
    return saida


class BruxosH3LatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip":
                    "A saida 'denoised_output' do SamplerCustomAdvanced do PASSE 1. "
                    "NAO use a saida 'output' -- ela ainda tem ruido do schedule."}),
                "model": ("MODEL", {"tooltip":
                    "O MESMO modelo dos dois passes (depois do Patch Sol-Attn, se voce usa). "
                    "Serve pra ler o model_sampling e somar o ruido do jeito certo pro flow-matching."}),
                "noise": ("NOISE", {"tooltip":
                    "Um RandomNoise. O ruido do passe 2 e somado AQUI -- por isso o sampler do "
                    "passe 2 vai com DisableNoise."}),
                "sigmas": ("SIGMAS", {"tooltip":
                    "Os sigmas BAIXOS (a segunda metade do SplitSigmas). Sao eles que definem "
                    "quanto ruido entra."}),
                "escala": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip":
                    "Quanto crescer no espaco. 1.5 costuma ser o melhor negocio: ~1.2x de custo "
                    "por 1.5x de resolucao.\n"
                    "O resultado e arredondado pra manter o latente PAR (a resolucao final precisa "
                    "ser multipla de 32)."}),
                "metodo": (["bilinear", "bicubic", "nearest-exact", "area"], {"default": "bicubic",
                    "tooltip":
                    "Interpolacao do latente. bicubic segura melhor o detalhe; bilinear e mais "
                    "macio; nearest-exact preserva blocos duros."}),
            },
            "optional": {
                "positive": ("CONDITIONING", {"tooltip":
                    "O MESMO positive usado no passe 1. LIGUE ISTO em ref2va: e aqui que os "
                    "latentes das suas referencias sao escalados junto do canvas. Sem isso as "
                    "referencias ficam na escala relativa errada e a identidade delas deforma."}),
                "negative": ("CONDITIONING", {"tooltip": "O negative do passe 1, se voce usa um."}),
                "audio_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip":
                    "Quanto ruido devolver ao AUDIO pro passe 2.\n"
                    "0.0 = trava o audio do passe 1 (use isto se voce nao liga pro audio -- e o "
                    "mais seguro, nao tem como embolar).\n"
                    "0.25-0.5 = polimento leve. 1.0 = remix completo.\n"
                    "Se o audio sair embolado, baixe isto ou rode mais steps no passe 1: o audio "
                    "do H3 assenta tarde no schedule."}),
                "escalar_condicionamento": ("BOOLEAN", {"default": True, "tooltip":
                    "Escala os latentes das referencias e os metadados latent_h/latent_w junto. "
                    "Deixe LIGADO em ref2va. Desligue so pra diagnosticar."}),
                "para_cpu": ("BOOLEAN", {"default": True, "tooltip":
                    "Estaciona o latente na RAM e libera cache da GPU antes do passe 2, SEM "
                    "descarregar modelo. Ajuda a caber o passe em resolucao maior.\n"
                    "NAO troque isto por um node de Empty Cache/force-unload: com o MiniMax "
                    "quantizado + SageAttention isso costuma derrubar o processo."}),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("latent", "positive", "negative", "info")
    OUTPUT_TOOLTIPS = (
        "Latente maior e ja com o ruido do passe 2 -> 'latent_image' do Sampler #2 (com DisableNoise).",
        "Positive com as referencias reescaladas -> monte um NOVO BasicGuider com ele.",
        "Negative reescalado (vazio se voce nao ligou nada).",
        "O que foi escalado, resolucao antes/depois e avisos.",
    )
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "H3 Latent Upscale 2-pass (Bruxos): escala o latente do H3 entre dois samplers pra terminar a geracao "
        "numa resolucao maior -- substituto local do H3-Regenerate-2K, que a MiniMax nao abriu. "
        "Trata o container audio+video (os nodes padrao quebram nele), soma o ruido do passe 2 pelo model_sampling "
        "do flow-matching, e -- o ponto critico -- REESCALA OS LATENTES DAS REFERENCIAS junto com latent_h/latent_w, "
        "senao elas ficam na escala errada em relacao ao novo canvas e a identidade deforma."
    )

    def run(self, latent, model, noise, sigmas, escala, metodo,
            positive=None, negative=None, audio_denoise=0.0,
            escalar_condicionamento=True, para_cpu=True):
        if not _OK:
            raise RuntimeError("[Bruxos H3 Upscale] torch indisponivel.")

        s = latent.get("samples", latent) if isinstance(latent, dict) else latent
        comps, tipo = _componentes(s)
        if not comps:
            raise ValueError(
                "[Bruxos H3 Upscale] nao reconheci o latente. Ele precisa vir do "
                "'MiniMax H3 Reference to Video' (via SamplerCustomAdvanced). "
                "Ponha o 'Inspetor de LATENT (Bruxos)' no fio pra ver o que esta chegando."
            )
        video = comps[0]
        if video.ndim != 5:
            raise ValueError(f"[Bruxos H3 Upscale] esperava video 5D [B,C,T,H,W]; veio "
                             f"{tuple(video.shape)}.")

        B, C, T, H, W = (int(v) for v in video.shape)
        nh, nw = _par(H * float(escala)), _par(W * float(escala))
        if (nh, nw) == (H, W):
            print("[Bruxos H3 Upscale] AVISO: a escala nao mudou o tamanho do latente "
                  "(escala perto de 1.0). Vou seguir e so re-ruidar.", flush=True)

        fator_real = ((nh / H) + (nw / W)) / 2.0
        print(f"[Bruxos H3 Upscale] latente {H}x{W} -> {nh}x{nw} "
              f"(pixels {W*16}x{H*16} -> {nw*16}x{nh*16}) | escala pedida {escala} "
              f"| real {fator_real:.3f} | {metodo}", flush=True)

        novo_video = _escalar_espacial(video, nh, nw, metodo)

        # ---- ruido do passe 2, pelo model_sampling (flow-matching) --------
        ms = model.get_model_object("model_sampling")
        if len(sigmas) > 1:
            escalar = torch.abs(sigmas[0] - sigmas[-1])
        else:
            escalar = sigmas[0]

        ruido = noise.generate_noise({"samples": novo_video})
        ruido = ruido.to(novo_video.device, dtype=novo_video.dtype)

        alvo = novo_video
        proc_in = getattr(getattr(model, "model", None), "process_latent_in", None)
        proc_out = getattr(getattr(model, "model", None), "process_latent_out", None)
        try:
            if proc_in is not None and torch.count_nonzero(alvo) > 0:
                alvo = proc_in(alvo)
        except Exception as e:
            print(f"[Bruxos H3 Upscale] process_latent_in nao aplicavel ({e}); seguindo cru.", flush=True)

        ruidado = ms.noise_scaling(escalar, ruido, alvo)
        try:
            if proc_out is not None:
                ruidado = proc_out(ruidado)
        except Exception as e:
            print(f"[Bruxos H3 Upscale] process_latent_out nao aplicavel ({e}); seguindo cru.", flush=True)
        ruidado = torch.nan_to_num(ruidado, nan=0.0, posinf=0.0, neginf=0.0)

        saida = list(comps)
        saida[0] = ruidado

        # ---- audio: travado por padrao -----------------------------------
        nota_audio = "audio do passe 1 mantido (audio_denoise=0)"
        if len(comps) > 1 and float(audio_denoise) > 0.0:
            aud = comps[1]
            r_aud = torch.randn(aud.shape, device=aud.device, dtype=torch.float32,
                                generator=None).to(aud.dtype)
            try:
                saida[1] = ms.noise_scaling(escalar * float(audio_denoise), r_aud, aud)
                nota_audio = f"audio re-ruidado a {float(audio_denoise):.2f}"
            except Exception as e:
                print(f"[Bruxos H3 Upscale] nao consegui re-ruidar o audio ({e}); mantendo o do passe 1.",
                      flush=True)

        container = _reconstruir(s, saida)
        out = dict(latent) if isinstance(latent, dict) else {}
        out["samples"] = container
        out.pop("noise_mask", None)   # a mascara antiga esta no tamanho velho

        # ---- condicionamento ---------------------------------------------
        cont = {"refs": 0, "detalhe": []}
        pos, neg = positive, negative
        if escalar_condicionamento:
            pos = _escalar_cond(positive, fator_real, metodo, cont)
            neg = _escalar_cond(negative, fator_real, metodo, cont)
            if positive is not None and cont["refs"] == 0:
                print("[Bruxos H3 Upscale] *** ATENCAO: liguei o positive mas NAO achei nenhum "
                      "latente de referencia pra escalar. ***\n"
                      "[Bruxos H3 Upscale] Em ref2va isso significa que as referencias vao ficar "
                      "na escala do canvas ANTIGO -- espere a identidade delas deformar.\n"
                      "[Bruxos H3 Upscale] Pode ser que a versao do ComfyUI mudou o formato. "
                      "Me mande esta mensagem que eu ajusto o localizador.", flush=True)
            for d in cont["detalhe"]:
                print(f"[Bruxos H3 Upscale]   ref {d}", flush=True)
        elif positive is not None:
            print("[Bruxos H3 Upscale] escalar_condicionamento DESLIGADO: as referencias ficam no "
                  "tamanho antigo (so faca isso pra diagnosticar).", flush=True)

        if para_cpu:
            try:
                import comfy.model_management as mm
                mm.soft_empty_cache()
            except Exception:
                pass

        info = (f"latente {H}x{W} -> {nh}x{nw} | pixels {W*16}x{H*16} -> {nw*16}x{nh*16} | "
                f"{metodo} | sigma {float(escalar):.4f} | {cont['refs']} referencia(s) reescalada(s) | "
                f"{nota_audio}")
        print(f"[Bruxos H3 Upscale] {info}", flush=True)
        return (out, pos, neg, info)


NODE_CLASS_MAPPINGS = {"BruxosH3LatentUpscale": BruxosH3LatentUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3LatentUpscale": "MiniMax H3 · Latent Upscale 2-pass (Bruxos)"
}
