# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — MiniMax H3: video/imagem -> LATENTE (pra denoise controlado)
============================================================================
Pra que serve: fazer com o H3 o mesmo que voce faz com o Wan -- pegar um video
que JA EXISTE, jogar ele no espaco latente e rodar o sampler com denoise BAIXO
(0.15-0.35) pra refinar/upscalar/inpaintar, em vez de gerar do zero.

O H3 nao e um modelo so de video: ele e AUDIO+VIDEO. O sampler dele trabalha
sobre um CONTAINER com DOIS latentes (video e audio), e sao dois VAEs separados
(minimax_h3_video_vae + minimax_h3_audio_vae). Por isso NAO adianta montar um
latente de video sozinho e mandar pro KSampler: falta metade da estrutura.

ESTRATEGIA DESTE NODE (e o motivo dele funcionar):
    Em vez de ADIVINHAR como o container do H3 e montado por dentro, a gente
    PEGA O CONTAINER PRONTO que o node nativo `MiniMax H3 Image to Video` ja
    devolve (a saida LATENT dele) e apenas TROCA o latente de video por um
    codificado do SEU video. Assim herdamos shape, dtype, device e a classe
    exata do container -- sem engenharia reversa e sem quebrar quando o
    ComfyUI atualizar.

Ligacao no grafo:

    MiniMax H3 Image to Video ─ LATENT ──> [reference_latent]
    Load Video ─ IMAGE ────────────────────> [frames]          ─> H3 Encode ─┐
    VAELoader (video vae) ─────────────────> [video_vae]                     │
    (opcional) audio + VAELoader(audio vae) > [audio]/[audio_vae]            │
                                                                             v
                                        SamplerCustomAdvanced (latent_image)
                                        BasicScheduler(denoise = 0.15-0.35)

BUGS CLASSICOS QUE ESTE NODE EVITA (e que causam aquele ruido tipo camuflagem):
  1. Normalizar 2x. O `vae.encode()` do ComfyUI JA recebe 0..1 e faz o
     `*2-1` internamente. Fazer `frames*2-1` antes = range errado = ruido.
  2. Permutar os eixos na mao. O `vae.encode()` espera [T,H,W,C] e faz o
     movedim sozinho. Mandar [B,C,T,H,W] embaralha os eixos.
  3. Des-normalizar 2x no decode. O `vae.decode()` JA devolve 0..1.
  4. Inventar um latente de audio de zeros com shape chutado.
"""

import logging

try:
    import torch
    import torch.nn.functional as F   # usado no _encaixar (auto_ajustar)
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/MiniMax H3"


# ---------------------------------------------------------------------------
# helpers de container (NestedTensor do H3 / lista / tensor puro)
# ---------------------------------------------------------------------------
def _componentes(samples):
    """Devolve (lista_de_tensores, rotulo_do_tipo).

    Cobre: NestedTensor (atributo .tensors ou iteravel), list/tuple, e tensor
    puro. NAO assume a API interna -- so olha o que existe."""
    if torch.is_tensor(samples):
        return [samples], "tensor"
    t = getattr(samples, "tensors", None)
    if isinstance(t, (list, tuple)) and t and all(torch.is_tensor(x) for x in t):
        return list(t), "nested(.tensors)"
    if isinstance(samples, (list, tuple)) and samples and all(torch.is_tensor(x) for x in samples):
        return list(samples), "lista"
    # ultimo recurso: iteravel de tensores
    try:
        it = list(samples)
        if it and all(torch.is_tensor(x) for x in it):
            return it, "iteravel"
    except Exception:
        pass
    return [], "desconhecido"


def _reconstruir(original, novos):
    """Monta um container NOVO do MESMO tipo do `original`, com `novos` dentro.
    Se nao conseguir, devolve a lista crua (o sampler costuma aceitar)."""
    if torch.is_tensor(original):
        return novos[0]
    if isinstance(original, list):
        return list(novos)
    if isinstance(original, tuple):
        return tuple(novos)
    cls = type(original)
    for tentativa in (lambda: cls(novos), lambda: cls(*novos)):
        try:
            return tentativa()
        except Exception:
            continue
    log.warning("[Bruxos H3] nao consegui reconstruir %s; devolvendo lista.", cls.__name__)
    return list(novos)


def _fmt(t):
    return f"{tuple(t.shape)} {t.dtype} {t.device}"


# ---------------------------------------------------------------------------
# 1) INSPETOR — descobre a estrutura real do latente do H3
# ---------------------------------------------------------------------------
class BruxosMinimaxH3LatentInspect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Qualquer LATENT. Use na saida do 'MiniMax H3 Image to Video' pra ver como o container do H3 e montado por dentro."}),
            },
            "optional": {
                "rotulo": ("STRING", {"default": "H3", "tooltip": "Nome que aparece no console, pra diferenciar varios inspetores no mesmo grafo."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    OUTPUT_TOOLTIPS = ("O mesmo latente, intacto (passthrough).", "Descricao da estrutura: quantos componentes, shapes, dtypes.")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Inspetor de LATENT (Bruxos): imprime a estrutura REAL do latente -- quantos tensores tem dentro "
        "(video/audio no caso do H3), shape, dtype e device de cada um. Passthrough: pode deixar no meio do fio. "
        "Use isto ANTES de tentar montar latente na mao; e assim que voce descobre o formato certo."
    )

    def run(self, latent, rotulo="H3"):
        s = latent.get("samples", latent) if isinstance(latent, dict) else latent
        comps, tipo = _componentes(s)
        linhas = [f"[{rotulo}] container={type(s).__name__} ({tipo}) | {len(comps)} componente(s)"]
        for i, c in enumerate(comps):
            papel = "video" if i == 0 else ("audio" if i == 1 else f"extra{i}")
            linhas.append(f"[{rotulo}]   [{i}] {papel:<6} {_fmt(c)}")
        if isinstance(latent, dict):
            outras = [k for k in latent.keys() if k != "samples"]
            if outras:
                linhas.append(f"[{rotulo}] outras chaves no dict: {outras}")
        txt = "\n".join(linhas)
        print(txt, flush=True)
        return (latent, txt)


# ---------------------------------------------------------------------------
# 2) ENCODE — video real -> latente do H3, reaproveitando o container nativo
# ---------------------------------------------------------------------------
class BruxosMinimaxH3EncodeVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip":
                    "Os frames do SEU video (Load Video). Sao eles que vao pro espaco latente."}),
                "video_vae": ("VAE", {"tooltip":
                    "O VAE DE VIDEO do H3 (minimax_h3_video_vae_*.safetensors). NAO e o de audio."}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip":
                    "O VAE DE AUDIO do H3 (minimax_h3_audio_vae_*.safetensors).\n"
                    "MESMO SEM QUERER AUDIO, ligue: o H3 e um modelo audio+video e o sampler espera os DOIS latentes. "
                    "Sem audio ligado a gente codifica SILENCIO neste VAE -- assim o latente de audio sai com o shape e a "
                    "distribuicao corretos, em vez de um tensor de zeros com formato chutado (que da ruido)."}),
                "audio": ("AUDIO", {"tooltip":
                    "[opcional] Audio do seu video. Se ligado (com audio_vae), o denoise refina audio+video juntos. "
                    "Se vazio, usamos silencio."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01, "tooltip":
                    "FPS do seu video. Usado SO pra calcular a duracao do silencio (n_frames / fps) quando nao ha audio ligado. "
                    "Se errar aqui, o latente de audio sai com duracao diferente do video."}),
                "audio_sample_rate": ("INT", {"default": 32000, "min": 8000, "max": 192000, "step": 100, "tooltip":
                    "Taxa que o VAE de audio do H3 espera: 32000 Hz (NAO 44100).\n"
                    "O model card e explicito: 'H3-AudioVAE compresses 32 kHz audio into a sequence of latent "
                    "tokens with a temporal rate of 40 Hz'. Se voce mandar 44100 amostras aqui, o VAE le como "
                    "44100/32000 = 1.38 SEGUNDOS -- o latente de audio sai 38% mais longo que o video e tudo "
                    "dessincroniza. Se o seu audio vier em outra taxa, reamostramos pra esta."}),
                "reference_latent": ("LATENT", {"tooltip":
                    "[opcional, mas o CAMINHO MAIS SEGURO] A saida LATENT do node nativo 'MiniMax H3 Image to Video'.\n"
                    "Se ligado, herdamos dele o container exato (classe, ordem dos componentes, dtype, device) e so trocamos "
                    "o video -- zero adivinhacao. Se NAO ligado, montamos o container do zero (NestedTensor[video, audio]).\n"
                    "Ligue ele se der qualquer erro estranho de formato."}),
                "auto_ajustar": ("BOOLEAN", {"default": True, "tooltip":
                    "[com reference_latent] Se o seu video nao bater com o node nativo, REDIMENSIONA e ajusta os frames "
                    "automaticamente pro tamanho exato que a referencia pede, e recodifica. Resolve sozinho o erro de "
                    "'shape nao bate' -- voce nao precisa acertar width/height/length na mao.\n"
                    "Fatores do VAE do H3: 16x no espaco, 4x no tempo (T_lat = (frames-1)/4)."}),
                "ajuste_espacial": (["esticar", "cobrir", "caber"], {"default": "cobrir", "tooltip":
                    "[auto_ajustar] Como encaixar seu video na resolucao do node nativo quando a proporcao difere. "
                    "cobrir = preenche e corta a sobra (mantem a proporcao, recomendado). caber = cabe inteiro com "
                    "barras pretas. esticar = distorce pra preencher."}),
                "strict": ("BOOLEAN", {"default": True, "tooltip":
                    "[so com reference_latent] Se o shape nao bater E o auto_ajustar estiver DESLIGADO: LIGADO faz o node "
                    "PARAR e explicar o que divergiu; desligado so avisa (costuma dar ruido)."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    OUTPUT_TOOLTIPS = (
        "Latente do H3 com o SEU video dentro -> ligue no 'latent_image' do SamplerCustomAdvanced e use denoise baixo.",
        "Resumo do que foi codificado e trocado.",
    )
    FUNCTION = "encode"
    CATEGORY = CAT
    DESCRIPTION = (
        "MiniMax H3 Encode Video (Bruxos): codifica o SEU video no latente do H3 pra rodar o sampler com denoise "
        "controlado (refino/upscale/inpaint), como voce faz no Wan. Em vez de adivinhar o container audio+video do H3, "
        "ele herda o container do node nativo (reference_latent) e so troca o componente de video. "
        "Faz o encode do jeito certo: sem normalizar 2x e sem permutar eixos na mao (as duas causas classicas "
        "daquele ruido tipo camuflagem)."
    )

    # -- encode correto (o ponto central deste arquivo) ---------------------
    @staticmethod
    def _encode_video(vae, frames):
        """frames: IMAGE [T,H,W,C] em 0..1.
        O VAE do ComfyUI JA faz movedim(-1,1) e (x*2-1) por dentro -- por isso
        entregamos CRU. Qualquer normalizacao/permutacao aqui = ruido.

        Tambem NAO usamos encode_tiled: no VAE do MiniMax H3 o encode_tiled so
        chama o encode normal, e o caminho tiled do ComfyUI mexe nos eixos de
        um jeito que quebra o encode temporal."""
        if frames.ndim != 4:
            raise ValueError(
                f"[Bruxos H3] 'frames' precisa ser IMAGE [T,H,W,C] (4 dimensoes), "
                f"mas veio com {frames.ndim} dimensoes {tuple(frames.shape)}.\n"
                f"NAO permute os eixos nem normalize antes: o vae.encode() do ComfyUI faz isso sozinho."
            )
        x = frames[:, :, :, :3] if frames.shape[-1] > 3 else frames
        T = int(x.shape[0])
        try:
            out = vae.encode(x)
        except Exception as e:
            if "non-empty list of Tensors" in str(e):
                raise ValueError(
                    f"[Bruxos H3] o VAE de video do H3 nao conseguiu fatiar {T} frame(s) no tempo "
                    f"(o encode temporal saiu vazio).\n"
                    f"Causa provavel: numero de frames pequeno demais ou fora do grid do modelo. "
                    f"O H3 costuma seguir 4n+1 (5, 9, ..., 81, 121, 125). Com 1 frame so ele costuma falhar.\n"
                    f"Conserto: mande um VIDEO (varios frames, de preferencia 4n+1). Se so tem uma imagem, "
                    f"repita ela ate fechar 4n+1 frames antes deste node.\n"
                    f"Erro original: {e}"
                ) from e
            raise
        if isinstance(out, dict) and "samples" in out:
            out = out["samples"]
        return out

    @staticmethod
    def _encode_audio(vae, audio, alvo_sr):
        """Mesmo caminho do VAEEncodeAudio do ComfyUI: waveform.movedim(1,-1)."""
        wf = audio.get("waveform") if isinstance(audio, dict) else audio
        if wf is None:
            return None
        sr = int(audio.get("sample_rate", alvo_sr)) if isinstance(audio, dict) else alvo_sr
        if sr != int(alvo_sr):
            try:
                import torchaudio
                wf = torchaudio.functional.resample(wf, sr, int(alvo_sr))
                print(f"[Bruxos H3] audio reamostrado {sr} -> {alvo_sr} Hz", flush=True)
            except Exception as e:
                print(f"[Bruxos H3] AVISO: nao consegui reamostrar o audio ({e}); "
                      f"mandando em {sr} Hz. Se o VAE esperar outra taxa, o audio sai errado.", flush=True)
        out = vae.encode(wf.movedim(1, -1))
        if isinstance(out, dict) and "samples" in out:
            out = out["samples"]
        return out

    # Fator ESPACIAL do VAE do H3: 16x (1920px -> 120 slots). Esse e confiavel.
    F_ESP = 16
    # Fator TEMPORAL nominal: 4x. MAS ele NAO e linear na pratica -- o encode do
    # ComfyUI fatia o video em blocos no tempo e CADA BLOCO gasta um slot extra.
    # Medido no proprio H3:   24 frames -> 7 latentes   (bate com T/4+1)
    #                        149 frames -> 42 latentes  (a formula previa 38!)
    # Por isso NAO da pra inverter a formula na mao. O metodo `_frames_para_tlat`
    # abaixo DESCOBRE o numero de frames por medicao, encodando miniaturas.
    F_TMP = 4

    # memo da calibracao: {(id(vae), T): T_lat}
    _MEMO_TL = {}

    @classmethod
    def _probe_tlat(cls, vae, T, lado=64):
        """Quantos slots latentes o VAE gera pra T frames? Mede de verdade,
        encodando um clipe MINIATURA (64x64) -- ~500x mais barato que o video
        real, e o fator temporal nao depende da resolucao."""
        chave = (id(vae), int(T))
        if chave in cls._MEMO_TL:
            return cls._MEMO_TL[chave]
        try:
            dummy = torch.zeros((int(T), lado, lado, 3), dtype=torch.float32)
            out = vae.encode(dummy)
            if isinstance(out, dict) and "samples" in out:
                out = out["samples"]
            tl = int(out.shape[2]) if out.ndim == 5 else int(out.shape[0])
        except Exception:
            tl = -1
        cls._MEMO_TL[chave] = tl
        return tl

    @classmethod
    def _frames_para_tlat(cls, vae, alvo_tl, teto=None):
        """Busca binaria: qual T de frames produz EXATAMENTE `alvo_tl` latentes.
        Devolve (T, exato_bool). T_lat e monotono crescente em T, entao a busca
        binaria e valida."""
        alvo_tl = int(alvo_tl)
        teto = int(teto or max(64, alvo_tl * cls.F_TMP * 2))
        lo, hi = 1, teto
        if cls._probe_tlat(vae, hi) < alvo_tl:
            # o teto nao alcanca: dobra ate alcancar (com limite de seguranca)
            for _ in range(4):
                hi *= 2
                if cls._probe_tlat(vae, hi) >= alvo_tl:
                    break
        melhor, melhor_erro = None, None
        while lo <= hi:
            meio = (lo + hi) // 2
            tl = cls._probe_tlat(vae, meio)
            if tl < 0:                      # encode falhou nesse T (poucos frames)
                lo = meio + 1
                continue
            erro = abs(tl - alvo_tl)
            if melhor is None or erro < melhor_erro:
                melhor, melhor_erro = meio, erro
            if tl == alvo_tl:
                # achou; empurra pra frente pra pegar o MAIOR T que ainda bate
                # (mais frames reais = menos frame repetido no fim)
                t2 = meio
                while cls._probe_tlat(vae, t2 + 1) == alvo_tl and t2 < teto:
                    t2 += 1
                return t2, True
            if tl < alvo_tl:
                lo = meio + 1
            else:
                hi = meio - 1
        return (melhor or alvo_tl * cls.F_TMP), False

    @staticmethod
    def _casar_latente(novo, ref):
        """Rede de seguranca: forca `novo` a ter EXATAMENTE o shape de `ref`.
        Espaco e tempo por interpolacao (trilinear) -- assim nada do video e
        jogado fora, so re-amostrado. Devolve (tensor, descricao_do_que_fiz)."""
        if tuple(novo.shape) == tuple(ref.shape):
            return novo, ""
        if novo.ndim != 5 or ref.ndim != 5:
            raise ValueError(
                f"[Bruxos H3] nao sei casar latentes de {novo.ndim}D com {ref.ndim}D "
                f"({tuple(novo.shape)} vs {tuple(ref.shape)})."
            )
        if int(novo.shape[1]) != int(ref.shape[1]):
            raise ValueError(
                f"[Bruxos H3] o numero de CANAIS do latente nao bate: {int(novo.shape[1])} "
                f"vs {int(ref.shape[1])}. Isso e VAE errado -- confira se ligou o "
                f"minimax_h3_VIDEO_vae em 'video_vae' (e nao o de audio ou um VAE do Wan)."
            )
        antes = tuple(novo.shape)
        alvo = (int(ref.shape[2]), int(ref.shape[3]), int(ref.shape[4]))
        x = F.interpolate(novo.float(), size=alvo, mode="trilinear", align_corners=False)
        if int(novo.shape[0]) != int(ref.shape[0]):
            x = x.expand(int(ref.shape[0]), -1, -1, -1, -1).contiguous()
        return x, f"reamostrado {antes} -> {tuple(x.shape)}"

    @staticmethod
    def _encaixar(frames, alvo_h, alvo_w, modo="cobrir"):
        """frames [T,H,W,C] -> [T,alvo_h,alvo_w,C], preservando proporcao."""
        x = frames.permute(0, 3, 1, 2)                      # [T,C,H,W]
        T, C, H, W = x.shape
        if (H, W) == (alvo_h, alvo_w):
            return frames
        if modo == "esticar":
            x = F.interpolate(x, size=(alvo_h, alvo_w), mode="bilinear", align_corners=False)
        else:
            s = max(alvo_w / W, alvo_h / H) if modo == "cobrir" else min(alvo_w / W, alvo_h / H)
            nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
            if modo == "cobrir":
                y0, x0 = max(0, (nh - alvo_h) // 2), max(0, (nw - alvo_w) // 2)
                x = x[:, :, y0:y0 + alvo_h, x0:x0 + alvo_w]
                if x.shape[2] != alvo_h or x.shape[3] != alvo_w:
                    x = F.interpolate(x, size=(alvo_h, alvo_w), mode="bilinear", align_corners=False)
            else:
                cvs = torch.zeros((T, C, alvo_h, alvo_w), dtype=x.dtype, device=x.device)
                y0, x0 = (alvo_h - nh) // 2, (alvo_w - nw) // 2
                cvs[:, :, y0:y0 + nh, x0:x0 + nw] = x
                x = cvs
        return x.permute(0, 2, 3, 1).clamp(0, 1)

    @staticmethod
    def _ajustar_tempo(frames, alvo_t):
        """Corta ou estende (repetindo o ultimo frame) ate ter alvo_t frames."""
        T = int(frames.shape[0])
        if T == alvo_t:
            return frames
        if T > alvo_t:
            return frames[:alvo_t]
        return torch.cat([frames, frames[-1:].repeat(alvo_t - T, 1, 1, 1)], dim=0)

    @staticmethod
    def _silencio(n_frames, fps, sample_rate, canais=2, device=None, dtype=None):
        """Waveform de SILENCIO com a mesma duracao do video, no formato AUDIO
        do ComfyUI: [B, C, N]. Isso vai pro VAE de audio de verdade -- e por
        isso o latente sai com shape e distribuicao corretos, sem chute."""
        dur = float(n_frames) / max(float(fps), 1e-6)
        n = max(1, int(round(dur * float(sample_rate))))
        return torch.zeros((1, int(canais), n), device=device, dtype=(dtype or torch.float32)), dur, n

    def encode(self, frames, video_vae,
               audio_vae=None, audio=None, fps=24.0, audio_sample_rate=44100,
               reference_latent=None, auto_ajustar=True, ajuste_espacial="cobrir", strict=True):
        if not _OK:
            raise RuntimeError("[Bruxos H3] torch indisponivel.")

        # ---- VIDEO (sempre) ----------------------------------------------
        T = int(frames.shape[0])
        print(f"[Bruxos H3] codificando {T} frame(s) {int(frames.shape[2])}x{int(frames.shape[1])} "
              f"no VAE de video...", flush=True)
        novo_video = self._encode_video(video_vae, frames)
        print(f"[Bruxos H3] video codificado: {_fmt(novo_video)}", flush=True)

        # ---- AUDIO: real se ligado, senao SILENCIO pelo VAE de audio ------
        # O AudioVAE do H3 opera a 32 kHz com taxa latente de 40 Hz (model card).
        # Workflow antigo salvo com 44100 continua carregando esse valor -- avisa.
        if audio_vae is not None and int(audio_sample_rate) != 32000:
            print(f"[Bruxos H3] AVISO: audio_sample_rate={audio_sample_rate}, mas o AudioVAE do H3 espera "
                  f"32000 Hz. Ele vai interpretar suas amostras como "
                  f"{float(audio_sample_rate)/32000.0:.2f}x a duracao real -> audio fora de sincronia "
                  f"com o video. Troque o widget pra 32000.", flush=True)

        novo_audio = None
        origem_audio = "nenhum"
        if audio_vae is not None:
            try:
                if audio is not None:
                    novo_audio = self._encode_audio(audio_vae, audio, audio_sample_rate)
                    origem_audio = "audio ligado"
                if novo_audio is None:
                    wf, dur, n = self._silencio(T, fps, audio_sample_rate,
                                                device=novo_video.device, dtype=torch.float32)
                    print(f"[Bruxos H3] sem audio -> codificando SILENCIO de {dur:.2f}s "
                          f"({n} amostras @ {audio_sample_rate}Hz) no VAE de audio...", flush=True)
                    out = audio_vae.encode(wf.movedim(1, -1))
                    novo_audio = out["samples"] if isinstance(out, dict) and "samples" in out else out
                    origem_audio = f"silencio {dur:.2f}s"
                print(f"[Bruxos H3] audio codificado: {_fmt(novo_audio)} ({origem_audio})", flush=True)
            except Exception as e:
                print(f"[Bruxos H3] AVISO: falhou o encode do audio ({e}).", flush=True)
                novo_audio = None
                origem_audio = "falhou"
        elif audio is not None:
            print("[Bruxos H3] AVISO: voce ligou 'audio' mas NAO ligou 'audio_vae' -- audio IGNORADO.", flush=True)

        # =================================================================
        # CAMINHO A: com reference_latent -> herda o container (mais seguro)
        # =================================================================
        if reference_latent is not None:
            ref = reference_latent.get("samples", reference_latent) if isinstance(reference_latent, dict) else reference_latent
            comps, tipo = _componentes(ref)
            if not comps:
                raise ValueError(
                    "[Bruxos H3] nao reconheci o 'reference_latent'. Ligue nele a saida LATENT do node nativo "
                    "'MiniMax H3 Image to Video', ou desconecte pra eu montar o container do zero. "
                    "Pra investigar, ponha o 'Inspetor de LATENT (Bruxos)' no fio."
                )
            ref_video = comps[0]
            print(f"[Bruxos H3] referencia: {type(ref).__name__} ({tipo}), {len(comps)} componente(s) | "
                  f"video={_fmt(ref_video)}", flush=True)

            if tuple(novo_video.shape) != tuple(ref_video.shape):
                # alvo em PIXELS que a referencia exige (16x espaco, 4x tempo)
                if ref_video.ndim == 5:
                    _b, _c, Tl, Hl, Wl = (int(v) for v in ref_video.shape)
                    alvo_h, alvo_w = Hl * self.F_ESP, Wl * self.F_ESP
                    alvo_t = None   # descoberto por medicao logo abaixo
                else:
                    alvo_h = alvo_w = alvo_t = None

                if auto_ajustar and alvo_h:
                    # Quantos frames dao EXATAMENTE Tl latentes? Mede, nao chuta.
                    alvo_t, exato = self._frames_para_tlat(video_vae, Tl)
                    marca = "medido" if exato else "aproximado"
                    print(f"[Bruxos H3] alvo temporal {marca}: {alvo_t} frames -> {Tl} latentes "
                          f"(a formula (T-1)/4 daria {Tl * self.F_TMP + 1} -- por isso eu meco)", flush=True)
                    print(f"[Bruxos H3] shape nao bateu -> AJUSTANDO automaticamente: "
                          f"{int(frames.shape[2])}x{int(frames.shape[1])} x{T}f  ->  "
                          f"{alvo_w}x{alvo_h} x{alvo_t}f ({ajuste_espacial})", flush=True)
                    fr2 = self._encaixar(frames, alvo_h, alvo_w, ajuste_espacial)
                    fr2 = self._ajustar_tempo(fr2, alvo_t)
                    novo_video = self._encode_video(video_vae, fr2)
                    print(f"[Bruxos H3] recodificado: {_fmt(novo_video)}", flush=True)
                    novo_video, ajuste = self._casar_latente(novo_video, ref_video)
                    if ajuste:
                        print(f"[Bruxos H3] ajuste fino no latente: {ajuste}", flush=True)
                else:
                    msg = (
                        f"[Bruxos H3] o latente do seu video {tuple(novo_video.shape)} NAO bate com o do "
                        f"reference_latent {tuple(ref_video.shape)}.\n"
                        f"Seu video: {int(frames.shape[2])}x{int(frames.shape[1])}, {T} frames.\n"
                    )
                    if alvo_h:
                        _t, _ex = self._frames_para_tlat(video_vae, Tl)
                        msg += (f"Pra bater, o node nativo pede: width={alvo_w}, height={alvo_h}, "
                                f"length={_t}{'' if _ex else ' (aproximado)'}.\n"
                                f"Opcoes: (a) ligue o 'auto_ajustar' e eu redimensiono sozinho; "
                                f"(b) ponha esses numeros no node nativo; "
                                f"(c) DESLIGUE o 'reference_latent' -- ai eu monto o container do zero e nao "
                                f"preciso casar shape.")
                    if strict:
                        raise ValueError(msg)
                    print(msg + "\n[Bruxos H3] strict=off: seguindo assim mesmo (provavel ruido).", flush=True)

            saida = list(comps)
            saida[0] = novo_video.to(dtype=ref_video.dtype, device=ref_video.device)
            trocou = ["video"]
            if novo_audio is not None and len(comps) > 1:
                ref_audio = comps[1]
                if tuple(novo_audio.shape) == tuple(ref_audio.shape):
                    saida[1] = novo_audio.to(dtype=ref_audio.dtype, device=ref_audio.device)
                    trocou.append("audio")
                else:
                    print(f"[Bruxos H3] audio {tuple(novo_audio.shape)} != referencia {tuple(ref_audio.shape)} "
                          f"-> MANTENDO o audio da referencia (confira o fps).", flush=True)

            container = _reconstruir(ref, saida)
            info = (f"[referencia] container={type(ref).__name__} | comps={len(saida)} | "
                    f"trocado={'+'.join(trocou)} | video={tuple(novo_video.shape)}")
            out = dict(reference_latent) if isinstance(reference_latent, dict) else {}
            out["samples"] = container
            print(f"[Bruxos H3] pronto: {info}", flush=True)
            return (out, info)

        # =================================================================
        # CAMINHO B: sem referencia -> monta o container do zero
        # =================================================================
        if novo_audio is None:
            raise ValueError(
                "[Bruxos H3] sem 'reference_latent' eu preciso do 'audio_vae' pra montar o container.\n"
                "O H3 e um modelo AUDIO+VIDEO: o sampler espera os DOIS latentes. Mesmo sem querer audio, "
                "ligue o minimax_h3_audio_vae -- eu codifico silencio nele.\n"
                "Alternativa: ligue a saida LATENT do node nativo 'MiniMax H3 Image to Video' em 'reference_latent'."
            )

        novo_audio = novo_audio.to(device=novo_video.device)
        partes = [novo_video, novo_audio]
        container, como = None, None
        try:
            import comfy.nested_tensor as _nt
            container = _nt.NestedTensor(partes)
            como = "NestedTensor"
        except Exception as e:
            print(f"[Bruxos H3] NestedTensor indisponivel ({e}); usando lista simples.", flush=True)
            container, como = partes, "lista"

        info = (f"[do zero] container={como} | video={tuple(novo_video.shape)} | "
                f"audio={tuple(novo_audio.shape)} ({origem_audio})")
        print(f"[Bruxos H3] pronto: {info}", flush=True)
        print("[Bruxos H3] DICA: se o sampler reclamar do formato, ligue a saida LATENT do node nativo "
              "'MiniMax H3 Image to Video' em 'reference_latent' -- ai o container vem pronto dele.", flush=True)
        return ({"samples": container}, info)


NODE_CLASS_MAPPINGS = {
    "BruxosMinimaxH3LatentInspect": BruxosMinimaxH3LatentInspect,
    "BruxosMinimaxH3EncodeVideo": BruxosMinimaxH3EncodeVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosMinimaxH3LatentInspect": "Inspetor de LATENT (Bruxos)",
    "BruxosMinimaxH3EncodeVideo": "MiniMax H3 · Encode Video -> Latente (Bruxos)",
}
