# H3 Context-IR — papéis de referência

## Sintaxe do campo `referencias`

```
tipo [papel]: descrição do que aparece
```

Uma por linha, **na mesma ordem dos slots** do node do H3:

```
ref_image_0  →  1ª linha  →  <Picture 1>
ref_image_1  →  2ª linha  →  <Picture 2>
ref_video_0  →  3ª linha  →  <Video 1>
ref_audio_0  →  4ª linha  →  <Audio 1>
```

Numeração é **por categoria**. Linha com `#` é ignorada.
Sem `[papel]` → o LLM decide lendo a sua `ideia`.

## IMAGEM

| papel | o que faz | sinônimos |
|---|---|---|
| **`sujeito`** | personagem, criatura ou objeto -> vira <Subject N> | `subject`, `personagem`, `character`, `pessoa` |
| **`cenario`** | ambiente, lugar, arquitetura -> vira <Subject N> | `scene`, `environment`, `ambiente_visual`, `local` |
| **`estilo`** | so o look (traco, paleta, material) -> vira <Subject N> | `style`, `look`, `arte` |
| **`frame`** | frame inicial / keyframe / ultimo frame -> vira <Picture N> | `keyframe`, `primeiro`, `ultimo`, `first` |

## VÍDEO

| papel | o que faz | sinônimos |
|---|---|---|
| **`previz`** | render de blocking/clay do Blender como FONTE: trava geometria e camera, e JOGA FORA o cinza sem textura | `blocking`, `blocagem`, `clay`, `argila` |
| **`camera`** | so movimento de camera e ritmo; a geometria nao entra | `movimento`, `motion`, `layout`, `ritmo` |
| **`fonte`** | video REAL de origem que vai ser editado (preserva o look dele) | `source`, `edicao`, `editar`, `edit` |
| **`continuar`** | clipe que o video novo CONTINUA | `continuacao`, `continuation`, `extender`, `continue` |

## ÁUDIO

| papel | o que faz | sinônimos |
|---|---|---|
| **`voz`** | timbre de voz; nao copia as palavras | `voice`, `timbre`, `fala` |
| **`trilha`** | musica de fundo reaproveitada | `musica`, `music`, `bgm`, `score` |
| **`copiar`** | audio copiado igualzinho | `copy`, `igual`, `asis`, `manter` |
| **`ambiente`** | ambiencia e efeitos sonoros | `ambience`, `sfx`, `efeitos`, `room` |

## Exemplo — cidade antiga + blocking

```
imagem [frame]: cidade antiga vista do alto, muralha de pedra,
                torres de terracota, luz de fim de tarde
video [previz]: render de blocking do Blender, caixas cinza,
                cada bloco marca a posicao e altura de um edificio
```

## Os 3 papéis de vídeo — não confunda

| se o vídeo é... | use |
|---|---|
| blocking/clay do Blender que vira a cena | `previz` |
| só pra emprestar o movimento de câmera | `camera` |
| filmagem **real** que vai ser editada | `fonte` |

`previz` = geometria travada **+** cinza descartado.
`fonte` preserva o look do source — errado pra previz.

## Limites do H3

```
duração ....... 4–15 s          saída 24 fps, áudio 32 kHz
imagens ....... até 9           vídeos até 3 (2–15 s cada)
áudios ........ até 3, NUNCA sozinho (exige imagem ou vídeo)
total ......... 12 arquivos     prompt até 7000 caracteres
resolução ..... múltiplo de 32
```

**length ≡ 5 (mod 17)** → 5, 22, 39, 56, 73, 90, 107, 124, 141…
**8.0 s = 192 frames** é a única duração que cai em segundo exato.

## Checklist da saída

- [ ] 6 seções, nesta ordem: `subject_definitions`, `summary`,
      `retention_analysis`, `detailed_description`,
      `overall_soundscape`, `non_diegetic_music`
- [ ] `<Video 1>` marcado **`fully_preserved`** (não `weak_reference`)
- [ ] termina com a frase dos 6 eixos (invariância)
- [ ] nenhuma label citada que não esteja conectada
- [ ] saída `avisos` **vazia**