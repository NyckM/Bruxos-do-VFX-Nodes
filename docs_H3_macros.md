# MiniMax H3 · Prompt Rápido — todas as macros

Referência do node **MiniMax H3 · Prompt Rápido @ # (Bruxos)**
(`bruxos_h3_prompt_rapido.py`). Nenhuma macro carrega modelo — é substituição
de texto, roda em milissegundos.

**Acento não importa.** `#cenário` e `#cenario` são a mesma coisa; o node
normaliza antes de comparar. Maiúscula também não importa.

**Número é opcional.** `#camera` amarra em `<Video 1>`; `#camera2` amarra em
`<Video 2>`.

---

## Referências — `@`

| Escrita | Vira | Sinônimos aceitos |
|---|---|---|
| `@img` `@img2` `@1` | `<Picture 1>` `<Picture 2>` `<Picture 1>` | `img` `imagem` `image` `picture` `pic` `foto` `p` `i` |
| `@video` `@v2` | `<Video 1>` `<Video 2>` | `video` `vid` `v` `clipe` `clip` |
| `@audio` `@a3` | `<Audio 1>` `<Audio 3>` | `audio` `som` `sound` `a` `aud` |

`@2` sem letra assume **imagem**, que é o caso comum.

## Fala — `#d:...#`

```
#d:Você não devia ter vindo aqui.#
```

vira `<d>Você não devia ter vindo aqui.</d>`. O `<d>` é o token especial que o
tokenizer do H3 usa para diálogo — escrever a fala fora dele faz o modelo tratar
como descrição, não como palavra dita.

## Escapes

`@@` e `##` produzem um `@` ou `#` literal.

---

# As 18 macros `#`

## Imagens → `<Picture N>`

### `#frame`
`frame` `keyframe` `primeiro` `ultimo` `first` `last` `ancora` `anchor`

Diz que a imagem **é o primeiro quadro** do vídeo — o movimento parte dali.

> The shot begins from `<Picture 1>`: its framing, subject placement and lighting are the first frame of the target video, and the motion continues from there.

**Use quando:** você tem o frame inicial pronto e quer que o vídeo continue dele.

---

### `#sujeito`
`sujeito` `subject` `personagem` `character` `pessoa` `char`

Fixa **identidade** — rosto, idade, corpo, cabelo, roupa, cores.

> The character in `<Picture 1>` keeps the identity shown there -- face, age, build, hair, clothing and colours stay the same in every shot.

**Use quando:** a mesma pessoa/criatura precisa atravessar o clipe sem mudar.

---

### `#cenario`
`cenario` `scene` `environment` `local` `lugar` `setting`

Pega **material, paleta e luz** do lugar.

> The environment takes its materials, palette, surface treatment and lighting from `<Picture 1>`.

**Use quando:** não existe vídeo de origem e o ambiente vem de uma imagem.
Se existe vídeo com a geometria, prefira este + `#previz` no vídeo — assim a
imagem dá só aparência e o vídeo dá a forma.

---

### `#fundo`
`fundo` `background` `bg` `trocarfundo` `trocadefundo` `novofundo`

**Troca o fundo** de um vídeo existente. É a macro mais longa porque o difícil
não é o fundo — é o sujeito responder à luz nova.

> The environment of `<Picture 1>` replaces the original background, and its light acts as the real light in the scene: the key shifts to its colour temperature, the silhouette picks up its rim light, shadow sides fill with its ambient, and contact shadows and reflections follow it. Any flicker or movement of that light shows on the subject over time. The subject's identity, hair strands, fabric edges, gesture timing, scale and framing are untouched, and the matte edge stays clean.

**Use quando:** o sujeito vem do vídeo e o ambiente vem da imagem. É o que
evita o resultado parecer recorte colado.

---

### `#estilo`
`estilo` `style` `look` `arte`

Só o **tratamento visual** — traço, paleta, grão, meio.

> The visual treatment of `<Picture 1>` -- medium, linework, palette, grain and rendering -- applies to the whole video. Its subject matter is not reproduced.

**Use quando:** a imagem é referência de arte, não de conteúdo. A última frase
é o que impede o assunto dela de vazar para o vídeo.

---

## Vídeos → `<Video N>`

### `#camera`
`camera` `movimento` `motion` `layout` `ritmo`

Pega **só o movimento e o enquadramento**. A aparência é descartada.

> The camera movement, framing, staging and shot timing follow `<Video 1>`. Every material, colour, texture and light comes from the image references instead, so the flat preview look of `<Video 1>` does not appear. Each block or proxy volume in `<Video 1>` resolves into a finished subject consistent with those images, and the areas `<Video 1>` leaves open stay open.

**Use quando:** você quer emprestar a trajetória de câmera e mais nada.

---

### `#previz`
`previz` `blocking` `blocagem` `clay` `argila` `blockout` `greybox` `proxy`

O render de blocking do Blender usado como **fonte de edição**. Trava geometria
e câmera, joga fora o cinza sem textura.

> The camera path, staging and layout follow `<Video 1>` exactly, and every solid form in the target corresponds to a form present in `<Video 1>` at that same moment -- where `<Video 1>` is open ground, sky or empty space, the target stays open as well. The appearance of `<Video 1>` does not carry over: its untextured grey surfaces, flat preview lighting and placeholder detail are replaced by the materials, colour and lighting of the image references. The description runs through to the last frame of `<Video 1>`.

**Use quando:** blocking do Blender. Diferente de `#camera` porque aqui a
**geometria é lei** — é o que impede o modelo de inventar prédio onde você não
modelou.

---

### `#fonte`
`fonte` `source` `edicao` `editar` `edit` `base`

O vídeo **real** que está sendo editado. Preserva o look dele.

> The target video is an edited version of `<Video 1>`. Its framing, lighting and setting are preserved while the described change is applied, and every solid form in the target matches a form already present in `<Video 1>` -- open ground, sky and empty space in `<Video 1>` stay that way. The description runs through to the last frame.

**Use quando:** filmagem de verdade recebendo uma alteração pontual.

---

### `#continuar`
`continuar` `continuacao` `continuation` `extender` `continue`

O novo vídeo **começa onde o outro parou**.

> The target video continues directly from the end of `<Video 1>`, picking up the same subjects, lighting, camera motion and pace at the moment `<Video 1>` stops.

**Use quando:** encadeando clipes de 8 s para fazer um plano longo.

---

### `#relight`
`relight` `reluz` `relighting` `iluminar`

Trava movimento e identidade, **libera luz e ambiente**.

> The target video is a relit version of `<Video 1>`. The camera path, framing and composition are identical to `<Video 1>` frame for frame, and the subject's motion, pose, gesture timing, expression and gaze are carried over exactly. What changes is the light and the environment. The new light acts as the actual light in the scene on the existing subject: the key shifts to its colour temperature, the silhouette picks up its rim light, shadow sides fill with its ambient, and contact shadows and reflections follow it. Any flicker or movement of that light shows on the subject over time, in sync with the source. Identity, hair strands, fabric edges, occlusions and scale are untouched, and the matte edge stays clean.

**Use quando:** o fundo muda e a cena inteira é reiluminada, mas a atuação e a
câmera não podem mudar um frame. É o `#fonte` **sem** a cláusula que preserva
iluminação — que era justamente o que impedia relight.

**Descreva a luz variando no tempo.** Luz escrita como estado fixo sai estática;
o flicker é o que vende fogo, e ele precisa estar no seu texto.

---

### `#depth`
`depth` `profundidade` `depthmap` `zdepth`

O vídeo é um **mapa de profundidade**: só geometria, o cinza não vira look.

> `<Video 1>` is a depth map, not footage: its brightness encodes distance from camera, near being bright and far being dark. Read it for spatial layout only -- the volume and placement of the subject, the depth of the environment behind it, the occlusion order between them, and how all of that shifts as the camera moves. The greyscale itself is not a look and does not appear in the target video: every material, colour, texture and light comes from the image references and the description instead. Solid form in the target follows the solid form in `<Video 1>`, for both the subject and the background.

**Use quando:** você gerou o depth com o Depth Anything V2 e quer que ele guie
sujeito **e** fundo.

Repare que a macro **abre dizendo o que o vídeo é**, antes de dizer o que fazer
com ele. O erro clássico é o modelo tratar o cinza como aparência e devolver um
vídeo acinzentado — a primeira frase existe para impedir isso.

---

### `#st`
`st` `styletransfer` `restyle` `estilizar` `transferencia` `restilizar`

Style transfer: trava a **estrutura inteira**, troca só o tratamento visual.

> The target video is a restyled version of `<Video 1>`. Camera path, framing, composition, subject identity, motion, gesture timing and expression are carried over from `<Video 1>` exactly, frame for frame, and nothing is added to or removed from the scene. What changes is the rendering treatment alone -- medium, linework, palette, shading, texture and grain -- applied evenly across subject and environment so both belong to the same image. Edges, occlusions and scale stay where they are.

**Use quando:** o vídeo vira desenho, pintura, anime — e nada mais pode mudar.

O `applied evenly across subject and environment` é o que evita o resultado mais
comum de style transfer malfeito: personagem estilizado sobre fundo fotográfico.

---

## Os três de vídeo que se parecem

| | trava | libera |
|---|---|---|
| `#st` | a estrutura inteira | o tratamento visual |
| `#relight` | movimento e identidade | luz e ambiente |
| `#depth` | só a geometria | tudo o mais |

`#fonte` fica de fora dessa tabela porque preserva **também** a iluminação — use
ele para alteração pontual em filmagem real, não para relight nem estilização.

---

## Áudios → `<Audio N>`

### `#voz`
`voz` `voice` `timbre` `fala`

Copia o **timbre**, não as palavras.

> The speaking voice matches the timbre, age and accent heard in `<Audio 1>`. The words spoken are the ones written in this prompt, not the words in `<Audio 1>`.

Combine com `#d:...#` para dizer o que a voz fala.

---

### `#trilha`
`trilha` `musica` `music` `bgm` `score`

> The background music of `<Audio 1>` plays under the shot, at its original tempo and mood.

---

### `#copiar`
`copiar` `copy` `igual` `asis` `manter`

> The audio of `<Audio 1>` is reproduced as it is.

---

### `#ambiente`
`ambiente` `ambience` `sfx` `efeitos` `room`

> The ambience and sound effects of `<Audio 1>` fill the scene.

---

## Sem referência

### `#invariancia`
`invariancia` `inv` `invar` `fecho`

> Apart from the changes described above, character identity, motion trajectory, camera path, composition, environment and sound all remain unchanged.

**Use sempre no fim** quando estiver editando algo. As seis coisas são nomeadas
uma a uma de propósito — "todo o resto continua igual" é vago demais e o modelo
deriva.

---

### `#semlegenda`
`semlegenda` `limpo` `semtexto` `notext`

> The frame carries no subtitles, captions, watermark, logo or on-screen text.

**Use quando:** aparecer legenda ou marca d'água fantasma no resultado.

---

# Exemplos

**Previz do Blender + arte:**
```
#previz  #sujeito2  #estilo3
Ancient city gate at dusk, torchlight raking across weathered stone.
#invariancia  #semlegenda
```

**Troca de fundo com fala:**
```
#fonte  #fundo2  #voz
She turns toward the window. #d:They're already here.#
#invariancia
```

**Encadeando planos:**
```
#continuar  #estilo2
The camera keeps drifting left as the crowd thins out.
```

---

# Limites do H3

- `<Picture>` até **9**, `<Video>` até **3**, `<Audio>` até **3**
- mas **no máximo 12 arquivos no total** — não dá para usar 9 + 3 + 3
- vídeo de referência: 2 a 15 s cada
- áudio nunca sozinho, sempre acompanhando imagem ou vídeo
- prompt até **7000 caracteres**
- resolução múltipla de 32

O node valida cada macro contra o teto **por tipo**. O total de 12 ele não
consegue checar, porque só enxerga o texto — essa conta é sua.

---

# Relação com o node Context-IR

`BruxosH3ContextIR` escreve as seis seções do Context-IR com um Qwen3 local:
é mais completo e muito mais lento. Este node é o caminho rápido.

Os textos dos papéis **não são os mesmos**. Lá são ordens para o LLM que
escreve o prompt ("Do NOT give it its own `<Picture N>` entry"); aqui são
conteúdo que o H3 lê direto. Mesmo conhecimento, leitores diferentes — se você
editar um catálogo, releia o outro.

---

*A ideia das macros `@` e `#` veio do
[ComfyUI-MiniMaxH3-Easy](https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy)
(nkxx188, MIT). Lá é um editor visual em JavaScript; aqui é macro em texto puro
sobre os papéis do pacote Bruxos.*
