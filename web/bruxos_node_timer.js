import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Bruxos do VFX - Timer por node.
// Mostra, em cada node, quanto ele levou pra rodar -- automaticamente, sem
// precisar plugar nenhum node de cronometro no fluxo.
//   - ROXO + "▶" enquanto o node esta rodando (conta ao vivo)
//   - VERDE com o tempo final quando termina
// Segundos ate 60s; depois minutos (ex.: 4m11s). Persiste ate a proxima run.
console.log("[Bruxos] timer por node carregado");

const COR_RODANDO = "#a855f7"; // roxo
const COR_PRONTO  = "#22c55e"; // verde
const COR_TEXTO   = "#0b0b0d";

const tempos = {};      // nodeId -> { start, end }  (em ms, performance.now)
let rodando = null;     // id do node em execucao
let rafOn = false;
let ligado = true;      // liga/desliga o selo (canvas, Node 1)
let modoTitulo = true;  // escreve o tempo no titulo (funciona no Node 2.0/Vue)
let _ultTitulo = 0;     // ultimo instante em que atualizei o titulo (throttle)

const MARK = " \u23F1";  // delimitador do sufixo no titulo (relogio)

function fmt(seg) {
  if (seg < 60) return seg.toFixed(1) + "s";
  const m = Math.floor(seg / 60);
  const s = seg - m * 60;
  return m + "m" + (s < 10 ? "0" : "") + s.toFixed(1) + "s";
}

// ---- tempo no titulo do node (render-mode-agnostico: Node 1 E Node 2.0) ----
function tituloBase(node) {
  if (node.__bruxosBaseTitle == null) {
    const t = node.title || "";
    node.__bruxosBaseTitle = t.split(MARK)[0].replace(/\s+$/, "");
  }
  return node.__bruxosBaseTitle;
}
function setTituloTempo(node, txt) {
  if (!modoTitulo) return;
  const base = tituloBase(node);
  try { node.title = txt ? `${base}${MARK} ${txt}` : base; } catch (e) {}
}
function restauraTitulos() {
  const nodes = app.graph?._nodes || [];
  for (const node of nodes) {
    if (node.__bruxosBaseTitle != null) {
      try { node.title = node.__bruxosBaseTitle; } catch (e) {}
    }
  }
  app.graph?.setDirtyCanvas(true, false);
}

// enquanto algo roda, redesenha p/ o contador subir ao vivo
// enquanto algo roda, atualiza o tempo AO VIVO (no titulo p/ Node 2.0, e redesenha
// o canvas p/ o selo do Node 1)
function tick() {
  if (rodando != null) {
    const t = tempos[rodando];
    if (t && t.end == null) {
      const agora = performance.now();
      if (agora - _ultTitulo > 250) {   // ~4x por segundo (nao pesa o Vue)
        _ultTitulo = agora;
        const node = app.graph?.getNodeById?.(Number(rodando));
        if (node) setTituloTempo(node, "\u25B6 " + fmt((agora - t.start) / 1000));
      }
    }
    app.graph?.setDirtyCanvas(true, false);
    requestAnimationFrame(tick);
  } else {
    rafOn = false;
    app.graph?.setDirtyCanvas(true, false); // desenho final
  }
}
function garanteRaf() {
  if (!rafOn) { rafOn = true; requestAnimationFrame(tick); }
}

// ---- eventos de execucao do ComfyUI ----
api.addEventListener("execution_start", () => {
  if (rodando != null && tempos[rodando]) tempos[rodando].end = performance.now();
  rodando = null;
  restauraTitulos();
});

api.addEventListener("executing", (e) => {
  const id = e.detail;             // id do node, ou null quando termina tudo
  const agora = performance.now();
  if (rodando != null && tempos[rodando]) {
    tempos[rodando].end = agora;
    const nprev = app.graph?.getNodeById?.(Number(rodando));
    if (nprev) setTituloTempo(nprev, fmt((tempos[rodando].end - tempos[rodando].start) / 1000));
  }
  if (id === null || id === undefined) { rodando = null; resumo(); return; }
  tempos[id] = { start: agora, end: null };
  rodando = id;
  const node = app.graph?.getNodeById?.(Number(id));
  if (node) setTituloTempo(node, "\u25B6");   // ▶ rodando
  garanteRaf();
});

// resumo ordenado no console (sem precisar de node nenhum no fluxo; funciona em qualquer modo)
function resumo() {
  const linhas = [];
  let total = 0;
  for (const id in tempos) {
    const t = tempos[id];
    if (t.end == null) continue;
    const seg = (t.end - t.start) / 1000;
    total += seg;
    const node = app.graph?.getNodeById?.(Number(id));
    const nome = node ? (tituloBase(node) || node.type || id) : id;
    linhas.push({ node: nome, tempo: fmt(seg), seg });
  }
  if (!linhas.length) return;
  linhas.sort((a, b) => b.seg - a.seg);
  console.log("%c[Bruxos] tempo por node (mais lento primeiro):", "color:#a855f7;font-weight:bold");
  console.table(linhas.map(({ node, tempo }) => ({ node, tempo })));
  console.log("[Bruxos] TOTAL:", fmt(total));
}

api.addEventListener("executed", (e) => {
  const id = e.detail?.node;
  if (id != null && tempos[id] && tempos[id].end == null) {
    tempos[id].end = performance.now();
    const node = app.graph?.getNodeById?.(Number(id));
    if (node) setTituloTempo(node, fmt((tempos[id].end - tempos[id].start) / 1000));
  }
});

// ---- desenho do selo (compartilhado) ----
// Desenha a pilula no ctx atual, com o canto inferior direito em (ox+w, oy+h).
function desenhaSeloEm(node, ctx, ox, oy, w, h) {
  const t = tempos[node.id];
  if (!t) return;
  const emExec = t.end == null;
  const fim = t.end == null ? performance.now() : t.end;
  const seg = Math.max(0, (fim - t.start) / 1000);
  const label = (emExec ? "\u25B6 " : "") + fmt(seg);

  ctx.save();
  ctx.font = "600 11px monospace";
  const padX = 6, hh = 16;
  const ww = ctx.measureText(label).width + padX * 2;
  const x = ox + w - ww - 6;
  // na linha do rodape (borda de baixo), ao lado do badge "Bruxos-do-VFX"
  const y = oy + h - hh / 2;

  const r = 7;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + ww, y, x + ww, y + hh, r);
  ctx.arcTo(x + ww, y + hh, x, y + hh, r);
  ctx.arcTo(x, y + hh, x, y, r);
  ctx.arcTo(x, y, x + ww, y, r);
  ctx.closePath();
  ctx.fillStyle = emExec ? COR_RODANDO : COR_PRONTO;
  ctx.globalAlpha = 0.95;
  ctx.fill();

  ctx.globalAlpha = 1;
  ctx.fillStyle = COR_TEXTO;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + padX, y + hh / 2 + 0.5);
  ctx.restore();
}

// modo CLASSICO: cada node desenha o proprio selo (coords locais) e "carimba" o frame.
function desenhaSelo(node, ctx) {
  if (!ligado || !node || node.flags?.collapsed) return;
  if (!tempos[node.id]) return;
  desenhaSeloEm(node, ctx, 0, 0, node.size[0], node.size[1]);
  node.__bruxosStamp = performance.now();
}

// modo NODE 2.0 (fallback): desenha no nivel do canvas os nodes que o per-node
// NAO desenhou neste frame (evita duplicar quando o classico ja desenhou).
function desenhaSelosCanvas(ctx) {
  if (!ligado) return;
  const nodes = app.graph?._nodes || [];
  const agora = performance.now();
  for (const node of nodes) {
    if (!node || node.flags?.collapsed) continue;
    if (!tempos[node.id]) continue;
    if (agora - (node.__bruxosStamp || 0) < 60) continue; // per-node ja desenhou
    desenhaSeloEm(node, ctx, node.pos[0], node.pos[1], node.size[0], node.size[1]);
  }
}

app.registerExtension({
  name: "Bruxos.NodeTimer",
  async beforeRegisterNodeDef(nodeType) {
    // encadeia em TODO node (Bruxos, SAM3, Bernini, core...), sem quebrar o desenho original
    const orig = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      orig?.apply(this, arguments);
      try { desenhaSelo(this, ctx); } catch (e) { /* nunca quebra o canvas */ }
    };
  },
  setup() {
    // toggle na engrenagem de configuracoes
    app.ui?.settings?.addSetting?.({
      id: "Bruxos.NodeTimer.enabled",
      name: "Bruxos: mostrar timer em cada node (selo, Node 1)",
      type: "boolean",
      defaultValue: true,
      onChange: (v) => { ligado = !!v; app.graph?.setDirtyCanvas(true, false); },
    });
    app.ui?.settings?.addSetting?.({
      id: "Bruxos.NodeTimer.titleMode",
      name: "Bruxos: mostrar tempo no titulo do node (funciona no Node 2.0)",
      type: "boolean",
      defaultValue: true,
      onChange: (v) => { modoTitulo = !!v; if (!v) restauraTitulos(); },
    });

    // Fallback p/ Node 2.0: desenha no nivel do canvas (o per-node onDrawForeground
    // as vezes nao e chamado no render novo). Patch no prototipo p/ sobreviver a
    // recriacao do canvas. O dedupe por __bruxosStamp evita desenhar 2x no classico.
    try {
      const LGC = app.canvas?.constructor;
      if (LGC && !LGC.prototype.__bruxosTimerPatched) {
        const orig = LGC.prototype.onDrawForeground;
        LGC.prototype.onDrawForeground = function (ctx, visible_rect) {
          orig?.call(this, ctx, visible_rect);
          try { desenhaSelosCanvas(ctx); } catch (e) { /* nunca quebra o canvas */ }
        };
        LGC.prototype.__bruxosTimerPatched = true;
      }
    } catch (e) {
      console.warn("[Bruxos] timer: hook de canvas nao instalado:", e);
    }
  },
});
