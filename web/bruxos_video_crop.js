// Bruxos do VFX — box de corte ARRASTAVEL sobre o preview de video
// =================================================================
// O 'Load Image (Bruxos)' ja tinha o box de corte visual; o 'Load Video'
// tinha os campos crop_x/y/w/h mas nenhum feedback -- voce mexia no numero
// sem ver o que estava enquadrando. Este arquivo desenha o mesmo box em cima
// do <video> do preview.
//
// Detalhe que importa: o <video> usa object-fit:contain, entao a imagem
// aparece LETTERBOXED dentro do elemento (sobra barra preta em cima/baixo ou
// nas laterais). Se desenhassemos o box sobre o elemento inteiro, ele ficaria
// deslocado em relacao ao video de verdade. Por isso calculamos o retangulo
// EXIBIDO a partir de videoWidth/videoHeight e desenhamos so ali dentro.
//
// O box so aparece quando fit_mode != "off (original)" -- fora disso o corte
// nao e aplicado e mostrar o box so confundiria.

import { app } from "../../scripts/app.js";

const COR_BOX = "#7df49a";
const COR_ALCA = "#ffffff";
const ALCA = 7;          // raio da alca em px
const MIN = 0.02;        // tamanho minimo do box (fracao)

function w(node, nome) {
  return (node.widgets || []).find((x) => x.name === nome);
}
function val(node, nome, def) {
  const x = w(node, nome);
  const v = x ? parseFloat(x.value) : NaN;
  return Number.isFinite(v) ? v : def;
}
function setVal(node, nome, v) {
  const x = w(node, nome);
  if (!x) return;
  const nv = Math.round(v * 1000) / 1000;
  if (x.value === nv) return;
  x.value = nv;
  try { x.callback?.(nv); } catch (e) {}
}

// "16:9" -> 16/9 ; "livre" -> null
function razao(node) {
  const a = w(node, "aspect");
  const s = a ? String(a.value) : "livre";
  const m = s.match(/(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)/i);
  if (!m) return null;
  const r = parseFloat(m[1]) / parseFloat(m[2]);
  return Number.isFinite(r) && r > 0 ? r : null;
}

function cortando(node) {
  const f = w(node, "fit_mode");
  return f ? !String(f.value).startsWith("off") : false;
}

// retangulo REALMENTE ocupado pelo video dentro do elemento (object-fit:contain)
function areaVideo(v) {
  const ew = v.clientWidth || 1, eh = v.clientHeight || 1;
  const vw = v.videoWidth || 16, vh = v.videoHeight || 9;
  const s = Math.min(ew / vw, eh / vh);
  const dw = vw * s, dh = vh * s;
  return { x: (ew - dw) / 2, y: (eh - dh) / 2, w: dw, h: dh, ew, eh, vw, vh };
}

function montar(node) {
  const prev = node._bruxosPrev;
  if (!prev || node._bxCrop) return node._bxCrop;

  const { wrap, video } = prev;
  wrap.style.position = "relative";

  const cv = document.createElement("canvas");
  cv.style.cssText =
    "position:absolute;left:0;top:0;pointer-events:auto;cursor:crosshair;" +
    "border-radius:6px;";
  // o canvas fica ENTRE o video e os controles nativos do <video>
  wrap.insertBefore(cv, video.nextSibling);

  const st = { cv, ctx: cv.getContext("2d"), drag: null };
  node._bxCrop = st;

  const desenhar = () => {
    if (!video.isConnected) return;
    const r = video.getBoundingClientRect();
    const wr = wrap.getBoundingClientRect();
    // posiciona o canvas exatamente sobre o elemento de video
    cv.style.left = (r.left - wr.left) + "px";
    cv.style.top = (r.top - wr.top) + "px";
    const W = Math.max(1, Math.round(r.width));
    const H = Math.max(1, Math.round(r.height));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }

    const ctx = st.ctx;
    ctx.clearRect(0, 0, W, H);
    if (!cortando(node)) { cv.style.pointerEvents = "none"; return; }
    cv.style.pointerEvents = "auto";

    const a = areaVideo(video);
    const cx = val(node, "crop_x", 0), cy = val(node, "crop_y", 0);
    const cw = val(node, "crop_w", 1), ch = val(node, "crop_h", 1);
    const bx = a.x + cx * a.w, by = a.y + cy * a.h;
    const bw = cw * a.w, bh = ch * a.h;

    // escurece o que fica FORA do corte (dentro da area do video)
    ctx.save();
    ctx.beginPath();
    ctx.rect(a.x, a.y, a.w, a.h);
    ctx.rect(bx, by, bw, bh);
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fill("evenodd");
    ctx.restore();

    // moldura + tercos
    ctx.strokeStyle = COR_BOX; ctx.lineWidth = 2;
    ctx.strokeRect(bx + 1, by + 1, bw - 2, bh - 2);
    ctx.strokeStyle = "rgba(125,244,154,0.35)"; ctx.lineWidth = 1;
    for (let i = 1; i < 3; i++) {
      ctx.beginPath();
      ctx.moveTo(bx + (bw * i) / 3, by); ctx.lineTo(bx + (bw * i) / 3, by + bh);
      ctx.moveTo(bx, by + (bh * i) / 3); ctx.lineTo(bx + bw, by + (bh * i) / 3);
      ctx.stroke();
    }

    // alcas
    ctx.fillStyle = COR_ALCA; ctx.strokeStyle = "#111"; ctx.lineWidth = 1;
    for (const [px, py] of [
      [bx, by], [bx + bw / 2, by], [bx + bw, by],
      [bx, by + bh / 2], [bx + bw, by + bh / 2],
      [bx, by + bh], [bx + bw / 2, by + bh], [bx + bw, by + bh],
    ]) {
      ctx.beginPath(); ctx.arc(px, py, ALCA / 2 + 1, 0, 6.284);
      ctx.fill(); ctx.stroke();
    }

    // legenda: resolucao resultante do corte
    const rw = Math.round(a.vw * cw), rh = Math.round(a.vh * ch);
    const rot = razao(node);
    const txt = `${rw}x${rh}` + (rot ? `  (${(w(node,"aspect")||{}).value})` : "");
    ctx.font = "11px monospace";
    const tw = ctx.measureText(txt).width;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(bx + 3, by + 3, tw + 8, 16);
    ctx.fillStyle = COR_BOX;
    ctx.fillText(txt, bx + 7, by + 15);
  };

  st.desenhar = desenhar;

  // ---- interacao -------------------------------------------------------
  const modoEm = (mx, my) => {
    const a = areaVideo(video);
    const cx = val(node, "crop_x", 0), cy = val(node, "crop_y", 0);
    const cw = val(node, "crop_w", 1), ch = val(node, "crop_h", 1);
    const bx = a.x + cx * a.w, by = a.y + cy * a.h;
    const bw = cw * a.w, bh = ch * a.h;
    const perto = (px, py) => Math.abs(mx - px) <= ALCA && Math.abs(my - py) <= ALCA;
    if (perto(bx, by)) return "nw";
    if (perto(bx + bw, by)) return "ne";
    if (perto(bx, by + bh)) return "sw";
    if (perto(bx + bw, by + bh)) return "se";
    if (perto(bx + bw / 2, by)) return "n";
    if (perto(bx + bw / 2, by + bh)) return "s";
    if (perto(bx, by + bh / 2)) return "w";
    if (perto(bx + bw, by + bh / 2)) return "e";
    if (mx > bx && mx < bx + bw && my > by && my < by + bh) return "move";
    return null;
  };

  const pos = (e) => {
    const r = cv.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  cv.addEventListener("pointermove", (e) => {
    if (st.drag) return;
    const m = modoEm(...pos(e));
    cv.style.cursor = !m ? "crosshair" : m === "move" ? "move"
      : (m === "n" || m === "s") ? "ns-resize"
      : (m === "e" || m === "w") ? "ew-resize"
      : (m === "nw" || m === "se") ? "nwse-resize" : "nesw-resize";
  });

  cv.addEventListener("pointerdown", (e) => {
    if (!cortando(node)) return;
    const [mx, my] = pos(e);
    const m = modoEm(mx, my);
    if (!m) return;
    e.stopPropagation(); e.preventDefault();
    cv.setPointerCapture(e.pointerId);
    st.drag = {
      modo: m, mx, my,
      x: val(node, "crop_x", 0), y: val(node, "crop_y", 0),
      w: val(node, "crop_w", 1), h: val(node, "crop_h", 1),
    };
  });

  cv.addEventListener("pointermove", (e) => {
    if (!st.drag) return;
    e.stopPropagation(); e.preventDefault();
    const a = areaVideo(video);
    const [mx, my] = pos(e);
    const dx = (mx - st.drag.mx) / a.w, dy = (my - st.drag.my) / a.h;
    const d = st.drag;
    let x = d.x, y = d.y, ww = d.w, hh = d.h;

    if (d.modo === "move") {
      x = Math.min(Math.max(0, d.x + dx), 1 - d.w);
      y = Math.min(Math.max(0, d.y + dy), 1 - d.h);
    } else {
      let l = d.x, t = d.y, rr = d.x + d.w, bb = d.y + d.h;
      if (d.modo.includes("w")) l = Math.min(d.x + dx, rr - MIN);
      if (d.modo.includes("e")) rr = Math.max(d.x + d.w + dx, l + MIN);
      if (d.modo.includes("n")) t = Math.min(d.y + dy, bb - MIN);
      if (d.modo.includes("s")) bb = Math.max(d.y + d.h + dy, t + MIN);
      l = Math.max(0, l); t = Math.max(0, t);
      rr = Math.min(1, rr); bb = Math.min(1, bb);
      x = l; y = t; ww = rr - l; hh = bb - t;

      // trava de proporcao: usa a razao em PIXELS (a fracao sozinha ignora
      // que o video nao e quadrado -- daria 16:9 errado).
      const rot = razao(node);
      if (rot) {
        const px = a.vw, py = a.vh;
        // h_frac = (w_frac * px) / (rot * py)
        hh = (ww * px) / (rot * py);
        if (hh > 1) { hh = 1; ww = (hh * rot * py) / px; }
        if (d.modo.includes("n")) y = Math.max(0, bb - hh);
        if (y + hh > 1) y = 1 - hh;
        if (x + ww > 1) x = 1 - ww;
      }
    }
    setVal(node, "crop_x", x); setVal(node, "crop_y", y);
    setVal(node, "crop_w", ww); setVal(node, "crop_h", hh);
    desenhar();
    node.setDirtyCanvas?.(true, true);
  });

  const soltar = (e) => {
    if (!st.drag) return;
    st.drag = null;
    try { cv.releasePointerCapture(e.pointerId); } catch (err) {}
  };
  cv.addEventListener("pointerup", soltar);
  cv.addEventListener("pointercancel", soltar);

  // redesenha continuamente enquanto o node existir (o video muda de tamanho,
  // o usuario troca de aspect/fit_mode, etc.)
  const laco = () => {
    if (!wrap.isConnected) return;
    try { desenhar(); } catch (e) {}
    st.raf = requestAnimationFrame(laco);
  };
  st.raf = requestAnimationFrame(laco);
  return st;
}

app.registerExtension({
  name: "BruxosDoVFX.VideoCropBox",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "BruxosLoadVideo") return;
    const criado = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = criado?.apply(this, arguments);
      // o preview e criado pela outra extensao; esperamos ele existir
      let n = 0;
      const tenta = () => {
        if (this._bruxosPrev) { try { montar(this); } catch (e) { console.error("[Bruxos crop video]", e); } return; }
        if (n++ < 60) setTimeout(tenta, 50);
      };
      tenta();
      return r;
    };
    const removido = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      try { if (this._bxCrop?.raf) cancelAnimationFrame(this._bxCrop.raf); } catch (e) {}
      this._bxCrop = null;
      return removido?.apply(this, arguments);
    };
  },
});
