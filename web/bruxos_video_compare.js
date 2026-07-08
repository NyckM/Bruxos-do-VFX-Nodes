import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Comparar Vídeos A/B (Bruxos) - player embutido no node, estilo Deno.
console.log("[Bruxos] comparar videos carregado");

// pagina standalone (ajuste se mudar o usuario/repo do GitHub Pages)
const BROWSER_URL = "https://nyckm.github.io/Bruxos-do-VFX-Nodes/video-compare/";
const H = 300;

const ROXO = "#a855f7", VERDE = "#22c55e", VERDE2 = "#4ade80";

function viewURL(ref) {
  const sub = ref.subfolder ? encodeURIComponent(ref.subfolder) : "";
  return api.apiURL(`/view?filename=${encodeURIComponent(ref.filename)}` +
    `&type=${ref.type || "temp"}&subfolder=${sub}&rand=${Math.random().toString(36).slice(2)}`);
}

function btn(label, color) {
  const b = document.createElement("button");
  b.textContent = label;
  b.style.cssText = "background:#0e0e12;border:1px solid #1d1d24;color:#ddd;" +
    "padding:5px 10px;border-radius:8px;font-size:11px;font-weight:600;cursor:pointer;";
  if (color) { b.style.borderColor = color; b.style.color = color; }
  return b;
}

function ensureUI(node) {
  if (node._cmp) return node._cmp;

  const wrap = document.createElement("div");
  wrap.style.cssText = "width:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:6px;";

  // barra de modos
  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;align-items:center;";
  const modes = {};
  ["Cortina:slider", "Lado a Lado:side", "Diferença:diff", "Alternar:toggle"].forEach((m) => {
    const [label, key] = m.split(":");
    const b = btn(label);
    b.onclick = () => setMode(node, key);
    modes[key] = b; bar.appendChild(b);
  });
  const swap = btn("⇄ Trocar", ROXO); bar.appendChild(swap);
  const openB = btn("↗ Abrir no navegador", VERDE2);
  openB.onclick = () => window.open(BROWSER_URL, "_blank");
  bar.appendChild(openB);

  // palco
  const stage = document.createElement("div");
  stage.style.cssText = "position:relative;width:100%;height:" + H + "px;background:#000;" +
    "border-radius:8px;overflow:hidden;";
  const va = document.createElement("video");
  const vb = document.createElement("video");
  for (const v of [va, vb]) {
    v.muted = true; v.loop = true; v.playsInline = true;
    v.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;";
  }
  const ba = document.createElement("div"); ba.textContent = "A";
  ba.style.cssText = "position:absolute;top:8px;left:8px;width:26px;height:26px;border-radius:50%;" +
    "display:flex;align-items:center;justify-content:center;font-weight:800;background:" + VERDE + ";color:#04130a;z-index:6;";
  const bb = document.createElement("div"); bb.textContent = "B";
  bb.style.cssText = "position:absolute;top:8px;right:8px;width:26px;height:26px;border-radius:50%;" +
    "display:flex;align-items:center;justify-content:center;font-weight:800;background:" + ROXO + ";color:#150522;z-index:6;";
  const divider = document.createElement("div");
  divider.style.cssText = "position:absolute;top:0;bottom:0;width:2px;background:" + VERDE2 +
    ";z-index:5;pointer-events:none;box-shadow:0 0 8px rgba(74,222,128,.7);";
  stage.append(va, vb, ba, bb, divider);

  // controles
  const ctr = document.createElement("div");
  ctr.style.cssText = "display:flex;gap:6px;align-items:center;";
  const play = btn("▶ Play", VERDE);
  ctr.appendChild(play);

  wrap.append(bar, stage, ctr);
  wrap.style.width = "100%";
  wrap.style.maxWidth = "100%";
  wrap.style.boxSizing = "border-box";
  wrap.style.overflow = "hidden";
  const widget = node.addDOMWidget("bruxos_compare_ui", "compare", wrap, { serialize: false });
  widget.computeSize = (w) => [w, H + 76];

  // No Node 2.0 o container pai do DOMWidget pode ficar mais largo que o node
  // e o player vaza pela lateral. Forca, a cada desenho, a largura do wrap a
  // acompanhar a largura REAL do node (menos a margem da moldura).
  const _origDraw = node.onDrawForeground;
  node.onDrawForeground = function (ctx) {
    const r = _origDraw ? _origDraw.apply(this, arguments) : undefined;
    try {
      const alvo = Math.max(120, (this.size?.[0] || 240) - 26);
      if (wrap.parentElement) {
        wrap.parentElement.style.width = alvo + "px";
        wrap.parentElement.style.maxWidth = alvo + "px";
        wrap.parentElement.style.overflow = "hidden";
        wrap.parentElement.style.boxSizing = "border-box";
      }
      wrap.style.width = alvo + "px";
      wrap.style.maxWidth = alvo + "px";
    } catch (e) {}
    return r;
  };

  node._cmp = { wrap, bar, modes, stage, va, vb, divider, play, swap, mode: "slider", split: 0.5 };

  // interações
  setMode(node, "slider");
  let showB = false;
  function toggleShow() { showB = !showB; vb.style.opacity = showB ? "1" : "0"; }
  stage.addEventListener("mousemove", (e) => {
    if (node._cmp.mode !== "slider") return;
    const r = stage.getBoundingClientRect();
    setSplit(node, (e.clientX - r.left) / r.width);
  });
  stage.addEventListener("click", () => { if (node._cmp.mode === "toggle") toggleShow(); });
  play.onclick = () => {
    if (va.paused) { va.play().catch(()=>{}); vb.play().catch(()=>{}); play.textContent = "⏸ Pause"; }
    else { va.pause(); vb.pause(); play.textContent = "▶ Play"; }
  };
  swap.onclick = () => { const t = va.src; va.src = vb.src; vb.src = t; };
  // sincronia leve
  function sync() {
    if (!va.paused && Math.abs((vb.currentTime||0)-(va.currentTime||0))>0.08)
      vb.currentTime = va.currentTime;
    requestAnimationFrame(sync);
  }
  requestAnimationFrame(sync);

  return node._cmp;
}

function setSplit(node, x) {
  const c = node._cmp; c.split = Math.max(0, Math.min(1, x));
  const px = c.split * c.stage.clientWidth;
  c.vb.style.clipPath = `inset(0 0 0 ${px}px)`;
  c.divider.style.left = px + "px";
}

function setMode(node, key) {
  const c = ensureUI(node); c.mode = key;
  Object.entries(c.modes).forEach(([k, b]) => {
    b.style.borderColor = k === key ? VERDE : "#1d1d24";
    b.style.color = k === key ? VERDE2 : "#ddd";
  });
  const { va, vb, divider } = c;
  va.style.clipPath = ""; vb.style.clipPath = ""; vb.style.mixBlendMode = "";
  va.style.width = "100%"; vb.style.width = "100%"; va.style.left = "0"; vb.style.left = "0";
  vb.style.opacity = "1"; divider.style.display = "none";
  if (key === "slider") { divider.style.display = "block"; setSplit(node, c.split); }
  else if (key === "side") { va.style.width = "50%"; vb.style.width = "50%"; vb.style.left = "50%"; }
  else if (key === "diff") { vb.style.mixBlendMode = "difference"; }
  else if (key === "toggle") { vb.style.opacity = "0"; }
}

function loadVideos(node, data) {
  const c = ensureUI(node);
  if (data.a) { c.va.src = viewURL(data.a); c.va.load(); }
  if (data.b) { c.vb.src = viewURL(data.b); c.vb.load(); }
  c.va.play().catch(()=>{}); c.vb.play().catch(()=>{});
  c.play.textContent = "⏸ Pause";
  node.setSize(node.computeSize());
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "BruxosDoVFX.VideoCompare",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData && nodeData.name === "BruxosVideoCompare") {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onCreated ? onCreated.apply(this, arguments) : undefined;
        ensureUI(this);
        return r;
      };
      const onExec = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        if (onExec) onExec.apply(this, arguments);
        const d = message && message.bruxos_compare && message.bruxos_compare[0];
        if (d) loadVideos(this, d);
      };
    }
  },
});
