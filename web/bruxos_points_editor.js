import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Bruxos do VFX - Editor de Pontos SAM3
// Clique ESQUERDO = ponto verde (selecionar) | clique DIREITO = ponto roxo (negar)
// Shift + arrastar = caixa (bbox)

const COR_VERDE = "#22c55e";   // selecionar
const COR_ROXO = "#a855f7";    // negar
const COR_BBOX = "#22c55e";

function getRealURL(ref) {
  return api.apiURL(
    `/view?filename=${encodeURIComponent(ref.filename)}&type=${ref.type}&subfolder=${ref.subfolder || ""}&rand=${Math.random()}`
  );
}

app.registerExtension({
  name: "BruxosDoVFX.PointsEditor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "BruxosPointsEditor") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      buildEditor(this);
      return r;
    };

    const onExec = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      if (onExec) onExec.apply(this, arguments);
      try {
        const p = message && message.preview && message.preview[0];
        if (p && p.preview_str) {
          const refs = JSON.parse(p.preview_str);
          this._bruxosPtsFrames = Array.isArray(refs) ? refs.map(getRealURL) : [];
          this._bruxosPtsLoadFrame(0);
        }
      } catch (e) { console.warn("[Bruxos] editor de pontos: preview parse", e); }
    };
  },
});

function buildEditor(node) {
  // esconde o widget "info" (STRING) usado só pra guardar o JSON serializado
  const infoWidget = node.widgets && node.widgets.find((w) => w.name === "info");
  if (infoWidget) {
    infoWidget.computeSize = () => [0, -4];
    infoWidget.type = "hidden_info";
  }

  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%;background:transparent;";

  // barra de ferramentas
  const toolbar = document.createElement("div");
  toolbar.style.cssText = "display:flex;align-items:center;gap:6px;font-size:11px;color:#ccc;flex-wrap:wrap;";
  toolbar.innerHTML = `
    <span style="display:flex;align-items:center;gap:3px;">
      <span style="width:9px;height:9px;border-radius:50%;background:${COR_VERDE};display:inline-block;"></span>
      esq = selecionar
    </span>
    <span style="display:flex;align-items:center;gap:3px;">
      <span style="width:9px;height:9px;border-radius:50%;background:${COR_ROXO};display:inline-block;"></span>
      dir = negar
    </span>
    <span style="opacity:.7;">shift+arrastar = caixa</span>
    <span style="flex:1;"></span>
    <button data-act="undo" style="background:#2a2a2a;color:#ccc;border:1px solid #444;border-radius:4px;padding:2px 6px;cursor:pointer;">↶</button>
    <button data-act="redo" style="background:#2a2a2a;color:#ccc;border:1px solid #444;border-radius:4px;padding:2px 6px;cursor:pointer;">↷</button>
    <button data-act="reset" style="background:#2a2a2a;color:#ccc;border:1px solid #444;border-radius:4px;padding:2px 6px;cursor:pointer;">limpar</button>
  `;

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "width:100%;display:block;border-radius:4px;background:#111;cursor:crosshair;";
  canvas.width = 400;
  canvas.height = 300;

  const frameRow = document.createElement("div");
  frameRow.style.cssText = "display:flex;align-items:center;gap:6px;font-size:11px;color:#ccc;";
  const frameSlider = document.createElement("input");
  frameSlider.type = "range";
  frameSlider.min = "0";
  frameSlider.max = "0";
  frameSlider.value = "0";
  frameSlider.style.cssText = "flex:1;";
  const frameLabel = document.createElement("span");
  frameLabel.textContent = "frame 0";
  frameLabel.style.cssText = "min-width:60px;text-align:right;";
  frameRow.appendChild(frameSlider);
  frameRow.appendChild(frameLabel);

  wrap.appendChild(toolbar);
  wrap.appendChild(canvas);
  wrap.appendChild(frameRow);

  node.addDOMWidget("bruxos_points_editor", "div", wrap, { serialize: false });

  // ---- estado ----
  node._bruxosPtsFrames = [];
  node._bruxosPtsFrameIndex = 0;
  node._bruxosPtsImg = null;
  node._bruxosPtsPositive = [];
  node._bruxosPtsNegative = [];
  node._bruxosPtsBoxes = [];
  node._bruxosPtsHistory = [];
  node._bruxosPtsHistIdx = -1;
  node._bruxosPtsDrag = null; // {x0,y0} durante shift+arrastar

  const ctx = canvas.getContext("2d");

  function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (node._bruxosPtsImg) {
      ctx.drawImage(node._bruxosPtsImg, 0, 0, canvas.width, canvas.height);
    }
    const sx = node._bruxosPtsImg ? canvas.width / node._bruxosPtsImg.naturalWidth : 1;
    const sy = node._bruxosPtsImg ? canvas.height / node._bruxosPtsImg.naturalHeight : 1;

    ctx.lineWidth = 2;
    ctx.strokeStyle = COR_BBOX;
    for (const b of node._bruxosPtsBoxes) {
      ctx.strokeRect(b.x * sx, b.y * sy, b.w * sx, b.h * sy);
    }
    for (const p of node._bruxosPtsPositive) {
      drawDot(p.x * sx, p.y * sy, COR_VERDE);
    }
    for (const p of node._bruxosPtsNegative) {
      drawDot(p.x * sx, p.y * sy, COR_ROXO);
    }
  }

  function drawDot(x, y, color) {
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  }

  function toImageCoords(evt) {
    const rect = canvas.getBoundingClientRect();
    const cx = (evt.clientX - rect.left) * (canvas.width / rect.width);
    const cy = (evt.clientY - rect.top) * (canvas.height / rect.height);
    if (!node._bruxosPtsImg) return { x: cx, y: cy };
    const sx = node._bruxosPtsImg.naturalWidth / canvas.width;
    const sy = node._bruxosPtsImg.naturalHeight / canvas.height;
    return { x: cx * sx, y: cy * sy };
  }

  function pushHistory() {
    const state = {
      positive: JSON.parse(JSON.stringify(node._bruxosPtsPositive)),
      negative: JSON.parse(JSON.stringify(node._bruxosPtsNegative)),
      boxes: JSON.parse(JSON.stringify(node._bruxosPtsBoxes)),
    };
    node._bruxosPtsHistory = node._bruxosPtsHistory.slice(0, node._bruxosPtsHistIdx + 1);
    node._bruxosPtsHistory.push(state);
    node._bruxosPtsHistIdx = node._bruxosPtsHistory.length - 1;
  }

  function restore(state) {
    node._bruxosPtsPositive = JSON.parse(JSON.stringify(state.positive));
    node._bruxosPtsNegative = JSON.parse(JSON.stringify(state.negative));
    node._bruxosPtsBoxes = JSON.parse(JSON.stringify(state.boxes));
    redraw();
    saveInfo();
  }

  function saveInfo() {
    const w = node.widgets && node.widgets.find((x) => x.name === "info");
    if (!w) return;
    w.value = JSON.stringify({
      positive_coords: node._bruxosPtsPositive,
      negative_coords: node._bruxosPtsNegative,
      bbox: node._bruxosPtsBoxes,
      frame_index: node._bruxosPtsFrameIndex,
    });
  }

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("mousedown", (e) => {
    if (e.shiftKey) {
      const p = toImageCoords(e);
      node._bruxosPtsDrag = { x0: p.x, y0: p.y };
      return;
    }
    const p = toImageCoords(e);
    if (e.button === 2) {
      node._bruxosPtsNegative.push(p);
    } else {
      node._bruxosPtsPositive.push(p);
    }
    redraw();
    pushHistory();
    saveInfo();
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!node._bruxosPtsDrag) return;
    const p = toImageCoords(e);
    redraw();
    const sx = node._bruxosPtsImg ? canvas.width / node._bruxosPtsImg.naturalWidth : 1;
    const sy = node._bruxosPtsImg ? canvas.height / node._bruxosPtsImg.naturalHeight : 1;
    const { x0, y0 } = node._bruxosPtsDrag;
    ctx.strokeStyle = COR_BBOX;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(Math.min(x0, p.x) * sx, Math.min(y0, p.y) * sy, Math.abs(p.x - x0) * sx, Math.abs(p.y - y0) * sy);
    ctx.setLineDash([]);
  });

  window.addEventListener("mouseup", (e) => {
    if (!node._bruxosPtsDrag) return;
    const p = toImageCoords(e);
    const { x0, y0 } = node._bruxosPtsDrag;
    node._bruxosPtsDrag = null;
    const w = Math.abs(p.x - x0);
    const h = Math.abs(p.y - y0);
    if (w > 3 && h > 3) {
      node._bruxosPtsBoxes.push({ x: Math.min(x0, p.x), y: Math.min(y0, p.y), w, h });
      pushHistory();
      saveInfo();
    }
    redraw();
  });

  toolbar.addEventListener("click", (e) => {
    const act = e.target && e.target.dataset && e.target.dataset.act;
    if (!act) return;
    if (act === "undo") {
      if (node._bruxosPtsHistIdx >= 0) {
        node._bruxosPtsHistIdx--;
        const state = node._bruxosPtsHistIdx >= 0
          ? node._bruxosPtsHistory[node._bruxosPtsHistIdx]
          : { positive: [], negative: [], boxes: [] };
        restore(state);
      }
    } else if (act === "redo") {
      if (node._bruxosPtsHistIdx < node._bruxosPtsHistory.length - 1) {
        node._bruxosPtsHistIdx++;
        restore(node._bruxosPtsHistory[node._bruxosPtsHistIdx]);
      }
    } else if (act === "reset") {
      node._bruxosPtsPositive = [];
      node._bruxosPtsNegative = [];
      node._bruxosPtsBoxes = [];
      pushHistory();
      redraw();
      saveInfo();
    }
  });

  frameSlider.addEventListener("input", () => {
    node._bruxosPtsLoadFrame(parseInt(frameSlider.value, 10) || 0);
  });

  node._bruxosPtsLoadFrame = function (idx) {
    const frames = node._bruxosPtsFrames || [];
    if (!frames.length) return;
    idx = Math.max(0, Math.min(idx, frames.length - 1));
    node._bruxosPtsFrameIndex = idx;
    frameSlider.max = String(frames.length - 1);
    frameSlider.value = String(idx);
    frameLabel.textContent = `frame ${idx} / ${frames.length - 1}`;
    const img = new Image();
    img.onload = () => {
      node._bruxosPtsImg = img;
      redraw();
    };
    img.src = frames[idx];
    saveInfo();
  };

  redraw();
}
