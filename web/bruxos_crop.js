import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Preview interativo do Load Image + Crop. Feito como DOM widget para funcionar
// tanto no canvas legado quanto no frontend Nodes 2.0 do ComfyUI.
console.log("[Bruxos] preview/crop de imagem carregado");

const PREVIEW_MAX_H = 480;
const PREVIEW_MIN_H = 240;
const MIN_CROP_PX = 18;
const ASPECTS = {
  "1:1": 1,
  "3:4": 3 / 4,
  "4:3": 4 / 3,
  "16:9": 16 / 9,
  "9:16": 9 / 16,
  "2:3": 2 / 3,
  "3:2": 3 / 2,
};

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function isVueNodes() {
  return !!window.LiteGraph?.vueNodesMode;
}

// No Nodes 1.0 o widget deve ser canvas-only para nao aparecer tambem na aba
// Parameters. No Nodes 2.0 canvasOnly precisa ser false, senao o Vue exclui o
// DOMWidget do corpo do node. O getter permite trocar o renderer sem recriar o
// widget.
function applyRendererVisibility(widget) {
  if (!widget?.options) return;
  Object.defineProperty(widget.options, "canvasOnly", {
    configurable: true,
    enumerable: true,
    get: () => !isVueNodes(),
  });
}

function placePreviewBeforeAdvanced(node, previewWidget, selectorName) {
  if (!isVueNodes() || !Array.isArray(node.widgets)) return;
  const uploadWidgets = node.widgets.filter((item) => {
    const text = `${item?.name || ""} ${item?.label || ""}`.toLowerCase();
    return item?.type === "button" && (text.includes("upload") || text.includes("escolher"));
  });
  const movers = [...uploadWidgets, previewWidget];
  const ordered = node.widgets.filter((item) => !movers.includes(item));
  const selectorIndex = ordered.findIndex((item) => item.name === selectorName);
  ordered.splice(Math.max(0, selectorIndex + 1), 0, ...movers);
  node.widgets.splice(0, node.widgets.length, ...ordered);
  node._widgetSlotsDirty = true;
  const finishLayout = () => {
    if (!node.graph) return;
    node.arrange?.();
    node.setDirtyCanvas?.(true, true);
  };
  if (node.graph) finishLayout();
  else requestAnimationFrame(finishLayout);
}

function value(node, name, fallback) {
  const w = widget(node, name);
  const n = Number(w?.value);
  return Number.isFinite(n) ? n : fallback;
}

function setValue(node, name, next) {
  const w = widget(node, name);
  if (!w) return;
  const rounded = Math.round(next * 1000) / 1000;
  if (Math.abs(Number(w.value) - rounded) < 0.0005) return;
  w.value = rounded;
  w.callback?.(rounded);
}

function inputRef(node) {
  const path = String(widget(node, "image_path")?.value || "").trim();
  // Caminhos absolutos nao podem ser abertos pelo browser. O backend continua
  // aceitando-os; o preview mostra uma explicacao em vez de uma imagem errada.
  if (path) return { absolutePath: path };
  const selected = String(widget(node, "image")?.value || "").replace(/\\/g, "/");
  if (!selected || selected.startsWith("(")) return null;
  const cut = selected.lastIndexOf("/");
  return {
    filename: cut >= 0 ? selected.slice(cut + 1) : selected,
    subfolder: cut >= 0 ? selected.slice(0, cut) : "",
    type: "input",
  };
}

function imageURL(ref) {
  const q = new URLSearchParams({
    filename: ref.filename,
    subfolder: ref.subfolder || "",
    type: ref.type || "input",
    rand: Math.random().toString(36).slice(2),
  });
  return api.apiURL("/view?" + q.toString());
}

function extensionOf(name) {
  const m = String(name || "").match(/\.([^.]+)$/);
  return m ? m[1].toUpperCase() : "—";
}

function actualAspect(node) {
  const preset = String(widget(node, "aspect")?.value || "livre");
  return ASPECTS[preset] || null;
}

function rotationDegrees(node) {
  const selected = String(widget(node, "girar")?.value || "off");
  if (selected.startsWith("-90")) return -90;
  if (selected.startsWith("90")) return 90;
  if (selected.startsWith("180")) return 180;
  return 0;
}

function boolValue(node, name) {
  return widget(node, name)?.value === true;
}

function orientedDimensions(node) {
  const meta = node._bruxosImageMeta;
  if (!meta?.width || !meta?.height) return null;
  return Math.abs(rotationDegrees(node)) === 90
    ? { width: meta.height, height: meta.width }
    : { width: meta.width, height: meta.height };
}

// O backend gira antes de aplicar crop/fit. O preview materializa a mesma
// orientacao em um canvas auxiliar, reutilizado enquanto imagem e giro forem os mesmos.
function orientedSource(node) {
  const preview = node._bruxosImagePreview;
  const image = preview?.image;
  if (!image?.complete || !image.naturalWidth) return null;
  const degrees = rotationDegrees(node);
  const flipH = boolValue(node, "flip_horizontal");
  const flipV = boolValue(node, "flip_vertical");
  const key = `${image.src}|${image.naturalWidth}x${image.naturalHeight}|${degrees}|${flipH}|${flipV}`;
  if (preview.orientedCanvas && preview.orientedKey === key) return preview.orientedCanvas;

  const source = document.createElement("canvas");
  const quarterTurn = Math.abs(degrees) === 90;
  source.width = quarterTurn ? image.naturalHeight : image.naturalWidth;
  source.height = quarterTurn ? image.naturalWidth : image.naturalHeight;
  const ctx = source.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  if (degrees === 90) {
    ctx.translate(source.width, 0);
    ctx.rotate(Math.PI / 2);
  } else if (degrees === -90) {
    ctx.translate(0, source.height);
    ctx.rotate(-Math.PI / 2);
  } else if (degrees === 180) {
    ctx.translate(source.width, source.height);
    ctx.rotate(Math.PI);
  }
  ctx.drawImage(image, 0, 0);

  let result = source;
  if (flipH || flipV) {
    result = document.createElement("canvas");
    result.width = source.width;
    result.height = source.height;
    const flipped = result.getContext("2d");
    flipped.imageSmoothingEnabled = true;
    flipped.imageSmoothingQuality = "high";
    flipped.translate(flipH ? result.width : 0, flipV ? result.height : 0);
    flipped.scale(flipH ? -1 : 1, flipV ? -1 : 1);
    flipped.drawImage(source, 0, 0);
  }
  preview.orientedCanvas = result;
  preview.orientedKey = key;
  return result;
}

function clampBox(box) {
  box.w = Math.max(0.01, Math.min(1, box.w));
  box.h = Math.max(0.01, Math.min(1, box.h));
  box.x = Math.max(0, Math.min(1 - box.w, box.x));
  box.y = Math.max(0, Math.min(1 - box.h, box.y));
  return box;
}

function currentBox(node) {
  return clampBox({
    x: value(node, "crop_x", 0),
    y: value(node, "crop_y", 0),
    w: value(node, "crop_w", 1),
    h: value(node, "crop_h", 1),
  });
}

function commitBox(node, box) {
  clampBox(box);
  setValue(node, "crop_x", box.x);
  setValue(node, "crop_y", box.y);
  setValue(node, "crop_w", box.w);
  setValue(node, "crop_h", box.h);
  render(node);
  node.setDirtyCanvas?.(true, true);
}

function fitPreset(node, keepCenter = true) {
  const ratio = actualAspect(node);
  const dims = orientedDimensions(node);
  if (!ratio || !dims?.width || !dims?.height) {
    render(node);
    return;
  }
  const old = currentBox(node);
  const cx = keepCenter ? old.x + old.w / 2 : 0.5;
  const cy = keepCenter ? old.y + old.h / 2 : 0.5;
  // Normalized width/height is not the pixel aspect. Correct by source W/H.
  const normalizedRatio = ratio * dims.height / dims.width;
  let w = old.w;
  let h = w / normalizedRatio;
  if (h > 1) {
    h = Math.min(1, old.h);
    w = h * normalizedRatio;
  }
  if (w > 1) {
    w = 1;
    h = w / normalizedRatio;
  }
  commitBox(node, { x: cx - w / 2, y: cy - h / 2, w, h });
}

function canvasGeometry(node) {
  const p = node._bruxosImagePreview;
  const dims = orientedDimensions(node);
  if (!p || !dims?.width || !dims?.height) return null;
  const W = p.canvas.width;
  const H = p.canvas.height;
  const scale = Math.min(W / dims.width, H / dims.height);
  const iw = dims.width * scale;
  const ih = dims.height * scale;
  return { x: (W - iw) / 2, y: (H - ih) / 2, w: iw, h: ih };
}

function fittedRect(W, H, aspect, padding = 8) {
  const maxW = Math.max(1, W - padding * 2);
  const maxH = Math.max(1, H - padding * 2);
  let w = maxW;
  let h = w / aspect;
  if (h > maxH) {
    h = maxH;
    w = h * aspect;
  }
  return { x: (W - w) / 2, y: (H - h) / 2, w, h };
}

function outputAspect(node, sourceAspect) {
  const tw = value(node, "target_width", 0);
  const th = value(node, "target_height", 0);
  // _target_from_one no Python preserva o aspecto quando apenas um lado existe.
  return tw > 0 && th > 0 ? tw / th : sourceAspect;
}

function drawCrop(ctx, g, box, active) {
  const x = g.x + box.x * g.w;
  const y = g.y + box.y * g.h;
  const w = box.w * g.w;
  const h = box.h * g.h;

  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,.57)";
  ctx.beginPath();
  ctx.rect(g.x, g.y, g.w, g.h);
  ctx.rect(x, y, w, h);
  ctx.fill("evenodd");

  ctx.strokeStyle = active ? "#a855f7" : "rgba(255,255,255,.72)";
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);

  // Regra dos tercos.
  ctx.strokeStyle = active ? "rgba(255,255,255,.48)" : "rgba(255,255,255,.24)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 2; i++) {
    ctx.beginPath();
    ctx.moveTo(x + w * i / 3, y);
    ctx.lineTo(x + w * i / 3, y + h);
    ctx.moveTo(x, y + h * i / 3);
    ctx.lineTo(x + w, y + h * i / 3);
    ctx.stroke();
  }

  const handles = [
    [x, y], [x + w / 2, y], [x + w, y],
    [x, y + h / 2], [x + w, y + h / 2],
    [x, y + h], [x + w / 2, y + h], [x + w, y + h],
  ];
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#7c3aed";
  for (const [hx, hy] of handles) {
    ctx.beginPath();
    ctx.rect(hx - 4, hy - 4, 8, 8);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function renderInfo(node) {
  const p = node._bruxosImagePreview;
  const m = node._bruxosImageMeta;
  if (!p) return;
  if (!m) {
    p.info.textContent = "";
    return;
  }
  const dims = orientedDimensions(node) || m;
  const b = currentBox(node);
  const cw = Math.max(1, Math.round(dims.width * b.w));
  const ch = Math.max(1, Math.round(dims.height * b.h));
  const tw = value(node, "target_width", 0);
  const th = value(node, "target_height", 0);
  const fit = String(widget(node, "fit_mode")?.value || "off").split(" ")[0];
  const output = tw || th ? ` · saída ${tw || "auto"}×${th || "auto"}` : "";
  p.info.innerHTML =
    `<span>${dims.width} × ${dims.height} px</span>` +
    `<span>${m.format}</span>` +
    `<span>crop ${cw} × ${ch}${output}</span>` +
    `<span>${fit}</span>`;
}

function render(node) {
  const p = node._bruxosImagePreview;
  if (!p) return;
  const ctx = p.canvas.getContext("2d");
  const W = p.canvas.width;
  const H = p.canvas.height;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, W, H);
  const g = canvasGeometry(node);
  const source = orientedSource(node);
  if (g && source) {
    const mode = String(widget(node, "fit_mode")?.value || "off").split(" ")[0];
    const sourceAspect = source.width / source.height;
    if (mode === "stretch") {
      const stage = fittedRect(W, H, outputAspect(node, sourceAspect));
      ctx.drawImage(source, stage.x, stage.y, stage.w, stage.h);
      ctx.strokeStyle = "rgba(168,85,247,.8)";
      ctx.strokeRect(stage.x, stage.y, stage.w, stage.h);
    } else if (mode === "pad") {
      const stage = fittedRect(W, H, outputAspect(node, sourceAspect));
      ctx.fillStyle = "#000";
      ctx.fillRect(stage.x, stage.y, stage.w, stage.h);
      const scale = Math.min(stage.w / source.width, stage.h / source.height);
      const iw = source.width * scale;
      const ih = source.height * scale;
      ctx.drawImage(source, stage.x + (stage.w - iw) / 2, stage.y + (stage.h - ih) / 2, iw, ih);
      ctx.strokeStyle = "rgba(168,85,247,.8)";
      ctx.strokeRect(stage.x, stage.y, stage.w, stage.h);
    } else {
      ctx.drawImage(source, g.x, g.y, g.w, g.h);
      if (mode === "crop") drawCrop(ctx, g, currentBox(node), true);
    }
  } else {
    ctx.fillStyle = "#999";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(p.message || "Escolha uma imagem", W / 2, H / 2);
  }
  renderInfo(node);
}

function hitTest(node, px, py) {
  const g = canvasGeometry(node);
  if (!g) return null;
  const b = currentBox(node);
  const x0 = g.x + b.x * g.w, x1 = x0 + b.w * g.w;
  const y0 = g.y + b.y * g.h, y1 = y0 + b.h * g.h;
  const tol = 10;
  const near = (a, b) => Math.abs(a - b) <= tol;
  const horiz = near(px, x0) ? "w" : near(px, x1) ? "e" : "";
  const vert = near(py, y0) ? "n" : near(py, y1) ? "s" : "";
  if (horiz && py >= y0 - tol && py <= y1 + tol) return vert + horiz;
  if (vert && px >= x0 - tol && px <= x1 + tol) return vert;
  if (px >= x0 && px <= x1 && py >= y0 && py <= y1) return "move";
  return null;
}

function pointerPosition(canvas, event) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - r.left) * canvas.width / r.width,
    y: (event.clientY - r.top) * canvas.height / r.height,
  };
}

function resizeFromDrag(node, drag, dx, dy) {
  const g = canvasGeometry(node);
  if (!g) return;
  const sx = dx / g.w, sy = dy / g.h;
  const start = drag.box;
  let left = start.x, top = start.y;
  let right = start.x + start.w, bottom = start.y + start.h;
  if (drag.mode.includes("w")) left += sx;
  if (drag.mode.includes("e")) right += sx;
  if (drag.mode.includes("n")) top += sy;
  if (drag.mode.includes("s")) bottom += sy;

  const minW = MIN_CROP_PX / g.w, minH = MIN_CROP_PX / g.h;
  left = Math.max(0, Math.min(right - minW, left));
  right = Math.min(1, Math.max(left + minW, right));
  top = Math.max(0, Math.min(bottom - minH, top));
  bottom = Math.min(1, Math.max(top + minH, bottom));

  const ratio = actualAspect(node);
  if (ratio) {
    const dims = orientedDimensions(node) || node._bruxosImageMeta;
    const nr = ratio * dims.height / dims.width;
    const anchorX = drag.mode.includes("w") ? right : left;
    const anchorY = drag.mode.includes("n") ? bottom : top;
    let w = right - left, h = bottom - top;
    // Use the axis with the greatest relative movement as driver.
    if (Math.abs(sx) >= Math.abs(sy)) h = w / nr;
    else w = h * nr;
    w = Math.min(w, drag.mode.includes("w") ? anchorX : 1 - anchorX);
    h = Math.min(h, drag.mode.includes("n") ? anchorY : 1 - anchorY);
    if (h > w / nr) h = w / nr; else w = h * nr;
    left = drag.mode.includes("w") ? anchorX - w : anchorX;
    right = left + w;
    top = drag.mode.includes("n") ? anchorY - h : anchorY;
    bottom = top + h;
  }
  commitBox(node, { x: left, y: top, w: right - left, h: bottom - top });
}

function installPointerEvents(node, canvas) {
  canvas.addEventListener("pointerdown", (event) => {
    const p = pointerPosition(canvas, event);
    const mode = hitTest(node, p.x, p.y);
    if (!mode) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    node._bruxosCropDrag = { mode, start: p, box: currentBox(node) };
  });
  canvas.addEventListener("pointermove", (event) => {
    const p = pointerPosition(canvas, event);
    const drag = node._bruxosCropDrag;
    if (!drag) {
      const mode = hitTest(node, p.x, p.y);
      canvas.style.cursor = mode === "move" ? "move" :
        mode?.length === 2 ? `${mode}-resize` :
        mode ? `${mode}-resize` : "default";
      return;
    }
    event.preventDefault();
    const g = canvasGeometry(node);
    if (drag.mode === "move") {
      commitBox(node, {
        x: drag.box.x + (p.x - drag.start.x) / g.w,
        y: drag.box.y + (p.y - drag.start.y) / g.h,
        w: drag.box.w,
        h: drag.box.h,
      });
    } else {
      resizeFromDrag(node, drag, p.x - drag.start.x, p.y - drag.start.y);
    }
  });
  const finish = () => { node._bruxosCropDrag = null; };
  canvas.addEventListener("pointerup", finish);
  canvas.addEventListener("pointercancel", finish);
}

function resizeCanvas(node) {
  const p = node._bruxosImagePreview;
  if (!p) return;
  // O frontend ja transforma DOMWidgets junto com o graph. Use somente as
  // dimensoes CSS realmente entregues pelo layout; aplicar ds.scale aqui
  // compensava o zoom duas vezes e descolava o preview da moldura do node.
  const cssWidth = Math.max(1, p.wrap.clientWidth || (node.size?.[0] || 300) - 28);
  const dims = orientedDimensions(node);
  const aspect = dims ? dims.width / dims.height : 16 / 9;
  const cssHeight = Math.round(Math.min(
    PREVIEW_MAX_H,
    Math.max(PREVIEW_MIN_H, cssWidth / aspect)
  ));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  p.canvas.style.width = "100%";
  p.canvas.style.height = cssHeight + "px";
  const nextW = Math.round(cssWidth * dpr), nextH = Math.round(cssHeight * dpr);
  if (p.canvas.width !== nextW || p.canvas.height !== nextH) {
    p.canvas.width = nextW;
    p.canvas.height = nextH;
  }
  render(node);
}

function previewHeight(node, width) {
  const contentWidth = Math.max(180, (Number(width) || node.size?.[0] || 300) - 28);
  const dims = orientedDimensions(node);
  const aspect = dims ? dims.width / dims.height : 16 / 9;
  return Math.min(PREVIEW_MAX_H, Math.max(PREVIEW_MIN_H, contentWidth / aspect)) +
    31;
}

function resizeNodeToContent(node) {
  requestAnimationFrame(() => {
    const computed = node.computeSize?.();
    if (!computed) return;
    const width = Math.max(Number(node.size?.[0]) || 0, Number(computed[0]) || 0);
    const height = Math.max(120, Number(computed[1]) || 0);
    node.setSize?.([width, height]);
    node.setDirtyCanvas?.(true, true);
  });
}

function loadSelectedImage(node) {
  const p = ensurePreview(node);
  const ref = inputRef(node);
  node._bruxosImageMeta = null;
  if (!ref) {
    p.message = "Escolha uma imagem";
    p.image.removeAttribute("src");
    render(node);
    return;
  }
  if (ref.absolutePath) {
    p.message = "Preview indisponível para image_path absoluto";
    p.image.removeAttribute("src");
    render(node);
    return;
  }
  p.message = "Carregando…";
  p.image.onload = () => {
    node._bruxosImageMeta = {
      width: p.image.naturalWidth,
      height: p.image.naturalHeight,
      format: extensionOf(ref.filename),
      filename: ref.filename,
    };
    fitPreset(node);
    resizeCanvas(node);
    resizeNodeToContent(node);
  };
  p.image.onerror = () => {
    node._bruxosImageMeta = null;
    p.message = "Não foi possível abrir a imagem";
    render(node);
  };
  p.image.src = imageURL(ref);
}

function ensurePreview(node) {
  if (node._bruxosImagePreview) return node._bruxosImagePreview;
  const wrap = document.createElement("div");
  wrap.style.cssText =
    "width:100%;max-width:100%;min-width:0;box-sizing:border-box;" +
    "padding:0 2px;overflow:hidden;contain:layout;";
  const canvas = document.createElement("canvas");
  canvas.style.cssText =
    "display:block;width:100%;max-width:100%;min-width:0;background:#111;border:1px solid #3b3b44;" +
    "border-radius:7px;box-sizing:border-box;touch-action:none;";
  const info = document.createElement("div");
  info.style.cssText =
    "display:flex;flex-wrap:wrap;justify-content:space-between;gap:2px 10px;" +
    "padding:5px 2px 2px;color:#bdbdc7;font:10px/1.35 ui-monospace,monospace;";
  const image = new Image();
  image.decoding = "async";
  wrap.append(canvas, info);

  const measureHeight = () => previewHeight(node, node.size?.[0]);
  const domWidget = node.addDOMWidget("bruxos_image_crop_preview", "preview", wrap, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: measureHeight,
    getMaxHeight: measureHeight,
    margin: 0,
  });
  domWidget.serialize = false;
  domWidget.serializeValue = () => undefined;
  // IMPORTANTE: mantenha este widget DEPOIS dos widgets de entrada. Versoes
  // antigas/hibritas do frontend ignoram serialize:false ao restaurar
  // widgets_values; inserir o preview antes deles desloca todos os valores.
  domWidget.computeSize = (width) => [width, previewHeight(node, width)];
  if (isVueNodes()) {
    domWidget.computeLayoutSize = () => ({
      minWidth: 1,
      minHeight: measureHeight(),
      maxHeight: measureHeight(),
    });
  }
  domWidget.getHeight = measureHeight;
  applyRendererVisibility(domWidget);
  node._bruxosImagePreview = { wrap, canvas, info, image, widget: domWidget, message: "" };
  placePreviewBeforeAdvanced(node, domWidget, "image");
  installPointerEvents(node, canvas);
  if (typeof ResizeObserver !== "undefined") {
    node._bruxosImagePreview.resizeObserver = new ResizeObserver(() => resizeCanvas(node));
    node._bruxosImagePreview.resizeObserver.observe(wrap);
  }
  resizeCanvas(node);
  return node._bruxosImagePreview;
}

function hookNode(node) {
  ensurePreview(node);
  const originalResize = node.onResize;
  node.onResize = function () {
    const result = originalResize?.apply(this, arguments);
    requestAnimationFrame(() => resizeCanvas(this));
    return result;
  };
  const refreshNames = [
    "crop_x", "crop_y", "crop_w", "crop_h",
    "target_width", "target_height", "fit_mode",
  ];
  for (const name of refreshNames) {
    const w = widget(node, name);
    if (!w) continue;
    const original = w.callback;
    w.callback = function () {
      const result = original?.apply(this, arguments);
      requestAnimationFrame(() => render(node));
      return result;
    };
  }
  const aspectWidget = widget(node, "aspect");
  if (aspectWidget) {
    const original = aspectWidget.callback;
    aspectWidget.callback = function () {
      const result = original?.apply(this, arguments);
      // Nodes 2.0 atualiza o valor reativo logo depois do callback. Aguarde um
      // frame para calcular com a proporcao nova, nao com a anterior.
      requestAnimationFrame(() => fitPreset(node));
      return result;
    };
  }
  const rotateWidget = widget(node, "girar");
  if (rotateWidget) {
    const original = rotateWidget.callback;
    rotateWidget.callback = function () {
      const result = original?.apply(this, arguments);
      requestAnimationFrame(() => {
        const preview = node._bruxosImagePreview;
        if (preview) {
          preview.orientedCanvas = null;
          preview.orientedKey = null;
        }
        // Um quarto de volta troca largura e altura; atualize canvas e box.
        fitPreset(node);
        resizeCanvas(node);
        resizeNodeToContent(node);
      });
      return result;
    };
  }
  for (const name of ["flip_horizontal", "flip_vertical"]) {
    const flipWidget = widget(node, name);
    if (!flipWidget) continue;
    const original = flipWidget.callback;
    flipWidget.callback = function () {
      const result = original?.apply(this, arguments);
      requestAnimationFrame(() => {
        const preview = node._bruxosImagePreview;
        if (preview) {
          preview.orientedCanvas = null;
          preview.orientedKey = null;
        }
        render(node);
        node.setDirtyCanvas?.(true, true);
      });
      return result;
    };
  }
  for (const name of ["image", "image_path"]) {
    const w = widget(node, name);
    if (!w) continue;
    const original = w.callback;
    w.callback = function () {
      const result = original?.apply(this, arguments);
      loadSelectedImage(node);
      return result;
    };
  }

  loadSelectedImage(node);
}

app.registerExtension({
  name: "BruxosDoVFX.LoadImageCropPreview.Node2",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "BruxosLoadImage") return;
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      hookNode(this);
      return result;
    };
    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigure?.apply(this, arguments);
      setTimeout(() => {
        loadSelectedImage(this);
      }, 0);
      setTimeout(() => {
        loadSelectedImage(this);
      }, 80);
      return result;
    };
    const originalRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this._bruxosImagePreview?.resizeObserver?.disconnect();
      this._bruxosImagePreview?.image?.removeAttribute("src");
      return originalRemoved?.apply(this, arguments);
    };
  },
});
