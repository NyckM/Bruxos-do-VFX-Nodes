import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Preview de video (DOM widget, Nodes 2.0) para Load Video (Bruxos) e
// Save Video (Bruxos), + infos do video no Load.
console.log("[Bruxos] preview de video carregado");

const PREVIEW_MAX_SIDE = 720; // arquivo leve, mas nitido para rosto/crop
const MAX_H = 500;   // preview principal grande, semelhante ao Load Image nativo
const INFO_H = 154;  // transporte + origem + pedido + saida real + frames/fps/audio

function isVueNodes() {
  return !!window.LiteGraph?.vueNodesMode;
}

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
  const movers = [...uploadWidgets, node._bruxosVideoGalleryButton, previewWidget].filter(Boolean);
  const ordered = node.widgets.filter((item) => !movers.includes(item));
  const selectorIndex = ordered.findIndex((item) => item.name === selectorName);
  ordered.splice(Math.max(0, selectorIndex + 1), 0, ...movers);
  node.widgets.splice(0, node.widgets.length, ...ordered);
  node._widgetSlotsDirty = true;
  const finishLayout = () => {
    // onNodeCreated tambem roda durante configure(), antes de LGraph.add().
    // setDirtyCanvas exige node.graph e lanca NullGraphError sem esta guarda.
    if (!node.graph) return;
    node.arrange?.();
    node.setDirtyCanvas?.(true, true);
  };
  if (node.graph) finishLayout();
  else requestAnimationFrame(finishLayout);
}

// Transporte compartilhado entre TODOS os previews criados por este arquivo.
// Isso inclui Load Video e Save Video: um pode ser o mestre e os demais seguem.
const TRANSPORT = globalThis.__bruxosVideoTransport || {
  players: new Set(), leader: null, applying: false, raf: 0,
};
globalThis.__bruxosVideoTransport = TRANSPORT;

function fmtTime(seconds) {
  const s = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${String(m).padStart(2, "0")}:${rest.toFixed(2).padStart(5, "0")}`;
}

function transportButton(text, title) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = text;
  b.title = title;
  b.style.cssText =
    "height:25px;padding:2px 8px;border-radius:6px;border:1px solid #4a4a55;" +
    "background:#17171d;color:#ddd;font:600 10px sans-serif;cursor:pointer;white-space:nowrap;";
  return b;
}

function updateTransportUI(p) {
  if (!p || !p.video) return;
  const v = p.video;
  p.play.textContent = v.paused ? "▶ Play" : "⏸ Pause";
  p.sync.textContent = p.syncEnabled ? "🔗 Sync" : "⛓ Sync off";
  p.sync.style.borderColor = p.syncEnabled ? "#22c55e" : "#4a4a55";
  p.sync.style.color = p.syncEnabled ? "#4ade80" : "#999";
  if (!p.draggingSeek) {
    const d = Number.isFinite(v.duration) ? v.duration : 0;
    p.seek.max = String(Math.max(0.001, d));
    p.seek.value = String(Math.min(d || 0, v.currentTime || 0));
  }
  p.clock.textContent = `${fmtTime(v.currentTime)} / ${fmtTime(v.duration)}`;
}

function members(origin) {
  if (!origin.syncEnabled) return [origin];
  return [...TRANSPORT.players].filter((p) => p.syncEnabled && p.video?.isConnected);
}

function setGroupTime(origin, seconds) {
  TRANSPORT.leader = origin;
  TRANSPORT.applying = true;
  for (const p of members(origin)) {
    const d = Number.isFinite(p.video.duration) ? p.video.duration : 0;
    if (d > 0) p.video.currentTime = Math.max(0, Math.min(seconds, Math.max(0, d - 0.001)));
    updateTransportUI(p);
  }
  TRANSPORT.applying = false;
}

function playGroup(origin) {
  TRANSPORT.leader = origin;
  origin.video.playbackRate = 1;
  const t = origin.video.currentTime || 0;
  TRANSPORT.applying = true;
  for (const p of members(origin)) {
    const d = Number.isFinite(p.video.duration) ? p.video.duration : 0;
    if (d > 0) p.video.currentTime = Math.max(0, Math.min(t, Math.max(0, d - 0.001)));
    p.video.playbackRate = origin.video.playbackRate || 1;
    p.video.play().catch(() => {});
    updateTransportUI(p);
  }
  TRANSPORT.applying = false;
  ensureTransportLoop();
}

function pauseGroup(origin) {
  TRANSPORT.leader = origin;
  // Cancela o proximo tick antes de pausar. Sem isto, um frame de sincronismo
  // ja agendado ainda pode chamar play() novamente em um dos previews.
  if (TRANSPORT.raf) cancelAnimationFrame(TRANSPORT.raf);
  TRANSPORT.raf = 0;
  TRANSPORT.applying = true;
  for (const p of members(origin)) {
    p.video.pause();
    updateTransportUI(p);
  }
  TRANSPORT.applying = false;
}

function stopGroup(origin) {
  TRANSPORT.leader = origin;
  // O mesmo cuidado do Pause e essencial no Stop, que tambem rebobina.
  if (TRANSPORT.raf) cancelAnimationFrame(TRANSPORT.raf);
  TRANSPORT.raf = 0;
  TRANSPORT.applying = true;
  for (const p of members(origin)) {
    p.video.pause();
    try { p.video.currentTime = 0; } catch (e) {}
    updateTransportUI(p);
  }
  TRANSPORT.applying = false;
}

function ensureTransportLoop() {
  if (TRANSPORT.raf) return;
  const tick = () => {
    const lead = TRANSPORT.leader;
    if (!lead || !lead.video?.isConnected || lead.video.paused || !lead.syncEnabled) {
      for (const p of TRANSPORT.players) updateTransportUI(p);
      TRANSPORT.raf = 0;
      return;
    }
    TRANSPORT.raf = requestAnimationFrame(tick);
    const lt = lead.video.currentTime || 0;
    for (const p of members(lead)) {
      if (p === lead) { updateTransportUI(p); continue; }
      const d = Number.isFinite(p.video.duration) ? p.video.duration : 0;
      if (!d) continue;
      const target = Math.max(0, Math.min(lt, Math.max(0, d - 0.001)));
      const drift = target - (p.video.currentTime || 0);
      // Corrige saltando so quando a diferenca e visivel. Para drift pequeno,
      // uma variacao minima de velocidade evita os videos ficarem tremendo.
      if (Math.abs(drift) > 0.12) p.video.currentTime = target;
      else p.video.playbackRate = Math.max(0.96, Math.min(1.04, 1 + drift * 0.18));
      if (p.video.paused) p.video.play().catch(() => {});
      updateTransportUI(p);
    }
    updateTransportUI(lead);
  };
  TRANSPORT.raf = requestAnimationFrame(tick);
}

function registerTransport(p) {
  TRANSPORT.players.add(p);
  p.play.onclick = () => p.video.paused ? playGroup(p) : pauseGroup(p);
  p.stop.onclick = () => stopGroup(p);
  p.sync.onclick = () => {
    p.syncEnabled = !p.syncEnabled;
    if (p.syncEnabled && TRANSPORT.leader && TRANSPORT.leader !== p) {
      setGroupTime(TRANSPORT.leader, TRANSPORT.leader.video.currentTime || 0);
      if (!TRANSPORT.leader.video.paused) playGroup(TRANSPORT.leader);
    }
    updateTransportUI(p);
  };
  p.seek.addEventListener("pointerdown", () => { p.draggingSeek = true; });
  p.seek.addEventListener("input", () => {
    setGroupTime(p, Number(p.seek.value) || 0);
    updateTransportUI(p);
  });
  const endSeek = () => { p.draggingSeek = false; updateTransportUI(p); };
  p.seek.addEventListener("change", endSeek);
  p.seek.addEventListener("pointerup", endSeek);
  for (const ev of ["play", "pause", "timeupdate", "ratechange", "ended"])
    p.video.addEventListener(ev, () => {
      updateTransportUI(p);
      if (ev === "play" && p === TRANSPORT.leader) ensureTransportLoop();
    });
  updateTransportUI(p);
}

function viewURL(ref, folderType) {
  const sub = ref.subfolder ? encodeURIComponent(ref.subfolder) : "";
  const type = ref.type || folderType || "input";
  return api.apiURL(
    `/view?filename=${encodeURIComponent(ref.filename)}` +
    `&type=${type}&subfolder=${sub}&rand=${Math.random().toString(36).slice(2)}`
  );
}

function videoChoices(node) {
  const selector = node.widgets?.find((item) => item.name === "video");
  let values = selector?.options?.values;
  if (typeof values === "function") {
    try { values = values(); } catch (_) { values = []; }
  }
  if (!Array.isArray(values)) values = selector?.value ? [selector.value] : [];
  const videoExt = /\.(mp4|mov|mkv|avi|webm|gif|m4v|mpg|mpeg|wmv|flv)$/i;
  return [...new Set(values.map((item) => String(item || "")).filter((item) => videoExt.test(item)))];
}

function thumbnailURL(filename) {
  const clean = String(filename).replace(/\s+\[(input|output|temp)\]\s*$/i, "");
  return api.apiURL(`/bruxos/video_thumbnail?filename=${encodeURIComponent(clean)}`);
}

function openVideoGallery(node) {
  document.querySelector(".bruxos-video-gallery-overlay")?.remove();
  const choices = videoChoices(node);
  const overlay = document.createElement("div");
  overlay.className = "bruxos-video-gallery-overlay";
  overlay.style.cssText =
    "position:fixed;inset:0;z-index:100000;background:rgba(0,0,0,.72);" +
    "display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box;";
  const panel = document.createElement("div");
  panel.style.cssText =
    "width:min(980px,96vw);max-height:88vh;display:flex;flex-direction:column;gap:10px;" +
    "background:#202126;border:1px solid #51535d;border-radius:12px;padding:14px;" +
    "box-shadow:0 18px 60px rgba(0,0,0,.55);box-sizing:border-box;";
  const top = document.createElement("div");
  top.style.cssText = "display:flex;align-items:center;gap:10px;";
  const title = document.createElement("strong");
  title.textContent = "Escolher video pelo primeiro frame";
  title.style.cssText = "color:#eee;font:600 14px sans-serif;white-space:nowrap;";
  const search = document.createElement("input");
  search.placeholder = "Filtrar videos...";
  search.style.cssText =
    "flex:1;min-width:80px;height:32px;border:1px solid #4b4d57;border-radius:7px;" +
    "background:#15161a;color:#eee;padding:0 10px;box-sizing:border-box;";
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "Fechar";
  close.style.cssText =
    "height:32px;padding:0 12px;border:1px solid #555864;border-radius:7px;" +
    "background:#30323a;color:#eee;cursor:pointer;";
  const grid = document.createElement("div");
  grid.style.cssText =
    "display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;" +
    "overflow:auto;min-height:180px;padding:2px;";
  top.append(title, search, close);
  panel.append(top, grid);
  overlay.append(panel);

  const dismiss = () => overlay.remove();
  close.onclick = dismiss;
  overlay.addEventListener("pointerdown", (event) => {
    event.stopPropagation();
    if (event.target === overlay) dismiss();
  });
  panel.addEventListener("pointerdown", (event) => event.stopPropagation());

  const cards = [];
  for (const filename of choices) {
    const card = document.createElement("button");
    card.type = "button";
    card.title = filename;
    card.dataset.search = filename.toLowerCase();
    card.style.cssText =
      "display:flex;flex-direction:column;gap:6px;min-width:0;padding:6px;" +
      "border:1px solid #444650;border-radius:8px;background:#15161a;color:#ddd;" +
      "cursor:pointer;text-align:left;";
    const image = document.createElement("img");
    image.loading = "lazy";
    image.alt = "";
    image.src = thumbnailURL(filename);
    image.style.cssText =
      "display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#080808;border-radius:5px;";
    image.onerror = () => {
      image.removeAttribute("src");
      image.alt = "Preview indisponivel";
    };
    const label = document.createElement("span");
    label.textContent = filename;
    label.style.cssText =
      "display:block;width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:11px sans-serif;";
    card.append(image, label);
    card.onclick = () => {
      const selector = node.widgets?.find((item) => item.name === "video");
      if (selector) {
        selector.value = filename;
        selector.callback?.(filename);
        node.setDirtyCanvas?.(true, true);
      }
      dismiss();
    };
    cards.push(card);
    grid.append(card);
  }
  if (!choices.length) {
    grid.textContent = "Nenhum video encontrado no input do ComfyUI.";
    grid.style.color = "#aaa";
  }
  search.oninput = () => {
    const query = search.value.trim().toLowerCase();
    for (const card of cards) card.style.display = card.dataset.search.includes(query) ? "flex" : "none";
  };
  document.body.append(overlay);
  search.focus();
}

function ensurePreview(node) {
  if (node._bruxosPrev) return node._bruxosPrev;

  const wrap = document.createElement("div");
  wrap.style.cssText =
    "width:100%;max-width:100%;box-sizing:border-box;display:block;" +
    "overflow:hidden;contain:layout;padding:0 2px;";

  const video = document.createElement("video");
  video.muted = true;
  video.loop = true;
  video.autoplay = true;
  video.playsInline = true;
  video.controls = false;
  // min-width:0 / max-width:100% evitam que o video vaze do node
  video.style.cssText =
    "display:block;width:100%;max-width:100%;min-width:0;height:auto;" +
    "max-height:" + MAX_H + "px;object-fit:contain;background:#000;" +
    "border-radius:6px;";

  const controls = document.createElement("div");
  controls.style.cssText =
    "display:flex;align-items:center;gap:5px;width:100%;box-sizing:border-box;" +
    "padding:5px 1px 1px;min-width:0;";
  const play = transportButton("▶ Play", "Play/Pause. Com Sync ligado, controla Load e Save juntos.");
  const stop = transportButton("■ Stop", "Para e volta todos os players sincronizados para 00:00.");
  const sync = transportButton("🔗 Sync", "Liga/desliga este preview no transporte compartilhado.");
  const fullscreen = transportButton("\u26f6 Tela", "Abre a referencia em tela cheia (duplo clique no video tambem abre).");
  const seek = document.createElement("input");
  seek.type = "range"; seek.min = "0"; seek.max = "1"; seek.step = "0.01"; seek.value = "0";
  seek.title = "Arraste para buscar; os players com Sync seguem juntos.";
  seek.style.cssText = "flex:1 1 80px;min-width:45px;accent-color:#22c55e;";
  const clock = document.createElement("span");
  clock.style.cssText = "color:#aaa;font:10px monospace;white-space:nowrap;";
  controls.append(play, stop, sync, fullscreen, seek, clock);

  const info = document.createElement("div");
  info.style.cssText =
    "width:100%;box-sizing:border-box;margin-top:4px;font-size:10px;" +
    "line-height:1.35;color:#bbb;font-family:monospace;white-space:pre-wrap;" +
    "word-break:break-word;text-align:left;";

  wrap.append(video, controls, info);

  const measureHeight = (width = node.size?.[0]) => {
    let height = INFO_H;
    if (node._bruxosPrev?.video?.style?.display !== "none") {
      const aspect = node._bruxosMeta?.aspect;
      const contentWidth = Math.max(80, (Number(width) || 200) - 24);
      height += aspect
        ? Math.min(MAX_H, Math.max(60, contentWidth / aspect)) + 8
        : 160;
    }
    return height;
  };

  if (!node._bruxosVideoGalleryButton) {
    const gallery = node.addWidget(
      "button",
      "Escolher video por miniatura",
      null,
      () => openVideoGallery(node),
      { serialize: false },
    );
    gallery.serialize = false;
    node._bruxosVideoGalleryButton = gallery;
  }

  const currentHeight = () => measureHeight(node.size?.[0]);
  const widget = node.addDOMWidget("bruxos_preview", "preview", wrap, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: currentHeight,
    getMaxHeight: currentHeight,
    margin: 0,
  });
  // ComfyUI 0.28 ignora options.serialize -> forca por propriedade + serializeValue
  // pra este widget NAO entrar em widgets_values (senao desloca os valores).
  try {
    widget.serialize = false;
    widget.serializeValue = () => undefined;
  } catch (e) {}

  widget.computeSize = (width) => [width, measureHeight(width)];
  if (isVueNodes()) {
    widget.computeLayoutSize = () => ({
      minWidth: 1,
      minHeight: currentHeight(),
      maxHeight: currentHeight(),
    });
  }
  widget.getHeight = currentHeight;
  applyRendererVisibility(widget);

  const preview = {
    wrap, video, controls, play, stop, sync, fullscreen, seek, clock, info, widget,
    syncEnabled: true, draggingSeek: false,
  };
  node._bruxosPrev = preview;
  placePreviewBeforeAdvanced(node, widget, "video");
  registerTransport(preview);

  const openFullscreen = () => {
    const fn = video.requestFullscreen || video.webkitRequestFullscreen;
    if (fn) {
      try { fn.call(video); } catch (e) {}
    }
  };
  fullscreen.onclick = (e) => { e.preventDefault(); e.stopPropagation(); openFullscreen(); };
  video.addEventListener("dblclick", (e) => {
    e.preventDefault(); e.stopPropagation(); openFullscreen();
  });

  video.addEventListener("loadedmetadata", () => {
    node._bruxosMeta = {
      w: video.videoWidth,
      h: video.videoHeight,
      dur: video.duration,
      aspect: video.videoWidth && video.videoHeight
        ? video.videoWidth / video.videoHeight : null,
    };
    renderInfo(node);
    if (preview.syncEnabled && TRANSPORT.leader && TRANSPORT.leader !== preview)
      setGroupTime(TRANSPORT.leader, TRANSPORT.leader.video.currentTime || 0);
    else if (!TRANSPORT.leader) TRANSPORT.leader = preview;
    updateTransportUI(preview);
    resizeNodeToContent(node);
  });
  video.addEventListener("error", () => {
    node._bruxosPrev.info.textContent =
      "(preview indisponivel para este arquivo neste navegador)";
    node.setDirtyCanvas(true, true);
  });

  const oldRemoved = node.onRemoved;
  node.onRemoved = function () {
    TRANSPORT.players.delete(preview);
    if (TRANSPORT.leader === preview) TRANSPORT.leader = null;
    if (oldRemoved) return oldRemoved.apply(this, arguments);
  };
  return preview;
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

function renderInfo(node) {
  const p = node._bruxosPrev;
  if (!p) return;
  const m = node._bruxosMeta || {};
  // Nao misturar estes dois estados:
  //   probe = metadados do ARQUIVO selecionado (antes do processamento)
  //   py    = shape REAL devolvido pela ultima execucao do node
  // Antes, probeAndFill fazia Object.assign em _bruxosPyInfo e sobrescrevia
  // 720x1280 (saida real) por 1080x1920 (arquivo original) logo apos executar.
  const probe = node._bruxosProbeInfo || {};
  const py = node._bruxosPyInfo || {};
  const lines = [];
  const srcW = probe.width || m.w;
  const srcH = probe.height || m.h;
  if (srcW && srcH) lines.push("arquivo   : " + srcW + "x" + srcH);

  // Mostra o pedido atual mesmo antes de rodar. Com os dois lados preenchidos,
  // esse e exatamente o shape que _resize_frame produz no Python.
  const get = (name, fallback = 0) => {
    const w = node.widgets && node.widgets.find((x) => x.name === name);
    return w ? w.value : fallback;
  };
  const cw = Number(get("custom_width", 0)) || 0;
  const ch = Number(get("custom_height", 0)) || 0;
  let reqW = cw, reqH = ch;
  if (srcW && srcH) {
    if (cw > 0 && ch <= 0) reqH = Math.max(1, Math.round(srcH * cw / srcW));
    if (ch > 0 && cw <= 0) reqW = Math.max(1, Math.round(srcW * ch / srcH));
  }
  if (reqW > 0 && reqH > 0) lines.push("pedido    : " + reqW + "x" + reqH);
  if (py.width && py.height) lines.push("saida real: " + py.width + "x" + py.height);
  else if (reqW > 0 && reqH > 0) lines.push("saida real: execute para confirmar");

  const frames = probe.frame_count || py.frame_count;
  if (frames) {
    let l = "frames    : " + frames;
    if (probe.trim_frames != null && probe.trim_frames !== frames)
      l += "  ->  " + probe.trim_frames + " apos corte";
    lines.push(l);
  }
  const secs = probe.duration || m.dur;
  if (secs) {
    let l = "duracao   : " + (Math.round(secs * 100) / 100) + "s";
    if (probe.trim_duration != null && Math.abs(probe.trim_duration - secs) > 0.01)
      l += "  ->  " + (Math.round(probe.trim_duration * 100) / 100) + "s";
    lines.push(l);
  }
  if (probe.skip_first_frames || (probe.select_every_nth && probe.select_every_nth > 1) || probe.frame_load_cap) {
    lines.push("corte     : pula " + (probe.skip_first_frames || 0) +
      " | 1 a cada " + (probe.select_every_nth || 1) +
      " | limite " + (probe.frame_load_cap ? probe.frame_load_cap : "-"));
  }
  const f = py.output_fps || probe.trim_fps || probe.fps || py.source_fps;
  if (f) lines.push("fps       : " + (Math.round(f * 1000) / 1000));
  if (probe.format) lines.push("formato   : " + probe.format);
  if (py.has_audio != null) lines.push("audio     : " + (py.has_audio ? "sim" : "nao"));
  lines.push("preview   : arquivo original (a saida segue os numeros acima)");
  p.info.textContent = lines.join("\n");
}

// monta a URL do preview JA CORTADO (servidor aplica skip/cap/nth/force_rate)
function previewURL(node, ref) {
  const tp = trimParams(node);
  const p = new URLSearchParams();
  p.set("filename", ref.filename);
  p.set("type", ref.type || "input");
  p.set("subfolder", ref.subfolder || "");
  p.set("maxside", String(PREVIEW_MAX_SIDE));
  for (const k in tp) if (tp[k] != null && tp[k] !== "") p.set(k, tp[k]);
  // token por seleção: garante que o navegador nunca reuse o preview de OUTRO video
  p.set("v", (node._bruxosPrevVer || 0).toString());
  return api.apiURL("/bruxos/video_preview?" + p.toString());
}

// recarrega o preview cortado (debounce p/ nao spammar ao arrastar slider)
function refreshPreview(node) {
  const p = ensurePreview(node);
  const ref = refFromInputWidget(node);
  if (!ref) return;
  node._bruxosPrevVer = (node._bruxosPrevVer || 0) + 1;  // URL unica por atualizacao
  if (node._bruxosPrevTimer) clearTimeout(node._bruxosPrevTimer);
  node._bruxosPrevTimer = setTimeout(() => {
    p.video.src = previewURL(node, ref);
    p.video.style.display = "block";
    p.video.load();
    playGroup(p);
    resizeNodeToContent(node);
  }, 250);
  probeAndFill(node, ref, "input");   // atualiza os numeros (frames apos corte)
}

// le os valores de corte dos widgets do node
function trimParams(node) {
  const g = (n) => {
    const w = node.widgets && node.widgets.find((x) => x.name === n);
    return w ? w.value : undefined;
  };
  return {
    skip_first_frames: g("skip_first_frames"),
    select_every_nth: g("select_every_nth"),
    frame_load_cap: g("frame_load_cap"),
    force_rate: g("force_rate"),
  };
}

// pergunta ao servidor frames/resolucao/fps/duracao do arquivo escolhido
function probeAndFill(node, ref, folderType) {
  if (!ref || !ref.filename) return;
  const sub = ref.subfolder ? encodeURIComponent(ref.subfolder) : "";
  const type = ref.type || folderType || "input";
  let url = `/bruxos/video_probe?filename=${encodeURIComponent(ref.filename)}` +
              `&type=${type}&subfolder=${sub}`;
  const tp = trimParams(node);
  for (const k in tp) if (tp[k] != null && tp[k] !== "") url += `&${k}=${encodeURIComponent(tp[k])}`;
  api.fetchApi(url)
    .then((r) => (r.ok ? r.json() : null))
    .then((info) => {
      if (!info || info.error) return;
      node._bruxosProbeInfo = info;
      renderInfo(node);
      node.setDirtyCanvas(true, true);
    })
    .catch(() => {});
}

function showVideo(node, ref, folderType) {
  const p = ensurePreview(node);
  if (!ref || !ref.filename) return;
  p.video.src = viewURL(ref, folderType);
  p.video.style.display = "block";
  p.video.load();
  playGroup(p);
  resizeNodeToContent(node);
  probeAndFill(node, ref, folderType);   // preenche frames/resolucao na hora
}

function refFromInputWidget(node) {
  const w = node.widgets && node.widgets.find((x) => x.name === "video");
  const pathW = node.widgets && node.widgets.find((x) => x.name === "video_path");
  if (pathW && pathW.value && String(pathW.value).trim()) return null;
  if (!w || !w.value) return null;
  const val = String(w.value).replace(/\\/g, "/");
  const idx = val.lastIndexOf("/");
  return {
    filename: idx >= 0 ? val.slice(idx + 1) : val,
    subfolder: idx >= 0 ? val.slice(0, idx) : "",
    type: "input",
  };
}

function hookLoadVideo(node) {
  ensurePreview(node);
  // Nao imponha altura minima aqui. O onNodeCreated tambem roda ao restaurar
  // workflows e os antigos 620px sobrescreviam o resize manual do usuario,
  // deixando uma enorme area vazia com os advanced inputs escondidos.
  if (!node.size?.[0] || node.size[0] < 320)
    node.setSize([Math.max(320, node.size?.[0] || 0), Math.max(120, node.size?.[1] || 0)]);
  const vWidget = node.widgets && node.widgets.find((x) => x.name === "video");
  if (vWidget) {
    const orig = vWidget.callback;
    vWidget.callback = function () {
      const r = orig ? orig.apply(this, arguments) : undefined;
      refreshPreview(node);
      return r;
    };
  }
  // Ao mexer nos widgets de corte, re-renderiza o preview cortado + numeros.
  ["skip_first_frames", "select_every_nth", "frame_load_cap", "force_rate"].forEach((nm) => {
    const w = node.widgets && node.widgets.find((x) => x.name === nm);
    if (!w) return;
    const o = w.callback;
    w.callback = function () {
      const r = o ? o.apply(this, arguments) : undefined;
      refreshPreview(node);
      return r;
    };
  });
  // Resize e fit nao mudam o player (ele e uma pre-visualizacao leve do arquivo),
  // mas os numeros precisam responder imediatamente para nao parecer que o node
  // ignorou custom_width/custom_height.
  ["custom_width", "custom_height", "target_width", "target_height", "fit_mode", "girar"].forEach((nm) => {
    const w = node.widgets && node.widgets.find((x) => x.name === nm);
    if (!w) return;
    const o = w.callback;
    w.callback = function () {
      const r = o ? o.apply(this, arguments) : undefined;
      // A configuracao mudou: a ultima 'saida real' pertence a execucao anterior.
      node._bruxosPyInfo = null;
      renderInfo(node);
      node.setDirtyCanvas(true, true);
      return r;
    };
  });
  refreshPreview(node);
}

app.registerExtension({
  name: "BruxosDoVFX.VideoPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const name = nodeData && nodeData.name;
    if (name === "BruxosLoadVideo") {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onCreated ? onCreated.apply(this, arguments) : undefined;
        hookLoadVideo(this);
        return r;
      };
      // onConfigure roda DEPOIS que o ComfyUI restaura os valores salvos do
      // workflow. Sem isto, o refreshPreview inicial (em onNodeCreated) pega o
      // valor velho/vazio do widget 'video' e mostra OUTRO video ao reabrir a
      // workflow -- so corrigia se o usuario trocasse o seletor na mao.
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
        // agora os widgets ja tem o valor salvo; re-renderiza no video certo.
        // 2 disparos: imediato e um com folga (caso o valor chegue 1 tick depois)
        try { refreshPreview(this); } catch (e) {}
        setTimeout(() => { try { refreshPreview(this); } catch (e) {} }, 60);
        return r;
      };
      const onExec = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        if (onExec) onExec.apply(this, arguments);
        try {
          if (message && message.bruxos_info && message.bruxos_info[0]) {
            this._bruxosPyInfo = JSON.parse(message.bruxos_info[0]);
            renderInfo(this);
          }
          // mantem o preview no video SELECIONADO (cortado), nao troca pelo
          // que veio na mensagem (evita "pular" p/ outro arquivo).
          const sel = refFromInputWidget(this);
          if (sel) refreshPreview(this);
          else if (message && message.bruxos_video && message.bruxos_video[0])
            showVideo(this, message.bruxos_video[0], "input");
        } catch (e) { console.warn("[Bruxos] info parse", e); }
      };
    }

    if (name === "BruxosSaveVideo") {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onCreated ? onCreated.apply(this, arguments) : undefined;
        ensurePreview(this);
        return r;
      };
      const onExec = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        if (onExec) onExec.apply(this, arguments);
        const ref = message && (
          (message.gifs && message.gifs[0]) ||
          (message.videos && message.videos[0]) ||
          (message.images && message.images[0])
        );
        if (ref) showVideo(this, ref, "output");
      };
    }
  },
});
