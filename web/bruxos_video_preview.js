import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Preview de video (DOM widget, Nodes 2.0) para Load Video (Bruxos) e
// Save Video (Bruxos), + infos do video no Load.
console.log("[Bruxos] preview de video carregado");

const MAX_H = 240;   // altura maxima do player
const INFO_H = 46;   // area de texto de infos

function viewURL(ref, folderType) {
  const sub = ref.subfolder ? encodeURIComponent(ref.subfolder) : "";
  const type = ref.type || folderType || "input";
  return api.apiURL(
    `/view?filename=${encodeURIComponent(ref.filename)}` +
    `&type=${type}&subfolder=${sub}&rand=${Math.random().toString(36).slice(2)}`
  );
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
  video.controls = true;
  // min-width:0 / max-width:100% evitam que o video vaze do node
  video.style.cssText =
    "display:block;width:100%;max-width:100%;min-width:0;height:auto;" +
    "max-height:" + MAX_H + "px;object-fit:contain;background:#000;" +
    "border-radius:6px;";

  const info = document.createElement("div");
  info.style.cssText =
    "width:100%;box-sizing:border-box;margin-top:4px;font-size:10px;" +
    "line-height:1.35;color:#bbb;font-family:monospace;white-space:pre-wrap;" +
    "word-break:break-word;text-align:left;";

  wrap.append(video, info);

  const widget = node.addDOMWidget("bruxos_preview", "preview", wrap, {
    serialize: false,
    hideOnZoom: false,
  });

  widget.computeSize = function (width) {
    let h = INFO_H;
    if (node._bruxosPrev && node._bruxosPrev.video.style.display !== "none") {
      const aspect = node._bruxosMeta && node._bruxosMeta.aspect;
      // -24: margem p/ a moldura do node; evita o preview vazar pela lateral
      const w = (node.size && node.size[0] ? node.size[0] : width) - 24;
      if (aspect) h += Math.min(MAX_H, Math.max(60, w / aspect)) + 8;
      else h += 160;
    }
    return [width, h];
  };

  video.addEventListener("loadedmetadata", () => {
    node._bruxosMeta = {
      w: video.videoWidth,
      h: video.videoHeight,
      dur: video.duration,
      aspect: video.videoWidth && video.videoHeight
        ? video.videoWidth / video.videoHeight : null,
    };
    renderInfo(node);
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
  });
  video.addEventListener("error", () => {
    node._bruxosPrev.info.textContent =
      "(preview indisponivel para este arquivo neste navegador)";
    node.setDirtyCanvas(true, true);
  });

  node._bruxosPrev = { wrap, video, info, widget };
  return node._bruxosPrev;
}

function renderInfo(node) {
  const p = node._bruxosPrev;
  if (!p) return;
  const m = node._bruxosMeta || {};
  const py = node._bruxosPyInfo || {};
  const lines = [];
  const W = py.width || m.w;
  const H = py.height || m.h;
  if (W && H) lines.push("resolucao : " + W + "x" + H);
  if (py.frame_count) {
    let l = "frames    : " + py.frame_count;
    if (py.trim_frames != null && py.trim_frames !== py.frame_count)
      l += "  ->  " + py.trim_frames + " apos corte";
    lines.push(l);
  }
  const secs = py.duration || m.dur;
  if (secs) {
    let l = "duracao   : " + (Math.round(secs * 100) / 100) + "s";
    if (py.trim_duration != null && Math.abs(py.trim_duration - secs) > 0.01)
      l += "  ->  " + (Math.round(py.trim_duration * 100) / 100) + "s";
    lines.push(l);
  }
  if (py.skip_first_frames || (py.select_every_nth && py.select_every_nth > 1) || py.frame_load_cap) {
    lines.push("corte     : pula " + (py.skip_first_frames || 0) +
      " | 1 a cada " + (py.select_every_nth || 1) +
      " | limite " + (py.frame_load_cap ? py.frame_load_cap : "-"));
  }
  const f = py.output_fps || py.fps || py.source_fps;
  if (f) lines.push("fps       : " + (Math.round(f * 1000) / 1000));
  if (py.format) lines.push("formato   : " + py.format);
  if (py.has_audio != null) lines.push("audio     : " + (py.has_audio ? "sim" : "nao"));
  p.info.textContent = lines.join("\n");
}

// monta a URL do preview JA CORTADO (servidor aplica skip/cap/nth/force_rate)
function previewURL(node, ref) {
  const tp = trimParams(node);
  const p = new URLSearchParams();
  p.set("filename", ref.filename);
  p.set("type", ref.type || "input");
  p.set("subfolder", ref.subfolder || "");
  p.set("maxside", "360");
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
    p.video.play && p.video.play().catch(() => {});
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
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
      node._bruxosPyInfo = Object.assign({}, node._bruxosPyInfo || {}, info);
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
  p.video.play && p.video.play().catch(() => {});
  node.setSize(node.computeSize());
  node.setDirtyCanvas(true, true);
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
  const vWidget = node.widgets && node.widgets.find((x) => x.name === "video");
  if (vWidget) {
    const orig = vWidget.callback;
    vWidget.callback = function () {
      const r = orig ? orig.apply(this, arguments) : undefined;
      refreshPreview(node);
      return r;
    };
  }
  // ao mexer nos widgets de corte, re-renderiza o preview cortado + numeros
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
