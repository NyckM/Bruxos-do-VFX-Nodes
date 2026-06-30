import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Preview de video (DOM widget, compativel com Nodes 2.0) para os nodes
// Load Video (Bruxos) e Save Video (Bruxos), + infos do video no Load.
console.log("[Bruxos] preview de video carregado");

const PREVIEW_H = 200;

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
    "width:100%;display:flex;flex-direction:column;align-items:center;gap:4px;";

  const video = document.createElement("video");
  video.muted = true;
  video.loop = true;
  video.autoplay = true;
  video.playsInline = true;
  video.controls = true;
  video.style.cssText =
    "width:100%;max-height:" + PREVIEW_H + "px;object-fit:contain;" +
    "background:#000;border-radius:6px;display:none;";

  const info = document.createElement("div");
  info.style.cssText =
    "width:100%;font-size:10px;line-height:1.35;color:#bbb;" +
    "font-family:monospace;white-space:pre-wrap;word-break:break-word;text-align:left;";

  wrap.append(video, info);

  const widget = node.addDOMWidget("bruxos_preview", "preview", wrap, {
    serialize: false,
    hideOnZoom: false,
  });
  widget.computeSize = function (w) {
    const extra = video.style.display === "none" ? 0 : PREVIEW_H + 8;
    return [w, extra + 46];
  };

  video.addEventListener("loadedmetadata", () => {
    node._bruxosMeta = {
      w: video.videoWidth,
      h: video.videoHeight,
      dur: video.duration,
    };
    renderInfo(node);
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
  if (py.frame_count != null) lines.push("frames    : " + py.frame_count);
  if (py.output_fps || py.source_fps) {
    const f = py.output_fps || py.source_fps;
    lines.push("fps       : " + (Math.round(f * 1000) / 1000));
  } else if (m.dur && m.w) {
    lines.push("duracao   : " + (Math.round(m.dur * 100) / 100) + "s");
  }
  if (py.format) lines.push("formato   : " + py.format);
  if (py.has_audio != null) lines.push("audio     : " + (py.has_audio ? "sim" : "nao"));
  p.info.textContent = lines.join("\n");
}

function showVideo(node, ref, folderType) {
  const p = ensurePreview(node);
  if (!ref || !ref.filename) return;
  p.video.src = viewURL(ref, folderType);
  p.video.style.display = "block";
  p.video.play?.().catch(() => {});
  p.widget.computeSize && node.setSize(node.computeSize());
  node.setDirtyCanvas(true, true);
}

// ---- Load Video: preview ao vivo a partir do widget "video" ----
function refFromInputWidget(node) {
  const w = node.widgets?.find((x) => x.name === "video");
  const pathW = node.widgets?.find((x) => x.name === "video_path");
  if (pathW && pathW.value && String(pathW.value).trim()) return null; // caminho absoluto: /view nao serve
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
  const vWidget = node.widgets?.find((x) => x.name === "video");
  if (vWidget) {
    const orig = vWidget.callback;
    vWidget.callback = function () {
      const r = orig ? orig.apply(this, arguments) : undefined;
      const ref = refFromInputWidget(node);
      if (ref) showVideo(node, ref, "input");
      return r;
    };
  }
  // preview inicial (se ja tiver um valor)
  const ref0 = refFromInputWidget(node);
  if (ref0) showVideo(node, ref0, "input");
}

app.registerExtension({
  name: "BruxosDoVFX.VideoPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const name = nodeData?.name;
    if (name === "BruxosLoadVideo") {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onCreated ? onCreated.apply(this, arguments) : undefined;
        hookLoadVideo(this);
        return r;
      };
      const onExec = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExec?.apply(this, arguments);
        try {
          if (message?.bruxos_info?.[0]) {
            this._bruxosPyInfo = JSON.parse(message.bruxos_info[0]);
            renderInfo(this);
          }
          if (message?.bruxos_video?.[0]) showVideo(this, message.bruxos_video[0], "input");
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
        onExec?.apply(this, arguments);
        const ref = message?.gifs?.[0] || message?.videos?.[0] || message?.images?.[0];
        if (ref) showVideo(this, ref, "output");
      };
    }
  },
});
