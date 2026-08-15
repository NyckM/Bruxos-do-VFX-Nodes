// Bruxos Motion Timeline — DAW-style timeline + mask painter for the Motion Lab.
// UI derivada de ComfyUI-MAINodes/web/motion_editor.js.
// Copyright (c) 2026 MatlowAI (Dylan Matlow), usada sob licenca MIT.
// Edits the BruxosMotionTimeline node's serialized state (see the node docstring
// for the contract). Queue once to load the filmstrip; edits then happen
// entirely client-side until the next queue.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LEGAL_STEP = 17;
// per-step cost goes as tokens**COST_EXP (attention dominates); see motion.py
const COST_EXP = 1.7;
const tokensFor = (n) => Math.floor((n - 5) / 17) * 5 + 2;

function tokStartFrame(t) {
  const c = Math.floor(t / 5), i = t % 5;
  return c * 17 + (i === 0 ? 0 : 4 * (i - 1) + 1);
}
function tokenStarts(len) {
  const out = [];
  for (let t = 0; tokStartFrame(t) < len; t++) out.push(tokStartFrame(t));
  return out;
}
function legalCeil(n) {
  const k = Math.max(2, Math.ceil((n - 5) / LEGAL_STEP));
  return LEGAL_STEP * k + 5;
}
function uid() { return Math.random().toString(36).slice(2, 8); }
function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }

const PARAM_RANGE = { hold: [0, 8], feather: [0, 256], strength: [0, 1] };
const BLOCK_COLORS = ["#e06f6f", "#6fa8e0", "#7fce7f", "#d8b45a", "#b98fd6",
                      "#5ec8c0"];

function envValue(auto, param, frame, dflt) {
  const pts = auto?.[param];
  if (!pts || !pts.length) return dflt;
  const s = [...pts].sort((a, b) => a[0] - b[0]);
  if (frame <= s[0][0]) return s[0][1];
  if (frame >= s[s.length - 1][0]) return s[s.length - 1][1];
  for (let i = 0; i < s.length - 1; i++) {
    const [f0, v0] = s[i], [f1, v1] = s[i + 1];
    if (f0 <= frame && frame <= f1)
      return f1 === f0 ? v1 : v0 + (frame - f0) / (f1 - f0) * (v1 - v0);
  }
  return dflt;
}

class MotionEditor {
  constructor(node) {
    this.node = node;
    this.state = { v: 1, blocks: [] };
    this.frames = { paint: [], strip: [] };
    this.profile = [];
    this.length = 124;
    this.fps = 24;
    this.cur = 0;
    this.sel = null;            // selected block id
    this.tool = "brush";
    this.brushR = 0.035;
    this.target = "frame";      // "frame" | "block"
    this.laneParam = null;
    this.onion = true;
    this.snap = true;
    this.undoStack = [];
    this.imgCache = new Map();
    this.report = "";
    this.drag = null;

    this.sw = node.widgets.find((w) => w.name === "editor_state");
    if (this.sw) {
      try { const s = JSON.parse(this.sw.value || "{}"); if (s.blocks) this.state = s; } catch (e) { /* fresh */ }
      this.sw.computeSize = () => [0, -4];
      if (this.sw.element) this.sw.element.style.display = "none";
      if (this.sw.inputEl) this.sw.inputEl.style.display = "none";
    }
    this.buildDOM();
    this.redrawAll();
  }

  // ---------- state ----------
  pushUndo() {
    this.undoStack.push(JSON.stringify(this.state));
    if (this.undoStack.length > 40) this.undoStack.shift();
  }
  undo() {
    const s = this.undoStack.pop();
    if (!s) return;
    this.state = JSON.parse(s);
    if (this.sel && !this.block(this.sel)) this.sel = null;
    this.commit(false);
    this.redrawAll();
  }
  commit(push = true) {
    if (push === "undo") this.pushUndo();
    if (this.sw) this.sw.value = JSON.stringify(this.state);
    this.node.graph?.setDirtyCanvas(true, true);
    this.updatePrice();
  }
  block(id) { return this.state.blocks.find((b) => b.id === id); }

  defaultBlock(a, z) {
    return { id: uid(), start: a, end: z, hold: 0,
      dials: { feather: 48, profile: "smoothstep", direction: "centered",
               grow: 0, fade: 6, strength: 1.0 },
      auto: {}, strokes: {}, static_strokes: [] };
  }

  snapRange(a, z) {
    if (!this.snap) return [a, z];
    const ts = tokenStarts(this.length);
    let s = 0, e = this.length - 1;
    for (const t of ts) if (t <= a) s = t;
    for (let i = 0; i < ts.length; i++)
      if (ts[i] > z) { e = ts[i] - 1; break; }
    return [s, e];
  }

  updatePrice() {
    const holds = new Array(this.length).fill(1);
    for (const b of this.state.blocks)
      for (let f = b.start; f <= Math.min(b.end, this.length - 1); f++) {
        let h = Math.round(envValue(b.auto, "hold", f, b.hold || 0));
        if (h <= 0) h = 4; // oracle value unknown client-side; estimate at peak
        holds[f] = Math.max(holds[f], h);
      }
    const dil = this.state.blocks.length
      ? legalCeil(holds.reduce((x, y) => x + y, 0)) : this.length;
    const t = (n) => (n / this.fps).toFixed(1);
    const timeX = Math.pow(
      tokensFor(dil) / Math.max(tokensFor(legalCeil(this.length)), 1), COST_EXP);
    this.priceEl.textContent =
      `${this.length}f (${t(this.length)}s) -> ~${dil}f (${t(dil)}s), ` +
      `${(dil / this.length).toFixed(2)}x frames / ~${timeX.toFixed(1)}x time per step` +
      (this.report ? ` | last run: ${this.report}` : "");
  }

  // ---------- server data ----------
  onExecuted(msg) {
    if (msg?.bruxos_motion_frames) this.frames.paint = msg.bruxos_motion_frames;
    if (msg?.bruxos_motion_frames) this.frames.strip = msg.bruxos_motion_frames;
    if (msg?.bruxos_motion_profile) this.profile = msg.bruxos_motion_profile;
    if (msg?.bruxos_motion_length?.length) this.length = msg.bruxos_motion_length[0];
    if (msg?.bruxos_motion_fps?.length) this.fps = msg.bruxos_motion_fps[0];
    if (msg?.bruxos_motion_report?.length) this.report = msg.bruxos_motion_report[0];
    this.imgCache.clear();
    this.cur = clamp(this.cur, 0, this.length - 1);
    this.redrawAll();
  }
  img(list, i, cb) {
    const f = list[i];
    if (!f) return null;
    const key = f.filename;
    if (this.imgCache.has(key)) return this.imgCache.get(key);
    const im = new Image();
    im.src = api.apiURL(
      `/view?filename=${encodeURIComponent(f.filename)}` +
      `&type=${f.type}&subfolder=${encodeURIComponent(f.subfolder)}`);
    im.onload = () => cb && cb();
    this.imgCache.set(key, im);
    if (this.imgCache.size > 300) {
      const k0 = this.imgCache.keys().next().value;
      this.imgCache.delete(k0);
    }
    return im;
  }

  // ---------- DOM ----------
  el(tag, style, parent, text) {
    const e = document.createElement(tag);
    Object.assign(e.style, style || {});
    if (text !== undefined) e.textContent = text;
    parent?.appendChild(e);
    return e;
  }
  btn(parent, label, cb, title) {
    const b = this.el("button", {
      background: "#333", color: "#ddd", border: "1px solid #555",
      borderRadius: "3px", padding: "2px 7px", cursor: "pointer",
      fontSize: "11px" }, parent, label);
    if (title) b.title = title;
    b.onclick = (e) => { e.stopPropagation(); cb(b); };
    return b;
  }

  buildDOM() {
    const root = this.el("div", {
      display: "flex", flexDirection: "column", gap: "4px",
      background: "#1b1b1b", padding: "6px", borderRadius: "6px",
      fontFamily: "sans-serif", fontSize: "11px", color: "#ccc",
      width: "100%", boxSizing: "border-box" });
    root.addEventListener("pointerdown", (e) => e.stopPropagation());
    root.addEventListener("keydown", (e) => e.stopPropagation());
    root.tabIndex = 0;
    root.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") { this.setFrame(this.cur - 1); e.preventDefault(); }
      if (e.key === "ArrowRight") { this.setFrame(this.cur + 1); e.preventDefault(); }
      if ((e.ctrlKey || e.metaKey) && e.key === "z") { this.undo(); e.preventDefault(); }
      if (e.key === "Delete" && this.sel) this.deleteBlock(this.sel);
    });

    // toolbar
    const tb = this.el("div", { display: "flex", gap: "5px", flexWrap: "wrap",
                                alignItems: "center" }, root);
    this.btn(tb, "+ block", () => {
      this.pushUndo();
      const a = clamp(this.cur, 0, this.length - 9);
      const [s, e] = this.snapRange(a, a + 16);
      const b = this.defaultBlock(s, e);
      this.state.blocks.push(b);
      this.sel = b.id;
      this.commit();
      this.redrawAll();
    }, "add a time block at the playhead (or drag on the block lane)");
    this.btn(tb, "delete", () => this.sel && this.deleteBlock(this.sel));
    this.snapBtn = this.btn(tb, "snap: on", (b) => {
      this.snap = !this.snap;
      b.textContent = `snap: ${this.snap ? "on" : "off"}`;
    }, "snap block edges to the model's token grid");
    this.el("span", { width: "8px" }, tb);
    this.toolBtns = {};
    for (const t of ["brush", "erase"]) {
      this.toolBtns[t] = this.btn(tb, t, () => this.setTool(t));
    }
    const bs = this.el("input", { width: "70px" }, tb);
    bs.type = "range"; bs.min = "0.005"; bs.max = "0.15"; bs.step = "0.005";
    bs.value = String(this.brushR);
    bs.title = "brush size";
    bs.oninput = () => { this.brushR = parseFloat(bs.value); this.drawPaint(); };
    this.targetBtn = this.btn(tb, "paint: this frame", (b) => {
      this.target = this.target === "frame" ? "block" : "frame";
      b.textContent = this.target === "frame" ? "paint: this frame"
                                              : "paint: whole block";
    }, "whether strokes apply to the current frame or the whole block");
    this.btn(tb, "clear frame", () => {
      const b = this.block(this.sel); if (!b) return;
      this.pushUndo();
      delete (b.strokes || {})[String(this.cur)];
      this.commit(); this.drawPaint();
    });
    this.onionBtn = this.btn(tb, "onion: on", (b) => {
      this.onion = !this.onion;
      b.textContent = `onion: ${this.onion ? "on" : "off"}`;
      this.drawPaint();
    });
    this.btn(tb, "undo", () => this.undo(), "ctrl+z");
    this.setTool("brush");

    // timeline
    this.tl = this.el("canvas", { width: "100%", cursor: "crosshair",
                                  borderRadius: "4px" }, root);
    this.tl.height = 130;
    this.bindTimeline();

    // automation lane
    this.laneWrap = this.el("div", { display: "none", flexDirection: "column",
                                     gap: "2px" }, root);
    this.laneLabel = this.el("div", {}, this.laneWrap, "automation");
    this.lane = this.el("canvas", { width: "100%", borderRadius: "4px",
                                    cursor: "crosshair" }, this.laneWrap);
    this.lane.height = 66;
    this.bindLane();

    // paint area
    this.paintWrap = this.el("div", { position: "relative", width: "100%" }, root);
    this.paint = this.el("canvas", { width: "100%", borderRadius: "4px",
                                     cursor: "none", display: "block" },
                         this.paintWrap);
    this.paint.height = 300;
    this.hint = this.el("div", {
      position: "absolute", inset: "0", display: "flex",
      alignItems: "center", justifyContent: "center", color: "#888",
      pointerEvents: "none", textAlign: "center" }, this.paintWrap,
      "queue once to load the filmstrip, then drag a block on the timeline and paint");
    this.bindPaint();

    // dials
    this.dials = this.el("div", { display: "flex", gap: "10px",
                                  flexWrap: "wrap", alignItems: "center",
                                  minHeight: "22px" }, root);
    this.priceEl = this.el("div", { color: "#9c9", minHeight: "14px" }, root);

    const dom = this.node.addDOMWidget("bruxos_motion_ui", "div", root, { serialize: false });
    dom.serialize = false;
    dom.serializeValue = () => undefined;
    const sz = this.node.size;
    this.node.setSize([Math.max(sz[0], 640), Math.max(sz[1], 700)]);
    this.root = root;
    new ResizeObserver(() => this.redrawAll()).observe(root);
  }

  setTool(t) {
    this.tool = t;
    for (const k in this.toolBtns)
      this.toolBtns[k].style.background = k === t ? "#5a5" : "#333";
  }
  setFrame(f) {
    this.cur = clamp(f, 0, this.length - 1);
    this.drawTimeline(); this.drawPaint(); this.drawLane();
  }
  deleteBlock(id) {
    this.pushUndo();
    this.state.blocks = this.state.blocks.filter((b) => b.id !== id);
    if (this.sel === id) this.sel = null;
    this.commit(); this.redrawAll();
  }

  // ---------- timeline ----------
  fx(f, w) { return f / Math.max(1, this.length) * w; }
  xf(x, w) { return clamp(Math.floor(x / w * this.length), 0, this.length - 1); }

  bindTimeline() {
    const c = this.tl;
    const pos = (e) => {
      const r = c.getBoundingClientRect();
      return [(e.clientX - r.left) * (c.width / r.width),
              (e.clientY - r.top) * (c.height / r.height)];
    };
    c.onpointerdown = (e) => {
      e.stopPropagation();
      const [x, y] = pos(e);
      const w = c.width;
      c.setPointerCapture(e.pointerId);
      if (y < 56 || y > 108) {              // strip or ruler: scrub
        this.drag = { kind: "scrub" };
        this.setFrame(this.xf(x, w));
        return;
      }
      // block lane
      const f = this.xf(x, w);
      for (const b of this.state.blocks) {
        const x0 = this.fx(b.start, w), x1 = this.fx(b.end + 1, w);
        if (x >= x0 - 4 && x <= x0 + 4) {
          this.pushUndo(); this.sel = b.id;
          this.drag = { kind: "resizeL", id: b.id }; return;
        }
        if (x >= x1 - 4 && x <= x1 + 4) {
          this.pushUndo(); this.sel = b.id;
          this.drag = { kind: "resizeR", id: b.id }; return;
        }
        if (x > x0 && x < x1) {
          this.pushUndo(); this.sel = b.id;
          this.drag = { kind: "move", id: b.id, grab: f - b.start };
          this.buildDials(); this.redrawAll(); return;
        }
      }
      this.drag = { kind: "create", a: f, z: f };
    };
    c.onpointermove = (e) => {
      if (!this.drag) return;
      const [x] = pos(e);
      const f = this.xf(x, c.width);
      const d = this.drag;
      if (d.kind === "scrub") { this.setFrame(f); return; }
      if (d.kind === "create") { d.z = f; this.drawTimeline(); return; }
      const b = this.block(d.id);
      if (!b) return;
      if (d.kind === "move") {
        const span = b.end - b.start;
        b.start = clamp(f - d.grab, 0, this.length - 1 - span);
        b.end = b.start + span;
      } else if (d.kind === "resizeL") b.start = clamp(f, 0, b.end);
      else if (d.kind === "resizeR") b.end = clamp(f, b.start, this.length - 1);
      this.drawTimeline(); this.updatePrice();
    };
    c.onpointerup = () => {
      const d = this.drag;
      this.drag = null;
      if (!d) return;
      if (d.kind === "create") {
        const a = Math.min(d.a, d.z), z = Math.max(d.a, d.z);
        if (z - a >= 2) {
          this.pushUndo();
          const [s, e2] = this.snapRange(a, z);
          const b = this.defaultBlock(s, e2);
          this.state.blocks.push(b);
          this.sel = b.id;
        }
      } else if (d.kind === "resizeL" || d.kind === "resizeR") {
        const b = this.block(d.id);
        if (b) [b.start, b.end] = this.snapRange(b.start, b.end);
      }
      this.commit(); this.redrawAll();
    };
  }

  drawTimeline() {
    const c = this.tl, g = c.getContext("2d");
    const w = c.width = c.clientWidth * (window.devicePixelRatio || 1) / (window.devicePixelRatio || 1) || c.width;
    c.width = c.clientWidth || 600;
    const W = c.width, H = c.height;
    g.fillStyle = "#111"; g.fillRect(0, 0, W, H);

    // filmstrip 0..54
    if (this.frames.strip.length) {
      const per = Math.max(1, Math.ceil(this.length / Math.floor(W / 34)));
      for (let f = 0; f < this.length; f += per) {
        const im = this.img(this.frames.strip, f, () => this.drawTimeline());
        const x = this.fx(f, W);
        if (im?.complete)
          g.drawImage(im, x, 0, Math.max(this.fx(per, W), 34), 54);
      }
    } else {
      g.fillStyle = "#666";
      g.fillText("filmstrip loads after the first queue", 8, 30);
    }

    // jerk profile 56..102
    g.fillStyle = "#182018"; g.fillRect(0, 56, W, 46);
    if (this.profile.length) {
      const mx = Math.max(...this.profile, 0.001);
      g.strokeStyle = "#7fce7f"; g.beginPath();
      this.profile.forEach((v, t) => {
        const x = this.fx(tokStartFrame(t), W);
        const y = 100 - v / mx * 40;
        t ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke();
    }
    // token grid ticks
    g.strokeStyle = "#2c2c2c";
    for (const t of tokenStarts(this.length)) {
      const x = this.fx(t, W);
      g.beginPath(); g.moveTo(x, 56); g.lineTo(x, 102); g.stroke();
    }

    // blocks overlay on 56..108
    this.state.blocks.forEach((b, i) => {
      const col = BLOCK_COLORS[i % BLOCK_COLORS.length];
      const x0 = this.fx(b.start, W), x1 = this.fx(b.end + 1, W);
      g.fillStyle = col + (b.id === this.sel ? "66" : "33");
      g.fillRect(x0, 56, x1 - x0, 52);
      g.fillStyle = col;
      g.fillRect(x0, 56, 3, 52); g.fillRect(x1 - 3, 56, 3, 52);
      g.font = "10px sans-serif";
      const painted = Object.keys(b.strokes || {}).length ||
                      (b.static_strokes || []).length;
      g.fillText(`${b.hold ? "h" + b.hold : "oracle"}${painted ? " *" : ""}`,
                 x0 + 6, 68);
    });

    // create preview
    if (this.drag?.kind === "create") {
      const x0 = this.fx(Math.min(this.drag.a, this.drag.z), W);
      const x1 = this.fx(Math.max(this.drag.a, this.drag.z) + 1, W);
      g.fillStyle = "#ffffff22"; g.fillRect(x0, 56, x1 - x0, 52);
    }

    // ruler + playhead
    g.fillStyle = "#222"; g.fillRect(0, 110, W, 20);
    g.fillStyle = "#888"; g.font = "9px sans-serif";
    for (let s = 0; s <= this.length / this.fps; s++) {
      const x = this.fx(s * this.fps, W);
      g.fillRect(x, 110, 1, 6); g.fillText(`${s}s`, x + 2, 126);
    }
    g.fillStyle = "#fff";
    g.fillRect(this.fx(this.cur, W), 0, 1.5, H);
    g.fillText(`f${this.cur}`, clamp(this.fx(this.cur, W) + 4, 4, W - 30), 8);
  }

  // ---------- automation lane ----------
  bindLane() {
    const c = this.lane;
    const pos = (e) => {
      const r = c.getBoundingClientRect();
      return [(e.clientX - r.left) * (c.width / r.width),
              (e.clientY - r.top) * (c.height / r.height)];
    };
    const toVal = (y) => {
      const [lo, hi] = PARAM_RANGE[this.laneParam];
      return clamp(lo + (1 - y / c.height) * (hi - lo), lo, hi);
    };
    const toFrame = (x) => this.xf(x, c.width);
    c.onpointerdown = (e) => {
      e.stopPropagation();
      const b = this.block(this.sel);
      if (!b || !this.laneParam) return;
      c.setPointerCapture(e.pointerId);
      const [x, y] = pos(e);
      b.auto = b.auto || {};
      const pts = (b.auto[this.laneParam] = b.auto[this.laneParam] || []);
      const f = toFrame(x);
      let hit = pts.findIndex((p) =>
        Math.abs(this.fx(p[0], c.width) - x) < 7 &&
        Math.abs((1 - (p[1] - PARAM_RANGE[this.laneParam][0]) /
          (PARAM_RANGE[this.laneParam][1] - PARAM_RANGE[this.laneParam][0]))
          * c.height - y) < 9);
      this.pushUndo();
      if (e.detail === 2 && hit >= 0) {        // double click: delete
        pts.splice(hit, 1);
        this.commit(); this.drawLane(); return;
      }
      if (hit < 0) {
        pts.push([clamp(f, b.start, b.end), toVal(y)]);
        pts.sort((p, q) => p[0] - q[0]);
        hit = pts.findIndex((p) => p[0] === clamp(f, b.start, b.end));
      }
      this.drag = { kind: "env", idx: hit };
    };
    c.onpointermove = (e) => {
      if (this.drag?.kind !== "env") return;
      const b = this.block(this.sel);
      const pts = b?.auto?.[this.laneParam];
      if (!pts) return;
      const [x, y] = pos(e);
      pts[this.drag.idx] = [clamp(toFrame(x), b.start, b.end), toVal(y)];
      this.drawLane();
    };
    c.onpointerup = () => {
      if (this.drag?.kind !== "env") return;
      const b = this.block(this.sel);
      b?.auto?.[this.laneParam]?.sort((p, q) => p[0] - q[0]);
      this.drag = null;
      this.commit(); this.drawLane();
    };
  }

  drawLane() {
    if (this.laneWrap.style.display === "none") return;
    const c = this.lane, g = c.getContext("2d");
    c.width = c.clientWidth || 600;
    const W = c.width, H = c.height;
    g.fillStyle = "#141420"; g.fillRect(0, 0, W, H);
    const b = this.block(this.sel);
    if (!b || !this.laneParam) return;
    const [lo, hi] = PARAM_RANGE[this.laneParam];
    const x0 = this.fx(b.start, W), x1 = this.fx(b.end + 1, W);
    g.fillStyle = "#ffffff10"; g.fillRect(x0, 0, x1 - x0, H);
    const yFor = (v) => (1 - (v - lo) / (hi - lo)) * H;
    const dflt = this.laneParam === "hold" ? (b.hold || 4)
      : (b.dials?.[this.laneParam] ?? (this.laneParam === "strength" ? 1 : 48));
    g.strokeStyle = "#89a"; g.beginPath();
    for (let f = b.start; f <= b.end; f++) {
      const v = envValue(b.auto, this.laneParam, f, dflt);
      const x = this.fx(f, W), y = yFor(v);
      f === b.start ? g.moveTo(x, y) : g.lineTo(x, y);
    }
    g.stroke();
    for (const [f, v] of b.auto?.[this.laneParam] || []) {
      g.fillStyle = "#fff";
      g.beginPath(); g.arc(this.fx(f, W), yFor(v), 4, 0, 7); g.fill();
    }
    g.fillStyle = "#fff";
    g.fillRect(this.fx(this.cur, W), 0, 1, H);
    g.fillStyle = "#889";
    g.fillText(`${this.laneParam}  [${lo}..${hi}]  dbl-click point to delete`,
               6, 10);
  }

  // ---------- paint ----------
  bindPaint() {
    const c = this.paint;
    const pos = (e) => {
      const r = c.getBoundingClientRect();
      return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
    };
    c.onpointerdown = (e) => {
      e.stopPropagation();
      const b = this.block(this.sel);
      if (!b || this.cur < b.start || this.cur > b.end) return;
      c.setPointerCapture(e.pointerId);
      this.pushUndo();
      const [x, y] = pos(e);
      this.stroke = { t: this.tool, r: this.brushR, pts: [[x, y]] };
      this.drawPaint();
    };
    c.onpointermove = (e) => {
      const [x, y] = pos(e);
      this.cursor = [x, y];
      if (this.stroke) {
        const [px, py] = this.stroke.pts[this.stroke.pts.length - 1];
        if (Math.hypot(x - px, y - py) > this.brushR / 4)
          this.stroke.pts.push([x, y]);
      }
      this.drawPaint();
    };
    const done = () => {
      if (!this.stroke) return;
      const b = this.block(this.sel);
      if (b) {
        if (this.target === "block") {
          (b.static_strokes = b.static_strokes || []).push(this.stroke);
        } else {
          b.strokes = b.strokes || {};
          (b.strokes[String(this.cur)] =
            b.strokes[String(this.cur)] || []).push(this.stroke);
        }
      }
      this.stroke = null;
      this.commit(); this.drawTimeline(); this.drawPaint();
    };
    c.onpointerup = done;
    c.onpointerleave = () => { this.cursor = null; this.drawPaint(); };
    c.onwheel = (e) => {
      e.preventDefault(); e.stopPropagation();
      this.setFrame(this.cur + Math.sign(e.deltaY));
    };
  }

  rasterizeTo(g, strokes, W, H, color) {
    for (const s of strokes || []) {
      g.globalCompositeOperation =
        s.t === "erase" ? "destination-out" : "source-over";
      g.strokeStyle = g.fillStyle = color;
      g.lineWidth = s.r * 2 * W;
      g.lineCap = g.lineJoin = "round";
      g.beginPath();
      s.pts.forEach(([x, y], i) =>
        i ? g.lineTo(x * W, y * H) : g.moveTo(x * W, y * H));
      g.stroke();
      if (s.pts.length === 1) {
        g.beginPath();
        g.arc(s.pts[0][0] * W, s.pts[0][1] * H, s.r * W, 0, 7);
        g.fill();
      }
    }
    g.globalCompositeOperation = "source-over";
  }

  frameStrokes(b, f) {
    return [...(b.static_strokes || []), ...((b.strokes || {})[String(f)] || [])];
  }

  drawPaint() {
    const c = this.paint, g = c.getContext("2d");
    const im = this.frames.paint.length
      ? this.img(this.frames.paint, this.cur, () => this.drawPaint()) : null;
    const cw = c.clientWidth || 600;
    const ar = im?.complete && im.naturalWidth
      ? im.naturalHeight / im.naturalWidth : 9 / 16;
    c.width = cw; c.height = Math.round(cw * ar);
    const W = c.width, H = c.height;
    g.fillStyle = "#000"; g.fillRect(0, 0, W, H);
    if (im?.complete) { g.drawImage(im, 0, 0, W, H); this.hint.style.display = "none"; }
    else if (this.frames.paint.length) this.hint.style.display = "none";

    const b = this.block(this.sel);
    if (b && this.cur >= b.start && this.cur <= b.end) {
      const off = document.createElement("canvas");
      off.width = W; off.height = H;
      const og = off.getContext("2d");
      const strokes = this.frameStrokes(b, this.cur);
      if (this.stroke) strokes.push(this.stroke);
      if (strokes.length) {
        this.rasterizeTo(og, strokes, W, H, "#ff3030");
        g.globalAlpha = 0.45; g.drawImage(off, 0, 0); g.globalAlpha = 1;
      } else {
        g.fillStyle = "#ff303018"; g.fillRect(0, 0, W, H);
        g.fillStyle = "#f88"; g.font = "11px sans-serif";
        g.fillText("whole frame regenerates in this block; paint to narrow it",
                   8, H - 10);
      }
      if (this.onion) {
        for (const [df, col] of [[-1, "#30a0ff"], [1, "#30ff80"]]) {
          const f2 = this.cur + df;
          if (f2 < b.start || f2 > b.end) continue;
          const s2 = this.frameStrokes(b, f2)
            .filter((s) => (b.strokes || {})[String(f2)]?.includes(s));
          if (!s2.length) continue;
          const o2 = document.createElement("canvas");
          o2.width = W; o2.height = H;
          this.rasterizeTo(o2.getContext("2d"), s2, W, H, col);
          g.globalAlpha = 0.18; g.drawImage(o2, 0, 0); g.globalAlpha = 1;
        }
      }
    } else if (b) {
      g.fillStyle = "#ccc8";
      g.fillText("playhead is outside the selected block", 8, 16);
    }

    if (this.cursor && b) {
      g.strokeStyle = this.tool === "erase" ? "#3af" : "#f33";
      g.beginPath();
      g.arc(this.cursor[0] * W, this.cursor[1] * H, this.brushR * W, 0, 7);
      g.stroke();
    }
    g.fillStyle = "#fff"; g.font = "11px sans-serif";
    g.fillText(`frame ${this.cur}/${this.length - 1}`, W - 92, 14);
  }

  // ---------- dials ----------
  buildDials() {
    this.dials.innerHTML = "";
    const b = this.block(this.sel);
    if (!b) {
      this.laneWrap.style.display = "none";
      this.el("span", { color: "#777" }, this.dials,
              "select or drag out a block to edit its dials");
      return;
    }
    b.dials = b.dials || {};
    const num = (label, get, set, min, max, step, autoKey) => {
      const wrap = this.el("span", { display: "inline-flex", gap: "3px",
                                     alignItems: "center" }, this.dials);
      this.el("span", { color: "#999" }, wrap, label);
      const inp = this.el("input", { width: "52px", background: "#222",
                                     color: "#ddd", border: "1px solid #444" },
                          wrap);
      inp.type = "number"; inp.min = min; inp.max = max; inp.step = step;
      inp.value = String(get());
      inp.onchange = () => {
        this.pushUndo(); set(parseFloat(inp.value)); this.commit();
        this.drawTimeline(); this.drawLane();
      };
      if (autoKey) {
        const ab = this.btn(wrap, "A", () => {
          this.laneParam = this.laneParam === autoKey ? null : autoKey;
          this.laneWrap.style.display = this.laneParam ? "flex" : "none";
          this.laneLabel.textContent =
            `automation: ${autoKey} (click to add points, drag to shape, dbl-click to delete)`;
          for (const bt of this.autoBtns) bt.style.background = "#333";
          if (this.laneParam) ab.style.background = "#a80";
          this.drawLane();
        }, "draw an automation envelope for this dial");
        if ((b.auto || {})[autoKey]?.length) ab.style.borderColor = "#a80";
        this.autoBtns.push(ab);
      }
    };
    const sel = (label, key, opts) => {
      const wrap = this.el("span", { display: "inline-flex", gap: "3px",
                                     alignItems: "center" }, this.dials);
      this.el("span", { color: "#999" }, wrap, label);
      const s = this.el("select", { background: "#222", color: "#ddd",
                                    border: "1px solid #444" }, wrap);
      for (const o of opts) {
        const op = this.el("option", {}, s, o); op.value = o;
      }
      s.value = b.dials[key];
      s.onchange = () => { this.pushUndo(); b.dials[key] = s.value; this.commit(); };
    };
    this.autoBtns = [];
    num("hold (0=oracle)", () => b.hold || 0, (v) => (b.hold = Math.round(v)),
        0, 8, 1, "hold");
    num("feather px", () => b.dials.feather ?? 48,
        (v) => (b.dials.feather = Math.round(v)), 0, 256, 4, "feather");
    sel("profile", "profile", ["linear", "smoothstep", "gaussian"]);
    sel("direction", "direction", ["centered", "inward", "outward"]);
    num("grow", () => b.dials.grow ?? 0, (v) => (b.dials.grow = Math.round(v)),
        0, 64, 2);
    num("fade f", () => b.dials.fade ?? 6, (v) => (b.dials.fade = Math.round(v)),
        0, 24, 1);
    num("strength", () => b.dials.strength ?? 1,
        (v) => (b.dials.strength = v), 0, 1, 0.05, "strength");
  }

  redrawAll() {
    this.drawTimeline();
    this.drawLane();
    this.drawPaint();
    this.buildDials();
    this.updatePrice();
  }
}

app.registerExtension({
  name: "BruxosDoVFX.MotionTimeline",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "BruxosMotionTimeline") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      try { this.bruxosMotion = new MotionEditor(this); }
      catch (e) { console.error("BruxosMotionTimeline widget failed:", e); }
      return r;
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      this.bruxosMotion?.onExecuted(message);
    };
  },
});
