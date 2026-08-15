import { app } from "../../scripts/app.js";

const COLORS = ["#22c55e", "#38bdf8", "#a78bfa", "#fb7185", "#facc15", "#2dd4bf", "#fb923c", "#e879f9"];
const MIN_SIZE = 0.035;
const EDITORS = new Map();
let editorSequence = 0;

// The current ComfyUI frontend captures graph gestures before DOM-widget
// bubbling handlers in some views. Delegate toolbar activation from window's
// capture phase; this also keeps the controls working when the node is mirrored
// in the properties panel.
window.addEventListener("click", (event) => {
  const button = event.target?.closest?.("[data-bruxos-tile-editor] button[data-a], [data-bruxos-tile-editor] button[data-p]");
  if (!button) return;
  const host = button.closest("[data-bruxos-tile-editor]");
  const editor = EDITORS.get(host?.dataset?.bruxosTileEditor);
  if (!editor) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  editor.toolbarAction(button);
}, true);

const PRESETS = {
  grid2: [
    [0, 0, .56, .56], [.44, 0, 1, .56],
    [0, .44, .56, 1], [.44, .44, 1, 1],
  ],
  center5: [
    [0, 0, .56, .56], [.44, 0, 1, .56],
    [0, .44, .56, 1], [.44, .44, 1, 1],
    [.25, .25, .75, .75],
  ],
  asymmetric5: [
    [0, 0, .50, .56], [0, .44, .50, 1],
    [.42, 0, 1, .40], [.42, .30, 1, .70], [.42, .60, 1, 1],
  ],
};

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function makeTiles(rects) {
  return rects.map((r, i) => ({id: i + 1, x0: r[0], y0: r[1], x1: r[2], y1: r[3], weight: 1}));
}

function clamp(v, lo = 0, hi = 1) { return Math.max(lo, Math.min(hi, v)); }
function snap(v, event) { return event?.altKey ? v : Math.round(v * 200) / 200; }

function sanitize(raw) {
  const arr = Array.isArray(raw) ? raw : raw?.tiles;
  if (!Array.isArray(arr) || !arr.length) return makeTiles(PRESETS.center5);
  return arr.slice(0, 24).map((t, i) => {
    let x0 = clamp(Number(t.x0) || 0), y0 = clamp(Number(t.y0) || 0);
    let x1 = clamp(Number(t.x1) || 1), y1 = clamp(Number(t.y1) || 1);
    if (x1 < x0) [x0, x1] = [x1, x0];
    if (y1 < y0) [y0, y1] = [y1, y0];
    x1 = Math.max(x1, x0 + MIN_SIZE); y1 = Math.max(y1, y0 + MIN_SIZE);
    return {id: Number(t.id) || i + 1, x0, y0, x1: clamp(x1), y1: clamp(y1), weight: clamp(Number(t.weight) || 1, .05, 8)};
  });
}

function coverage(tiles, n = 80) {
  let covered = 0;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const px = (x + .5) / n, py = (y + .5) / n;
    if (tiles.some((t) => t.x0 <= px && px <= t.x1 && t.y0 <= py && py <= t.y1)) covered++;
  }
  return covered / (n * n);
}

function buildEditor(node) {
  if (node._bruxosTileEditor) return;
  const jsonW = widget(node, "layout_json");
  if (jsonW) {
    jsonW.computeSize = () => [0, -4];
    if (jsonW.inputEl) jsonW.inputEl.style.display = "none";
  }

  const root = document.createElement("div");
  const editorId = `bruxos-tile-${++editorSequence}`;
  root.dataset.bruxosTileEditor = editorId;
  root.style.cssText = "display:flex;flex-direction:column;gap:6px;width:100%;box-sizing:border-box;padding:2px;color:#ddd;font:11px sans-serif;";
  const bar = document.createElement("div");
  bar.style.cssText = "display:flex;gap:5px;align-items:center;flex-wrap:wrap;";
  bar.innerHTML = `
    <button data-a="add">＋ Tile</button><button data-a="dup">⧉ Duplicar</button><button data-a="del">× Excluir</button>
    <span style="width:1px;height:20px;background:#444"></span>
    <button data-p="grid2">2×2</button><button data-p="center5">Centro 5</button><button data-p="asymmetric5">Assimétrico 5</button>
  `;
  for (const b of bar.querySelectorAll("button")) b.style.cssText = "height:25px;padding:2px 7px;border:1px solid #50505a;border-radius:5px;background:#202027;color:#ddd;cursor:pointer;";

  const canvas = document.createElement("canvas");
  canvas.tabIndex = 0;
  canvas.style.cssText = "display:block;width:100%;background:#0c0d11;border:1px solid #454550;border-radius:7px;touch-action:none;outline:none;cursor:default;";
  const status = document.createElement("div");
  status.style.cssText = "display:flex;justify-content:space-between;gap:8px;color:#aaa;font:10px ui-monospace,monospace;";
  const hint = document.createElement("div");
  hint.textContent = "Arraste o interior para mover · cantos para redimensionar · duplo clique cria tile · Alt = sem snap";
  hint.style.cssText = "color:#888;font-size:10px;";
  root.append(bar, canvas, status, hint);

  // ComfyUI's graph canvas listens for pointer events above DOM widgets.  Keep
  // editor gestures inside this widget so toolbar clicks and canvas drags are
  // not interpreted as graph selection/drag gestures.
  for (const eventName of ["pointerdown", "pointerup", "click", "dblclick", "keydown"])
    root.addEventListener(eventName, (e) => e.stopPropagation());

  const dom = node.addDOMWidget("bruxos_tile_canvas", "layout", root, {serialize: false, hideOnZoom: false});
  dom.serialize = false;
  dom.serializeValue = () => undefined;
  dom.computeSize = (width) => {
    const W = Number(widget(node, "canvas_width")?.value) || 832;
    const H = Number(widget(node, "canvas_height")?.value) || 480;
    const h = Math.max(230, Math.min(390, ((node.size?.[0] || width || 580) - 34) * H / W));
    return [width, h + 78];
  };

  let tiles = makeTiles(PRESETS.center5), selected = 4, drag = null;
  const ctx = canvas.getContext("2d");

  function resize() {
    const W = Number(widget(node, "canvas_width")?.value) || 832;
    const H = Number(widget(node, "canvas_height")?.value) || 480;
    const cssW = Math.max(300, (node.size?.[0] || 600) - 32);
    const cssH = Math.max(210, Math.min(360, cssW * H / W));
    canvas.style.height = `${cssH}px`;
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(cssW * dpr), bh = Math.round(cssH * dpr);
    if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
    draw();
  }

  function pos(e) {
    const r = canvas.getBoundingClientRect();
    return {x: clamp((e.clientX - r.left) / r.width), y: clamp((e.clientY - r.top) / r.height)};
  }

  function handles(t) {
    return {nw:[t.x0,t.y0], ne:[t.x1,t.y0], sw:[t.x0,t.y1], se:[t.x1,t.y1]};
  }

  function hit(p) {
    const radius = 11 / Math.max(200, canvas.getBoundingClientRect().width);
    for (let i = tiles.length - 1; i >= 0; i--) {
      const t = tiles[i];
      for (const [name, h] of Object.entries(handles(t))) if (Math.hypot(p.x-h[0], p.y-h[1]) <= radius) return {i, mode:name};
      if (t.x0 <= p.x && p.x <= t.x1 && t.y0 <= p.y && p.y <= t.y1) return {i, mode:"move"};
    }
    return null;
  }

  function draw() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0c0d11"; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = "#292b34"; ctx.lineWidth = 1;
    for (let i=1;i<10;i++) { ctx.beginPath(); ctx.moveTo(W*i/10,0); ctx.lineTo(W*i/10,H); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0,H*i/10); ctx.lineTo(W,H*i/10); ctx.stroke(); }
    tiles.forEach((t, i) => {
      const x=t.x0*W,y=t.y0*H,w=(t.x1-t.x0)*W,h=(t.y1-t.y0)*H,c=COLORS[i%COLORS.length];
      ctx.globalAlpha = i===selected ? .30 : .18; ctx.fillStyle=c; ctx.fillRect(x,y,w,h); ctx.globalAlpha=1;
      ctx.strokeStyle=c; ctx.lineWidth=i===selected?3:1.5; ctx.strokeRect(x,y,w,h);
      ctx.fillStyle="#fff"; ctx.font=`600 ${Math.max(11,Math.min(18,W/34))}px sans-serif`; ctx.fillText(String(i+1),x+8,y+20);
      if (i===selected) for (const hnd of Object.values(handles(t))) { ctx.fillStyle="#fff"; ctx.fillRect(hnd[0]*W-5,hnd[1]*H-5,10,10); ctx.strokeStyle=c; ctx.strokeRect(hnd[0]*W-5,hnd[1]*H-5,10,10); }
    });
    const cov=coverage(tiles), largest=Math.max(...tiles.map(t=>(t.x1-t.x0)*(t.y1-t.y0)));
    status.innerHTML=`<span>${tiles.length} tiles · maior ${(largest*100).toFixed(1)}%</span><span style="color:${cov>.999?'#4ade80':'#fb7185'}">cobertura ${(cov*100).toFixed(1)}%</span>`;
  }

  function save() {
    tiles.forEach((t,i)=>t.id=i+1);
    if (jsonW) jsonW.value = JSON.stringify({version:1,tiles});
    node.graph?.setDirtyCanvas(true, true);
    draw();
  }

  function load() {
    try { tiles = sanitize(JSON.parse(String(jsonW?.value || ""))); } catch (_) { tiles = makeTiles(PRESETS.center5); }
    selected = Math.min(Math.max(0, selected), tiles.length-1); resize();
  }

  canvas.addEventListener("pointerdown", (e) => {
    const p=pos(e), h=hit(p); selected=h?.i ?? -1;
    if (h) { drag={mode:h.mode,start:p,original:{...tiles[h.i]}}; canvas.setPointerCapture(e.pointerId); }
    draw();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag || selected<0) { const h=hit(pos(e)); canvas.style.cursor=h?.mode==="move"?"move":h?`${h.mode}-resize`:"crosshair"; return; }
    const p=pos(e), o=drag.original, dx=p.x-drag.start.x, dy=p.y-drag.start.y, t=tiles[selected];
    if (drag.mode==="move") {
      const w=o.x1-o.x0,h=o.y1-o.y0; let x0=snap(clamp(o.x0+dx,0,1-w),e), y0=snap(clamp(o.y0+dy,0,1-h),e);
      t.x0=x0;t.y0=y0;t.x1=x0+w;t.y1=y0+h;
    } else {
      if (drag.mode.includes("w")) t.x0=clamp(snap(p.x,e),0,t.x1-MIN_SIZE);
      if (drag.mode.includes("e")) t.x1=clamp(snap(p.x,e),t.x0+MIN_SIZE,1);
      if (drag.mode.includes("n")) t.y0=clamp(snap(p.y,e),0,t.y1-MIN_SIZE);
      if (drag.mode.includes("s")) t.y1=clamp(snap(p.y,e),t.y0+MIN_SIZE,1);
    }
    draw();
  });
  const finish=()=>{if(drag){drag=null;save();}};
  canvas.addEventListener("pointerup",finish); canvas.addEventListener("pointercancel",finish);
  canvas.addEventListener("dblclick", (e) => {
    const p=pos(e), w=.32,h=.32; tiles.push({id:tiles.length+1,x0:clamp(p.x-w/2,0,1-w),y0:clamp(p.y-h/2,0,1-h),x1:0,y1:0,weight:1});
    const t=tiles.at(-1);t.x1=t.x0+w;t.y1=t.y0+h;selected=tiles.length-1;save();
  });
  canvas.addEventListener("keydown",(e)=>{if((e.key==="Delete"||e.key==="Backspace")&&selected>=0&&tiles.length>1){tiles.splice(selected,1);selected=Math.min(selected,tiles.length-1);save();e.preventDefault();}});

  function toolbarAction(b) {
    if(b.dataset.p){tiles=makeTiles(PRESETS[b.dataset.p]);selected=tiles.length-1;save();return;}
    if(b.dataset.a==="add"){tiles.push({id:tiles.length+1,x0:.34,y0:.34,x1:.66,y1:.66,weight:1});selected=tiles.length-1;save();}
    if(b.dataset.a==="dup"&&selected>=0){const o=tiles[selected],w=o.x1-o.x0,h=o.y1-o.y0,x0=clamp(o.x0+.04,0,1-w),y0=clamp(o.y0+.04,0,1-h);tiles.push({...o,id:tiles.length+1,x0,y0,x1:x0+w,y1:y0+h});selected=tiles.length-1;save();}
    if(b.dataset.a==="del"&&selected>=0&&tiles.length>1){tiles.splice(selected,1);selected=Math.min(selected,tiles.length-1);save();}
  }
  for (const name of ["canvas_width","canvas_height"]) { const w=widget(node,name); if(w){const old=w.callback;w.callback=function(){const r=old?.apply(this,arguments);resize();return r;};} }
  const oldDraw=node.onDrawForeground; node.onDrawForeground=function(){const r=oldDraw?.apply(this,arguments);resize();return r;};
  node._bruxosTileEditor={root,canvas,load,resize,toolbarAction};
  EDITORS.set(editorId, node._bruxosTileEditor);
  node.setSize([Math.max(590,node.size?.[0]||0),Math.max(500,node.size?.[1]||0)]);
  load();
}

app.registerExtension({
  name:"BruxosDoVFX.CustomTileLayout",
  async beforeRegisterNodeDef(nodeType,nodeData){
    if(nodeData?.name!=="BruxosTileLayoutCustom")return;
    const created=nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated=function(){const r=created?.apply(this,arguments);buildEditor(this);return r;};
    const configured=nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure=function(){const r=configured?.apply(this,arguments);setTimeout(()=>this._bruxosTileEditor?.load(),0);setTimeout(()=>this._bruxosTileEditor?.load(),80);return r;};
  }
});
