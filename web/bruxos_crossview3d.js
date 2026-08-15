import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "BruxosCrossViewWarp3D";
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const wrap = a => ((a + 180) % 360 + 360) % 360 - 180;

function widget(node, name) { return node.widgets?.find(w => w.name === name); }
function setWidget(node, name, value) {
  const w = widget(node, name); if (!w) return;
  w.value = value; w.callback?.(value); node.setDirtyCanvas(true, true);
}
function getWidget(node, name, fallback=0) { return widget(node, name)?.value ?? fallback; }
function parseKeyframes(node) {
  try { const v = JSON.parse(String(getWidget(node, "keyframes", "") || "[]")); return Array.isArray(v) ? v : []; }
  catch { return []; }
}
function saveKeyframes(node, kfs) {
  kfs.sort((a,b)=>a.f-b.f);
  setWidget(node, "keyframes", JSON.stringify(kfs.map(k=>({f:Math.round(k.f),az:+k.az,el:+k.el,dist:+k.dist}))));
}
function ease(t, mode) {
  if (mode === "ease_in_out") return .5-.5*Math.cos(Math.PI*t);
  if (mode === "ease_in") return t*t;
  if (mode === "ease_out") return 1-(1-t)*(1-t);
  return t;
}
function sampleCamera(node, frame) {
  const k = parseKeyframes(node);
  if (!getWidget(node,"use_keyframes",false) || !k.length) return null;
  if (frame <= k[0].f) return k[0]; if (frame >= k[k.length-1].f) return k[k.length-1];
  let i=0; while (i<k.length-1 && frame>k[i+1].f) i++;
  const a=k[i], b=k[i+1], mode=getWidget(node,"interp_motion","linear");
  const u=ease((frame-a.f)/Math.max(1,b.f-a.f),mode);
  let da=wrap(b.az-a.az);
  return {f:frame, az:wrap(a.az+da*u), el:a.el+(b.el-a.el)*u, dist:a.dist+(b.dist-a.dist)*u};
}

async function imageData(url) {
  const img = new Image(); img.crossOrigin="anonymous";
  await new Promise((ok, bad)=>{img.onload=ok; img.onerror=bad; img.src=url;});
  const c=document.createElement("canvas"); c.width=img.naturalWidth; c.height=img.naturalHeight;
  const x=c.getContext("2d",{willReadFrequently:true}); x.drawImage(img,0,0);
  return x.getImageData(0,0,c.width,c.height);
}
function viewUrl(desc) {
  const q=new URLSearchParams({filename:desc.filename,subfolder:desc.subfolder||"",type:desc.type||"temp"});
  return api.apiURL(`/view?${q}`);
}

function parsePly(buffer) {
  const bytes=new Uint8Array(buffer); const marker=new TextEncoder().encode("end_header\n");
  let end=-1;
  outer: for(let i=0;i<=bytes.length-marker.length;i++){for(let j=0;j<marker.length;j++)if(bytes[i+j]!==marker[j])continue outer;end=i+marker.length;break;}
  if(end<0) throw new Error("PLY sem end_header");
  const header=new TextDecoder().decode(bytes.slice(0,end));
  const count=+(header.match(/element vertex\s+(\d+)/)?.[1]||0);
  const binary=/format binary_little_endian/.test(header);
  const props=[...header.matchAll(/property\s+(\w+)\s+(\w+)/g)].map(m=>({type:m[1],name:m[2]}));
  const xyz=[], rgb=[], scales=[], rots=[], opacity=[];
  const pushVertex=(v)=>{
    xyz.push(v.x||0,v.y||0,v.z||0);
    rgb.push(
      v.red!=null?v.red/255:clamp(.5+0.2820947918*(v.f_dc_0||0),0,1),
      v.green!=null?v.green/255:clamp(.5+0.2820947918*(v.f_dc_1||0),0,1),
      v.blue!=null?v.blue/255:clamp(.5+0.2820947918*(v.f_dc_2||0),0,1)
    );
    const sx=v.scale_0!=null?Math.exp(v.scale_0):.01;
    const sy=v.scale_1!=null?Math.exp(v.scale_1):sx;
    const sz=v.scale_2!=null?Math.exp(v.scale_2):sx;
    scales.push(sx,sy,sz);
    let qw=v.rot_0??1,qx=v.rot_1??0,qy=v.rot_2??0,qz=v.rot_3??0;
    const qn=Math.hypot(qw,qx,qy,qz)||1; rots.push(qw/qn,qx/qn,qy/qn,qz/qn);
    const op=v.opacity!=null?1/(1+Math.exp(-v.opacity)):1; opacity.push(clamp(op,0.02,1));
  };
  if(!binary){
    const lines=new TextDecoder().decode(bytes.slice(end)).trim().split(/\r?\n/);
    for(let n=0;n<Math.min(count,lines.length);n++){
      const a=lines[n].trim().split(/\s+/).map(Number),v={};
      for(let i=0;i<props.length;i++)v[props[i].name]=a[i]; pushVertex(v);
    }
  } else {
    const size={char:1,uchar:1,int8:1,uint8:1,short:2,ushort:2,int16:2,uint16:2,int:4,uint:4,int32:4,uint32:4,float:4,float32:4,double:8,float64:8};
    const read=(dv,o,t)=>{switch(t){case"char":case"int8":return dv.getInt8(o);case"uchar":case"uint8":return dv.getUint8(o);case"short":case"int16":return dv.getInt16(o,true);case"ushort":case"uint16":return dv.getUint16(o,true);case"int":case"int32":return dv.getInt32(o,true);case"uint":case"uint32":return dv.getUint32(o,true);case"double":case"float64":return dv.getFloat64(o,true);default:return dv.getFloat32(o,true)}};
    const stride=props.reduce((a,p)=>a+(size[p.type]||4),0), dv=new DataView(buffer,end); let off=0;
    for(let n=0;n<count && off+stride<=dv.byteLength;n++){
      const v={}; let q=off; for(const pr of props){v[pr.name]=read(dv,q,pr.type);q+=size[pr.type]||4;} off+=stride; pushVertex(v);
    }
  }
  return {xyz:new Float32Array(xyz),rgb:new Float32Array(rgb),scale3:new Float32Array(scales),rot:new Float32Array(rots),opacity:new Float32Array(opacity)};
}
function parseSplat(buffer) {
  const dv=new DataView(buffer), n=Math.floor(buffer.byteLength/32), xyz=new Float32Array(n*3), rgb=new Float32Array(n*3), scale3=new Float32Array(n*3), rot=new Float32Array(n*4), opacity=new Float32Array(n);
  for(let i=0;i<n;i++){
    const o=i*32; xyz[i*3]=dv.getFloat32(o,true);xyz[i*3+1]=dv.getFloat32(o+4,true);xyz[i*3+2]=dv.getFloat32(o+8,true);
    scale3[i*3]=Math.max(1e-6,dv.getFloat32(o+12,true));scale3[i*3+1]=Math.max(1e-6,dv.getFloat32(o+16,true));scale3[i*3+2]=Math.max(1e-6,dv.getFloat32(o+20,true));
    rgb[i*3]=dv.getUint8(o+24)/255;rgb[i*3+1]=dv.getUint8(o+25)/255;rgb[i*3+2]=dv.getUint8(o+26)/255;opacity[i]=clamp(dv.getUint8(o+27)/255,.02,1);
    let qw=(dv.getUint8(o+28)-128)/128,qx=(dv.getUint8(o+29)-128)/128,qy=(dv.getUint8(o+30)-128)/128,qz=(dv.getUint8(o+31)-128)/128;
    const qn=Math.hypot(qw,qx,qy,qz)||1;rot[i*4]=qw/qn;rot[i*4+1]=qx/qn;rot[i*4+2]=qy/qn;rot[i*4+3]=qz/qn;
  }
  return {xyz,rgb,scale3,rot,opacity};
}
function quatRotate(qw,qx,qy,qz,x,y,z){
  const tx=2*(qy*z-qz*y),ty=2*(qz*x-qx*z),tz=2*(qx*y-qy*x);
  return [x+qw*tx+(qy*tz-qz*ty),y+qw*ty+(qz*tx-qx*tz),z+qw*tz+(qx*ty-qy*tx)];
}

class Viewer {
  constructor(node, root) {
    this.node=node; this.root=root; this.points=null; this.depthFrames=[]; this.current=1; this.playing=false; this.last=0;
    root.innerHTML=`<div class="bxbar"><button data-a="import">Importar Gaussian</button><button data-a="reset">Resetar câmera</button><span class="bxhint">Botão esquerdo: orbitar · Direito/Shift: mover · Roda: zoom</span></div><canvas></canvas><div class="bxtime"><button data-a="play" title="Reproduz a animação da câmera pela timeline.">▶</button><button data-a="key" title="Cria ou atualiza um keyframe no frame atual.">◆ Keyframe</button><button data-a="del" title="Remove o keyframe do frame atual.">× Keyframe</button><input type="range" min="1" max="1" value="1"><span>1 / 1</span></div><input data-file type="file" accept=".ply,.splat,.ksplat" hidden>`;
    this.canvas=root.querySelector("canvas"); this.gl=this.canvas.getContext("webgl2",{alpha:false,antialias:false,premultipliedAlpha:false}); this.ctx=this.gl?null:this.canvas.getContext("2d"); this.gpu=null; this.gpuPointsRef=null; this.interacting=false; this.range=root.querySelector("input[type=range]"); this.label=root.querySelector(".bxtime span");
    root.querySelector('[data-a=import]').onclick=()=>root.querySelector('[data-file]').click();
    root.querySelector('[data-file]').onchange=e=>this.importFile(e.target.files?.[0]);
    root.querySelector('[data-a=reset]').onclick=()=>{setWidget(node,"azimuth",-30);setWidget(node,"elevation",20);setWidget(node,"distance",1);setWidget(node,"pivot_x",0);setWidget(node,"pivot_y",0);this.draw();};
    root.querySelector('[data-a=play]').onclick=e=>{this.playing=!this.playing;e.currentTarget.textContent=this.playing?"⏸":"▶";this.last=performance.now();requestAnimationFrame(t=>this.tick(t));};
    root.querySelector('[data-a=key]').onclick=()=>{const k=parseKeyframes(node).filter(x=>x.f!==this.current);k.push({f:this.current,az:+getWidget(node,"azimuth",0),el:+getWidget(node,"elevation",0),dist:+getWidget(node,"distance",1)});saveKeyframes(node,k);setWidget(node,"use_keyframes",true);this.draw();};
    root.querySelector('[data-a=del]').onclick=()=>{saveKeyframes(node,parseKeyframes(node).filter(x=>x.f!==this.current));this.draw();};
    this.range.oninput=()=>this.setFrame(+this.range.value);
    this.syncTimelineLength();
    this.bindMouse(); new ResizeObserver(()=>{this.syncTimelineLength();this.draw();}).observe(root);
  }
  bindMouse(){let drag=false,mode="orbit",lx=0,ly=0;const c=this.canvas;
    c.oncontextmenu=e=>e.preventDefault(); c.onpointerdown=e=>{drag=true;this.interacting=true;mode=(e.button===2||e.shiftKey)?"pan":"orbit";lx=e.clientX;ly=e.clientY;c.setPointerCapture(e.pointerId)};
    c.onpointermove=e=>{if(!drag)return;const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;if(mode==="orbit"){setWidget(this.node,"azimuth",wrap(+getWidget(this.node,"azimuth",0)-dx*.35));setWidget(this.node,"elevation",clamp(+getWidget(this.node,"elevation",0)-dy*.35,-90,90));}else{const d=+getWidget(this.node,"distance",1);setWidget(this.node,"pivot_x",+getWidget(this.node,"pivot_x",0)-dx*.002*d);setWidget(this.node,"pivot_y",+getWidget(this.node,"pivot_y",0)+dy*.002*d);}this.draw()};
    c.onpointerup=()=>{drag=false;this.interacting=false;this.draw()};c.onwheel=e=>{e.preventDefault();const d=+getWidget(this.node,"distance",1);setWidget(this.node,"distance",clamp(d*Math.exp(e.deltaY*.0015),.05,12));this.draw()};
  }
  timelineLength(){
    const explicit = Math.round(+getWidget(this.node,"frame_count",0) || 0);
    const batch = Math.round(+this.loadedFrameCount || 0);
    return Math.max(1, explicit || batch || 1);
  }
  syncTimelineLength(){
    const total=this.timelineLength();
    this.range.max=String(total);
    if(this.current>total)this.current=total;
    this.range.value=String(this.current);
    this.label.textContent=`${this.current} / ${total}`;
  }
  setFrame(f){this.syncTimelineLength();this.current=clamp(Math.round(f),1,+this.range.max);this.range.value=this.current;this.label.textContent=`${this.current} / ${this.range.max}`;const cam=sampleCamera(this.node,this.current);if(cam){setWidget(this.node,"azimuth",cam.az);setWidget(this.node,"elevation",cam.el);setWidget(this.node,"distance",cam.dist);}this.draw();}
  tick(t){if(!this.playing)return;const fps=+getWidget(this.node,"play_fps",24);if(t-this.last>=1000/fps){this.last=t;let n=this.current+1;if(n>+this.range.max){if(getWidget(this.node,"loop_playback",true))n=1;else{this.playing=false;this.root.querySelector('[data-a=play]').textContent="▶";return}}this.setFrame(n)}requestAnimationFrame(x=>this.tick(x));}
  async importFile(file){if(!file)return;try{const local=await file.arrayBuffer();this.points=file.name.toLowerCase().endsWith(".ply")?parsePly(local):file.name.toLowerCase().endsWith(".splat")?parseSplat(local):null;if(!this.points)throw new Error(".ksplat ainda nao tem parser local; converta para .ply ou .splat");const fd=new FormData();fd.append("file",file);const r=await fetch(api.apiURL("/bruxos/upload_gaussian"),{method:"POST",body:fd});const j=await r.json();if(!r.ok)throw new Error(j.error||"upload falhou");setWidget(this.node,"gaussian_file",j.filename);setWidget(this.node,"source_mode","import_gaussian");this.syncTimelineLength();this.setFrame(1);this.fitPoints();}catch(e){alert(`Bruxos 3D: ${e.message}`)}}
  async loadExecuted(data){try{const m=JSON.parse(data.bx_cv3d_meta?.[0]||"{}");this.loadedFrameCount=Math.max(1,m.frame_count||m.B||1);this.syncTimelineLength();if(data.bx_cv3d_import?.[0]){const f=data.bx_cv3d_import[0];const b=await (await fetch(api.apiURL(`/bruxos/gaussian?filename=${encodeURIComponent(f)}`))).arrayBuffer();this.points=f.toLowerCase().endsWith(".ply")?parsePly(b):f.toLowerCase().endsWith(".splat")?parseSplat(b):null;this.fitPoints();return;}const rgbs=data.bx_cv3d_frames||[],deps=data.bx_cv3d_depth||[];this.depthFrames=[];for(let i=0;i<Math.min(rgbs.length,deps.length);i++){const [r,d]=await Promise.all([imageData(viewUrl(rgbs[i])),imageData(viewUrl(deps[i]))]);this.depthFrames.push(this.makePoints(r,d,m));}this.points=this.depthFrames[0]||null;this.fitPoints();}catch(e){console.error("Bruxos viewer",e)}}
  makePoints(rgb,dep,m){const step=Math.max(1,Math.ceil(Math.sqrt((rgb.width*rgb.height)/120000))), xyz=[],col=[],sc=[];const fx=(m.fx||rgb.width)*rgb.width/(m.W||rgb.width),zlo=m.z_lo||0,zhi=m.z_hi||1;for(let y=0;y<rgb.height;y+=step)for(let x=0;x<rgb.width;x+=step){const i=(y*rgb.width+x)*4,z16=dep.data[i]*256+dep.data[i+1],z=zlo+(zhi-zlo)*z16/65535;xyz.push((x-rgb.width/2)/fx*z,(y-rgb.height/2)/fx*z,z);col.push(rgb.data[i]/255,rgb.data[i+1]/255,rgb.data[i+2]/255);sc.push(z/fx*step)}const scale3=[];for(const v of sc)scale3.push(v,v,v);const rot=new Float32Array((xyz.length/3)*4);const opacity=new Float32Array(xyz.length/3);for(let i=0;i<opacity.length;i++){rot[i*4]=1;opacity[i]=1}return{xyz:new Float32Array(xyz),rgb:new Float32Array(col),scale3:new Float32Array(scale3),rot,opacity}}
  fitPoints(){if(!this.points?.xyz?.length){this.draw();return}let sx=0,sy=0,sz=0,n=this.points.xyz.length/3;for(let i=0;i<n;i++){sx+=this.points.xyz[i*3];sy+=this.points.xyz[i*3+1];sz+=this.points.xyz[i*3+2]}setWidget(this.node,"pivot_x",sx/n);setWidget(this.node,"pivot_y",sy/n);setWidget(this.node,"pivot_z",sz/n);this.draw();}
  initGPU(){
    const gl=this.gl;if(!gl||this.gpu)return;
    const vs=`#version 300 es
    precision highp float;
    in vec3 aPos; in vec3 aColor; in float aScale; in float aOpacity;
    uniform mat4 uViewProj; uniform float uPointMul; uniform float uMode;
    out vec3 vColor; out float vOpacity;
    void main(){vec4 cp=uViewProj*vec4(aPos,1.0);gl_Position=cp;float z=max(0.02,abs(cp.w));gl_PointSize=clamp(aScale*uPointMul/z,1.0,64.0);vColor=aColor;vOpacity=aOpacity;}`;
    const fs=`#version 300 es
    precision highp float; in vec3 vColor; in float vOpacity; uniform float uMode; out vec4 outColor;
    void main(){vec2 p=gl_PointCoord*2.0-1.0;float r2=dot(p,p);if(r2>1.0)discard;float a=uMode<0.5?1.0:exp(-r2*3.8)*vOpacity;outColor=vec4(vColor,a);}`;
    const sh=(type,src)=>{const x=gl.createShader(type);gl.shaderSource(x,src);gl.compileShader(x);if(!gl.getShaderParameter(x,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(x));return x};
    const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pr);if(!gl.getProgramParameter(pr,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(pr));
    this.gpu={pr,bufs:[gl.createBuffer(),gl.createBuffer(),gl.createBuffer(),gl.createBuffer()],count:0,loc:{vp:gl.getUniformLocation(pr,'uViewProj'),pm:gl.getUniformLocation(pr,'uPointMul'),mode:gl.getUniformLocation(pr,'uMode')}};
  }
  uploadGPU(){const gl=this.gl,p=this.points;if(!gl||!p)return;this.initGPU();const g=this.gpu,n=p.xyz.length/3;let scale=new Float32Array(n);for(let i=0;i<n;i++){const sx=p.scale3?.[i*3]??.01,sy=p.scale3?.[i*3+1]??sx,sz=p.scale3?.[i*3+2]??sx;scale[i]=Math.max(sx,sy,sz)}const attrs=[p.xyz,p.rgb,scale,p.opacity||new Float32Array(n).fill(1)],sizes=[3,3,1,1],names=['aPos','aColor','aScale','aOpacity'];gl.useProgram(g.pr);for(let i=0;i<4;i++){gl.bindBuffer(gl.ARRAY_BUFFER,g.bufs[i]);gl.bufferData(gl.ARRAY_BUFFER,attrs[i],gl.STATIC_DRAW);const loc=gl.getAttribLocation(g.pr,names[i]);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,sizes[i],gl.FLOAT,false,0,0)}g.count=n;this.gpuPointsRef=p;}
  mat4(){const az=+getWidget(this.node,'azimuth',0)*Math.PI/180,el=+getWidget(this.node,'elevation',0)*Math.PI/180,dist=+getWidget(this.node,'distance',1),px=+getWidget(this.node,'pivot_x',0),py=+getWidget(this.node,'pivot_y',0),pz=+getWidget(this.node,'pivot_z',1);const eye=[px+dist*Math.cos(el)*Math.sin(az),py-dist*Math.sin(el),pz+dist*Math.cos(el)*Math.cos(az)],up=[0,-1,0];const sub=(a,b)=>a.map((v,i)=>v-b[i]),norm=a=>{const l=Math.hypot(...a)||1;return a.map(v=>v/l)},cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];const f=norm(sub([px,py,pz],eye)),r=norm(cross(up,f)),u=cross(f,r);const view=new Float32Array([r[0],u[0],f[0],0,r[1],u[1],f[1],0,r[2],u[2],f[2],0,-dot(r,eye),-dot(u,eye),-dot(f,eye),1]);const aspect=Math.max(.1,this.canvas.width/this.canvas.height),fov=(+getWidget(this.node,'hfov',50))*Math.PI/180,t=1/Math.tan(fov/2),near=.01,far=10000;const proj=new Float32Array([t/aspect,0,0,0,0,t,0,0,0,0,(far+near)/(far-near),1,0,0,(-2*far*near)/(far-near),0]);const out=new Float32Array(16);for(let c=0;c<4;c++)for(let rr=0;rr<4;rr++){let v=0;for(let k=0;k<4;k++)v+=proj[k*4+rr]*view[c*4+k];out[c*4+rr]=v}return out;}
  drawFallback(w,h,dpr){const x=this.ctx;x.setTransform(dpr,0,0,dpr,0,0);x.fillStyle='#0d0d11';x.fillRect(0,0,w,h);x.fillStyle='#aaa';x.font='14px sans-serif';x.fillText('WebGL2 indisponível. Execute ou importe um Gaussian.',18,h/2);}
  draw(){const c=this.canvas,rect=c.getBoundingClientRect(),dpr=devicePixelRatio||1;const q=String(getWidget(this.node,'preview_quality','equilibrado'));const scale=this.interacting?.5:(q==='leve'?.55:q==='completo'?1:.78),w=Math.max(10,Math.floor(rect.width*scale)),h=Math.max(180,Math.floor(rect.height*scale));if(c.width!==Math.floor(w*dpr)||c.height!==Math.floor(h*dpr)){c.width=Math.floor(w*dpr);c.height=Math.floor(h*dpr)}if(!this.gl){this.drawFallback(rect.width,rect.height,dpr);return}const gl=this.gl;gl.viewport(0,0,c.width,c.height);gl.clearColor(.05,.05,.07,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);const p=this.points;if(!p)return;if(this.gpuPointsRef!==p)this.uploadGPU();const g=this.gpu;gl.useProgram(g.pr);gl.uniformMatrix4fv(g.loc.vp,false,this.mat4());gl.uniform1f(g.loc.pm,(+getWidget(this.node,'point_size',1.6))*900*(this.interacting?.7:1));gl.uniform1f(g.loc.mode,getWidget(this.node,'render_mode','gaussian')==='gaussian'?1:0);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.disable(gl.DEPTH_TEST);let max=q==='leve'?250000:q==='completo'?2000000:750000;let count=Math.min(g.count,max);gl.drawArrays(gl.POINTS,0,count);}
  }}

app.registerExtension({
  name:"Bruxos.CrossView3D.Navigation",
  beforeRegisterNodeDef(nodeType,nodeData){
    if(nodeData.name!==NODE_NAME)return;
    const orig=nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated=function(){
      orig?.apply(this,arguments);

      const VIEW_H = 620;
      this.size[0] = Math.max(this.size[0], 520);

      const root = document.createElement("div");
      root.className = "bx3droot";
      root.style.width = "100%";
      root.style.height = `${VIEW_H}px`;
      root.style.minHeight = `${VIEW_H}px`;
      root.style.boxSizing = "border-box";

      // CSS fica no document.head. Colocar <style> dentro do widget fazia o Node 2.0
      // calcular uma altura errada e recortar o canvas/timeline.
      if (!document.getElementById("bx3droot-style")) {
        const style = document.createElement("style");
        style.id = "bx3droot-style";
        style.textContent = `
          .bx3droot{display:grid;grid-template-rows:auto minmax(430px,1fr) 52px;width:100%;height:620px;min-height:620px;background:#111;border:1px solid #3b3d43;border-radius:9px;overflow:hidden;color:#ddd;font:12px sans-serif;box-sizing:border-box}
          .bx3droot canvas{display:block;width:100%;height:100%;min-height:430px;touch-action:none;background:#0d0d11}
          .bx3droot .bxbar,.bx3droot .bxtime{display:flex;align-items:center;gap:7px;min-width:0;padding:7px 8px;background:#202124;box-sizing:border-box}
          .bx3droot .bxbar{border-bottom:1px solid #34363c;flex-wrap:wrap;min-height:48px}
          .bx3droot .bxtime{border-top:1px solid #34363c}
          .bx3droot button{flex:0 0 auto;background:#303238;color:#eee;border:1px solid #4b4e56;border-radius:5px;padding:6px 10px;line-height:1.1;cursor:pointer}
          .bx3droot button:hover{background:#3a3d44}
          .bx3droot .bxhint{opacity:.72;flex:1 1 190px;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
          .bx3droot .bxtime input[type=range]{flex:1 1 auto;min-width:100px}
          .bx3droot .bxtime span{flex:0 0 auto;min-width:54px;text-align:right;white-space:nowrap}
        `;
        document.head.appendChild(style);
      }

      const v = new Viewer(this, root);
      this.__bxViewer = v;

      // Nodes 2.0 nem sempre reexecuta o DOM widget quando frame_count muda.
      // Mantemos a timeline sincronizada diretamente com o widget.
      for (const name of ["frame_count", "use_keyframes"]) {
        const w = widget(this, name);
        if (!w || w.__bxWrapped) continue;
        const oldCb = w.callback;
        w.callback = (...args) => { oldCb?.apply(w, args); v.syncTimelineLength(); v.draw(); };
        w.__bxWrapped = true;
      }

      const domWidget = this.addDOMWidget("bruxos_viewport", "div", root, {
        serialize: false,
        hideOnZoom: false,
      });

      // Essencial no ComfyUI Nodes 2.0: reserva a altura real do DOM widget.
      domWidget.computeSize = function(width) {
        return [Math.max(360, width || 520), VIEW_H + 8];
      };
      domWidget.getHeight = function() {
        return VIEW_H + 8;
      };

      requestAnimationFrame(() => {
        this.setSize([Math.max(this.size[0], 520), Math.max(this.size[1], 1220)]);
        this.setDirtyCanvas(true, true);
        v.draw();
      });
    };
    const ex=nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted=function(message){ex?.apply(this,arguments);this.__bxViewer?.loadExecuted(message||{});};
  }
});
