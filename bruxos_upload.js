<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Comparar Vídeos · Bruxos do VFX</title>
<style>
  :root{
    --roxo:#a855f7; --roxo2:#7c3aed; --verde:#22c55e; --verde2:#4ade80;
    --bg:#0a0a0c; --bg2:#101014; --linha:#1d1d24; --txt:#e7e7ea; --mut:#8a8a96;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--txt);
    font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Arial,sans-serif;overflow:hidden}
  button{font-family:inherit;cursor:pointer}
  .app{display:flex;flex-direction:column;height:100%}

  /* topo */
  header{display:flex;align-items:center;gap:12px;padding:10px 14px;
    border-bottom:1px solid var(--linha);background:linear-gradient(90deg,#0c0c10,#0a0a0c)}
  .logo{font-weight:800;letter-spacing:.3px}
  .logo b{color:var(--verde2)} .logo i{color:var(--roxo);font-style:normal}
  .sub{color:var(--mut);font-size:12px}
  .spacer{flex:1}
  .btn{background:#0e0e12;border:1px solid var(--linha);color:var(--txt);
    padding:8px 14px;border-radius:10px;font-weight:600;font-size:13px;transition:.15s}
  .btn:hover{border-color:var(--roxo)}
  .btn.load{border-color:var(--verde);color:var(--verde2)}
  .btn.load:hover{background:rgba(34,197,94,.12)}
  .btn.swap{border-color:var(--roxo);color:var(--roxo)}
  .modes{display:flex;gap:6px;margin-left:6px}
  .mode{background:#0e0e12;border:1px solid var(--linha);color:var(--mut);
    padding:7px 12px;border-radius:9px;font-weight:600;font-size:12px}
  .mode.on{border-color:var(--verde);color:var(--verde2);box-shadow:0 0 0 1px rgba(34,197,94,.25) inset}

  /* palco */
  .stage{position:relative;flex:1;overflow:hidden;background:#000}
  .layer{position:absolute;inset:0;width:100%;height:100%}
  video{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000}
  #vb{} /* clip controlado por JS no modo slider */
  .badge{position:absolute;top:12px;width:34px;height:34px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;font-weight:800;z-index:6;
    border:2px solid #000}
  #ba{left:12px;background:var(--verde);color:#04130a}
  #bb{right:12px;background:var(--roxo);color:#150522}
  .divider{position:absolute;top:0;bottom:0;width:2px;background:var(--verde2);
    z-index:5;pointer-events:none;box-shadow:0 0 10px rgba(74,222,128,.7)}
  .handle{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
    width:34px;height:34px;border-radius:50%;background:#0a0a0c;border:2px solid var(--verde2);
    display:flex;align-items:center;justify-content:center;color:var(--verde2);font-size:13px}

  .hint{position:absolute;inset:0;display:flex;flex-direction:column;gap:10px;
    align-items:center;justify-content:center;text-align:center;color:var(--mut);
    padding:24px;z-index:7;pointer-events:none}
  .hint h1{margin:0;color:var(--verde2);font-size:26px}
  .hint .ab{color:var(--txt);font-weight:700}
  .hint .ab b{color:var(--verde2)} .hint .ab i{color:var(--roxo);font-style:normal}
  .hint small{color:#6b6b76}

  /* rodape / controles */
  footer{display:flex;align-items:center;gap:10px;padding:9px 12px;
    border-top:1px solid var(--linha);background:#0b0b0f;flex-wrap:wrap}
  .ico{width:38px;height:34px;border-radius:9px;background:#0e0e12;border:1px solid var(--linha);
    color:var(--txt);display:flex;align-items:center;justify-content:center;font-size:14px}
  .ico:hover{border-color:var(--roxo)}
  .ico.on{border-color:var(--verde);color:var(--verde2)}
  .time{font-variant-numeric:tabular-nums;color:var(--verde2);font-weight:700;font-size:13px;
    margin-left:auto}
  .seek{flex:1 1 260px;min-width:200px;display:flex;align-items:center}
  input[type=range]{width:100%;accent-color:var(--roxo)}
  .small{font-size:12px;color:var(--mut)}
  .pill{padding:6px 10px;border-radius:9px;border:1px solid var(--linha);background:#0e0e12;
    color:var(--txt);font-weight:600;font-size:12px}
  .hidden{display:none!important}
  .priv{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);z-index:7;
    color:#5f5f6a;font-size:11px;pointer-events:none}
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="logo"><b>Comparar</b> <i>Vídeos</i></div>
    <div class="sub">A/B sincronizado · linha do tempo única</div>
    <div class="spacer"></div>
    <button class="btn load" id="loadA">Carregar Vídeo A</button>
    <button class="btn swap" id="swap">⇄ Trocar</button>
    <button class="btn load" id="loadB">Carregar Vídeo B</button>
    <div class="modes">
      <button class="mode on" data-mode="slider">Cortina</button>
      <button class="mode" data-mode="side">Lado a Lado</button>
      <button class="mode" data-mode="diff">Diferença</button>
      <button class="mode" data-mode="toggle">Alternar</button>
    </div>
  </header>

  <div class="stage" id="stage">
    <video id="va" class="layer" playsinline></video>
    <video id="vb" class="layer" playsinline></video>
    <div class="badge" id="ba">A</div>
    <div class="badge" id="bb">B</div>
    <div class="divider" id="divider"><div class="handle">⇆</div></div>

    <div class="hint" id="hint">
      <h1>Comparar Vídeos</h1>
      <div>Solte dois vídeos para reproduzir na mesma linha do tempo.<br>
        Ideal para comparar resultados de upscale / interpolação.</div>
      <div class="ab">Metade esquerda = <b>A</b> · Metade direita = <i>B</i></div>
      <div>Arraste e solte de cada lado, ou clique para escolher</div>
      <small>Atalhos: <b>?</b> ajuda · <b>Espaço</b> play · <b>←/→</b> avançar · roda do mouse = zoom</small>
      <small>Os arquivos abrem só no seu navegador e nunca são enviados a lugar nenhum.</small>
    </div>
    <div class="priv">🔒 100% local · nada é enviado para servidores</div>
  </div>

  <footer>
    <button class="ico" id="play" title="Play/Pause (Espaço)">▶</button>
    <button class="ico on" id="loop" title="Repetir">↻</button>
    <button class="ico" id="prev" title="Quadro anterior">⏮</button>
    <button class="ico" id="next" title="Próximo quadro">⏭</button>
    <button class="pill" id="speed" title="Velocidade">1.0x</button>
    <button class="ico" id="mute" title="Mudo">🔊</button>
    <button class="ico on" id="audioA" title="Ouvir A">🔊A</button>
    <button class="ico" id="audioB" title="Ouvir B">🔇B</button>
    <div class="seek"><input type="range" id="seek" min="0" max="1000" value="0"></div>
    <div class="time"><span id="cur">00:00.00</span> / <span id="dur">00:00.00</span></div>
    <button class="ico" id="full" title="Tela cheia">⛶</button>
  </footer>
</div>

<input type="file" id="fileA" accept="video/*" class="hidden">
<input type="file" id="fileB" accept="video/*" class="hidden">

<script>
(function(){
  const $ = (id)=>document.getElementById(id);
  const va=$('va'), vb=$('vb'), stage=$('stage'), divider=$('divider'),
        hint=$('hint'), seek=$('seek');
  let mode='slider', split=0.5, haveA=false, haveB=false, audio='A', muted=false;
  const speeds=[0.25,0.5,1,1.5,2], spIdx=2; let speedIdx=spIdx;

  // ---- carregar arquivos ----
  function setSrc(v, file){ const url=URL.createObjectURL(file); v.src=url; v.load(); }
  $('loadA').onclick=()=>$('fileA').click();
  $('loadB').onclick=()=>$('fileB').click();
  $('fileA').onchange=e=>{ if(e.target.files[0]){ setSrc(va,e.target.files[0]); haveA=true; ready(); } };
  $('fileB').onchange=e=>{ if(e.target.files[0]){ setSrc(vb,e.target.files[0]); haveB=true; ready(); } };

  // drag & drop por lado
  stage.addEventListener('dragover',e=>{e.preventDefault();});
  stage.addEventListener('drop',e=>{
    e.preventDefault();
    const f=e.dataTransfer.files[0]; if(!f) return;
    const left = e.offsetX < stage.clientWidth/2;
    if(left){ setSrc(va,f); haveA=true; } else { setSrc(vb,f); haveB=true; }
    ready();
  });
  // clique em cada lado quando vazio
  stage.addEventListener('click',e=>{
    if(haveA&&haveB) return;
    const left = e.offsetX < stage.clientWidth/2;
    if(left&&!haveA) $('fileA').click(); else if(!left&&!haveB) $('fileB').click();
  });

  function ready(){
    if(haveA||haveB) hint.classList.add('hidden');
    applyMode();
    updateDur();
  }

  // ---- modos ----
  document.querySelectorAll('.mode').forEach(b=>{
    b.onclick=()=>{ document.querySelectorAll('.mode').forEach(x=>x.classList.remove('on'));
      b.classList.add('on'); mode=b.dataset.mode; applyMode(); };
  });
  function applyMode(){
    va.style.clipPath=''; vb.style.clipPath=''; vb.style.mixBlendMode='';
    va.style.width='100%'; vb.style.width='100%'; va.style.left='0'; vb.style.left='0';
    divider.style.display='none'; vb.style.opacity='1'; va.style.opacity='1';
    if(mode==='slider'){
      divider.style.display='block';
      setSplit(split);
    } else if(mode==='side'){
      va.style.width='50%'; va.style.left='0';
      vb.style.width='50%'; vb.style.left='50%';
    } else if(mode==='diff'){
      vb.style.mixBlendMode='difference';
    } else if(mode==='toggle'){
      setToggle(showB);
    }
  }
  // cortina
  function setSplit(x){
    split=Math.max(0,Math.min(1,x));
    const px=split*stage.clientWidth;
    vb.style.clipPath=`inset(0 0 0 ${px}px)`;       // B aparece à direita
    divider.style.left=px+'px';
  }
  stage.addEventListener('mousemove',e=>{ if(mode==='slider'&&(haveA||haveB)) setSplit(e.offsetX/stage.clientWidth); });

  // alternar
  let showB=false;
  function setToggle(b){ showB=b; vb.style.opacity=b?'1':'0'; }
  stage.addEventListener('click',()=>{ if(mode==='toggle'&&haveA&&haveB) setToggle(!showB); });

  // swap
  $('swap').onclick=()=>{
    const t=va.src; va.src=vb.src; vb.src=t;
    const h=haveA; haveA=haveB; haveB=h; ready();
  };

  // ---- playback sincronizado ----
  const longer=()=> (vb.duration||0) > (va.duration||0) ? vb : va;
  function playing(){ return !va.paused || !vb.paused; }
  function playPause(){
    if(playing()){ va.pause(); vb.pause(); $('play').textContent='▶'; }
    else { va.play().catch(()=>{}); vb.play().catch(()=>{}); $('play').textContent='⏸'; }
  }
  $('play').onclick=playPause;
  $('loop').onclick=()=>{ const on=!va.loop; va.loop=vb.loop=on; $('loop').classList.toggle('on',on); };
  va.loop=vb.loop=true;

  function fmt(t){ if(!isFinite(t))t=0; const m=Math.floor(t/60),s=(t%60).toFixed(2).padStart(5,'0'); return `${String(m).padStart(2,'0')}:${s}`; }
  function updateDur(){ $('dur').textContent=fmt(longer().duration||0); }
  function tick(){
    const L=longer(); const d=L.duration||0, c=L.currentTime||0;
    $('cur').textContent=fmt(c);
    if(d){ seek.value=Math.round(c/d*1000); }
    // mantém o outro vídeo em sincronia leve
    const other = L===va?vb:va;
    if(Math.abs((other.currentTime||0)-c)>0.08) other.currentTime=Math.min(c, other.duration||c);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
  va.addEventListener('loadedmetadata',updateDur);
  vb.addEventListener('loadedmetadata',updateDur);

  seek.addEventListener('input',()=>{
    const d=longer().duration||0, t=seek.value/1000*d;
    va.currentTime=Math.min(t, va.duration||t); vb.currentTime=Math.min(t, vb.duration||t);
  });

  // quadro a quadro (assume ~fps do vídeo: usa 1/30 como passo padrão)
  const STEP=1/30;
  $('prev').onclick=()=>seekBy(-STEP);
  $('next').onclick=()=>seekBy(STEP);
  function seekBy(dt){ va.pause(); vb.pause(); $('play').textContent='▶';
    const t=(longer().currentTime||0)+dt;
    va.currentTime=Math.max(0,Math.min(t,va.duration||t));
    vb.currentTime=Math.max(0,Math.min(t,vb.duration||t)); }

  // velocidade
  $('speed').onclick=()=>{ speedIdx=(speedIdx+1)%speeds.length; const r=speeds[speedIdx];
    va.playbackRate=vb.playbackRate=r; $('speed').textContent=r+'x'; };

  // áudio A/B/mudo
  function applyAudio(){ va.muted = muted || audio!=='A'; vb.muted = muted || audio!=='B';
    $('audioA').classList.toggle('on',audio==='A'&&!muted);
    $('audioB').classList.toggle('on',audio==='B'&&!muted);
    $('audioA').textContent=(audio==='A'?'🔊A':'🔇A'); $('audioB').textContent=(audio==='B'?'🔊B':'🔇B');
    $('mute').textContent=muted?'🔇':'🔊'; $('mute').classList.toggle('on',!muted); }
  $('audioA').onclick=()=>{ audio='A'; applyAudio(); };
  $('audioB').onclick=()=>{ audio='B'; applyAudio(); };
  $('mute').onclick=()=>{ muted=!muted; applyAudio(); };
  applyAudio();

  // tela cheia
  $('full').onclick=()=>{ if(!document.fullscreenElement) stage.requestFullscreen?.(); else document.exitFullscreen?.(); };

  // atalhos
  window.addEventListener('keydown',e=>{
    if(e.code==='Space'){ e.preventDefault(); playPause(); }
    else if(e.code==='ArrowLeft') seekBy(-STEP*5);
    else if(e.code==='ArrowRight') seekBy(STEP*5);
    else if(e.key==='?') alert('Atalhos:\nEspaço = play/pause\n←/→ = avançar/retroceder\nArraste vídeos para A (esq) e B (dir)\nModos: Cortina, Lado a Lado, Diferença, Alternar');
  });

  applyMode();
})();
</script>
</body>
</html>
