import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Autocomplete do campo 'texto' do node "MiniMax H3 · Prompt Rapido @ # (Bruxos)".
//
//   '#' -> lista as 15 macros de papel
//   '@' -> lista <Picture 1..9>, <Video 1..3>, <Audio 1..3>
//
// POR QUE OUVIR NO DOCUMENT E NAO NO WIDGET
//   A versao anterior procurava widget.inputEl. Isso so existe no canvas
//   classico; no Nodes 2.0 o campo multilinha e componente Vue e o inputEl
//   nunca aparece -- o popup simplesmente nunca abria.
//   Ouvir no document funciona nos dois, porque nos dois o que recebe o texto
//   e um <textarea> de verdade. O dono e resolvido DEPOIS, pelo elemento.
//
// O catalogo vem de /bruxos/h3_macros, que serve o proprio _BLOCOS do Python.
// Ideia do '@'/'#': ComfyUI-MiniMaxH3-Easy (nkxx188, MIT).

const NODE = "BruxosH3PromptRapido";
const WIDGET = "texto";

let CAT = null;
async function loadCat() {
  if (CAT) return CAT;
  try {
    const r = await api.fetchApi("/bruxos/h3_macros");
    if (r.status === 200) CAT = await r.json();
    else console.warn("[Bruxos H3] /bruxos/h3_macros respondeu", r.status);
  } catch (e) {
    console.warn("[Bruxos H3] nao consegui carregar o catalogo:", e);
  }
  return CAT;
}

// tira acento e caixa, igual o _sem_acento() do Python.
// A classe sao os diacriticos combinantes U+0300-U+036F -- caracteres
// invisiveis no editor. Se este arquivo sair do UTF-8, o filtro por acento
// para de funcionar em silencio.
const norm = (s) =>
  (s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

// ------------------------------------------------------------- catalogo ---
function itensPara(gatilho) {
  if (!CAT) return [];
  if (gatilho === "#") {
    return CAT.macros.map((m) => ({
      inserir: m.nome,
      titulo: "#" + m.nome,
      lado: m.tag ? `<${m.tag} N>` : "—",
      desc: m.curto,
      busca: [m.nome, ...(m.sinonimos || [])].map(norm),
    }));
  }
  const out = [];
  for (const r of CAT.refs) {
    for (let i = 1; i <= r.limite; i++) {
      out.push({
        inserir: r.principal + i,
        titulo: "@" + r.principal + i,
        lado: `<${r.tag} ${i}>`,
        desc: "",
        busca: (r.sinonimos || []).map((a) => norm(a + i)),
      });
    }
  }
  return out;
}

// ---------------------------------------------------------------- popup ---
const pop = document.createElement("div");
Object.assign(pop.style, {
  position: "fixed", zIndex: 100000, display: "none",
  maxHeight: "300px", overflowY: "auto", minWidth: "330px", maxWidth: "520px",
  background: "#1e1e1e", border: "1px solid #4a4a4a", borderRadius: "6px",
  boxShadow: "0 6px 22px rgba(0,0,0,.65)", padding: "4px",
  font: "12px system-ui, sans-serif", color: "#ddd",
});
document.body.appendChild(pop);

let alvo = null;      // textarea em edicao
let itens = [];
let sel = 0;
let gatilho = null;
let inicio = -1;

function fechar() {
  pop.style.display = "none";
  alvo = null; itens = []; gatilho = null; inicio = -1;
}

function desenhar() {
  if (!itens.length || !alvo) return fechar();
  pop.innerHTML = "";
  itens.forEach((it, i) => {
    const linha = document.createElement("div");
    Object.assign(linha.style, {
      padding: "5px 8px", borderRadius: "4px", cursor: "pointer",
      background: i === sel ? "#0d5c8c" : "transparent",
    });
    const topo = document.createElement("div");
    Object.assign(topo.style, { display: "flex", justifyContent: "space-between", gap: "14px" });
    const nome = document.createElement("span");
    nome.textContent = it.titulo;
    nome.style.fontWeight = "600";
    const tag = document.createElement("span");
    tag.textContent = it.lado;
    tag.style.opacity = "0.55";
    topo.append(nome, tag);
    linha.appendChild(topo);
    if (it.desc) {
      const d = document.createElement("div");
      d.textContent = it.desc;
      Object.assign(d.style, { opacity: "0.72", marginTop: "2px", lineHeight: "1.35" });
      linha.appendChild(d);
    }
    linha.addEventListener("mouseenter", () => { sel = i; desenhar(); });
    // mousedown e nao click: o click chega depois do blur, com o popup fechado
    linha.addEventListener("mousedown", (e) => { e.preventDefault(); aceitar(i); });
    pop.appendChild(linha);
  });
  const r = alvo.getBoundingClientRect();
  pop.style.left = Math.max(4, Math.min(r.left + 8, window.innerWidth - 540)) + "px";
  pop.style.top = Math.max(4, Math.min(r.bottom + 4, window.innerHeight - 320)) + "px";
  pop.style.display = "block";
  const marcado = pop.children[sel];
  if (marcado) marcado.scrollIntoView({ block: "nearest" });
}

function aceitar(i) {
  const it = itens[i];
  const ta = alvo;
  if (!it || !ta || inicio < 0) return fechar();
  const fim = ta.selectionStart;
  const antes = ta.value.slice(0, inicio);
  const depois = ta.value.slice(fim);
  const trecho = gatilho + it.inserir + " ";
  ta.value = antes + trecho + depois;
  const caret = (antes + trecho).length;
  fechar();                                   // fecha ANTES do dispatch
  ta.setSelectionRange(caret, caret);
  // avisa o ComfyUI (classico e Vue escutam 'input') para o valor persistir
  ta.dispatchEvent(new Event("input", { bubbles: true }));
  ta.dispatchEvent(new Event("change", { bubbles: true }));
  ta.focus();
}

// ------------------------------------------------------------ resolucao ---
function ehNosso(ta) {
  if (!ta || ta.tagName !== "TEXTAREA") return false;
  const nodes = (app.graph?._nodes || []).filter((n) => n.type === NODE);
  if (!nodes.length) return false;
  for (const n of nodes) {
    const w = n.widgets?.find((x) => x.name === WIDGET);
    if (!w) continue;
    const el = w.inputEl || w.element;
    if (el === ta) return true;
    if (el?.contains?.(ta)) return true;
  }
  // Nodes 2.0: as vezes o textarea nao esta pendurado no widget. Se houver
  // exatamente UM node desses no grafo e o texto bater, assume que e ele.
  if (nodes.length === 1) {
    const w = nodes[0].widgets?.find((x) => x.name === WIDGET);
    if (w && typeof w.value === "string") {
      const v = ta.value;
      if (v === w.value || w.value.startsWith(v.slice(0, Math.max(0, v.length - 2)))) return true;
    }
  }
  return false;
}

function avaliar(ta) {
  const caret = ta.selectionStart;
  const txt = ta.value.slice(0, caret);
  const m = /([#@])([A-Za-zÀ-ÿ0-9]*)$/.exec(txt);
  if (!m) return fechar();
  // '##' e '@@' sao escape literal: nao abre
  const antesDoGatilho = txt[txt.length - m[0].length - 1];
  if (antesDoGatilho === m[1]) return fechar();
  alvo = ta;
  gatilho = m[1];
  inicio = caret - m[0].length;
  const filtro = norm(m[2]);
  const todos = itensPara(gatilho);
  itens = filtro
    ? todos.filter((it) => it.busca.some((b) => b.startsWith(filtro)))
    : todos;
  sel = 0;
  desenhar();
}

// -------------------------------------------------------------- eventos ---
document.addEventListener("input", (e) => {
  const ta = e.target;
  if (!ehNosso(ta)) { if (alvo === ta) fechar(); return; }
  try { avaliar(ta); } catch (err) { console.warn("[Bruxos H3]", err); fechar(); }
});

// captura: precisa vir ANTES dos atalhos globais do ComfyUI, senao o Enter
// enfileira o prompt em vez de aceitar o item da lista.
document.addEventListener("keydown", (e) => {
  if (pop.style.display === "none" || !alvo || e.target !== alvo) return;
  if (e.key === "ArrowDown") {
    sel = (sel + 1) % itens.length; desenhar(); e.preventDefault(); e.stopPropagation();
  } else if (e.key === "ArrowUp") {
    sel = (sel - 1 + itens.length) % itens.length; desenhar(); e.preventDefault(); e.stopPropagation();
  } else if (e.key === "Enter" || e.key === "Tab") {
    aceitar(sel); e.preventDefault(); e.stopPropagation();
  } else if (e.key === "Escape") {
    fechar(); e.preventDefault(); e.stopPropagation();
  }
}, true);

document.addEventListener("mousedown", (e) => {
  if (!pop.contains(e.target) && e.target !== alvo) fechar();
});
window.addEventListener("blur", fechar);
document.addEventListener("scroll", fechar, true);

app.registerExtension({
  name: "BruxosDoVFX.H3Macros",
  async setup() {
    await loadCat();
    const n = CAT?.macros?.length ?? 0;
    console.log(`[Bruxos H3] macros carregadas: ${n}. Digite # ou @ no campo 'texto'.`);
  },
});
