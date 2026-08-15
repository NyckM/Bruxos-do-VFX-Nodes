import { app } from "../../scripts/app.js";

// Nao reordene node.widgets aqui: o renderer LiteGraph calcula widget.y antes
// do onNodeCreated terminar. Alterar a ordem depois disso fazia o DOMWidget do
// video sair do node e podia deslocar widgets_values.

const LOADERS = new Set(["BruxosLoadImage", "BruxosLoadVideo"]);

function isVueNodes() {
  return !!window.LiteGraph?.vueNodesMode;
}

function fitVisibleHeight(node) {
  const apply = () => {
    // computeSize mede apenas os widgets realmente visiveis. Use a altura
    // medida diretamente: Math.max com a altura atual impedia o node de
    // encolher depois de esconder os advanced inputs.
    const measured = node.computeSize?.();
    if (!measured) return;
    node.setSize?.([
      Math.max(Number(node.size?.[0]) || 0, Number(measured[0]) || 0),
      Math.max(120, Number(measured[1]) || 0),
    ]);
    node.arrange?.();
    node.setDirtyCanvas?.(true, true);
  };
  requestAnimationFrame(apply);
  // Nodes 2.0 conclui a transicao do NodeFooter um pouco depois do clique.
  // A segunda medida pega o estado final e tambem corrige workflows antigos
  // que foram salvos com a grande area vazia.
  setTimeout(apply, 80);
}

function loaderName(node) {
  return node?.comfyClass || node?.constructor?.comfyClass || node?.type || "";
}

function fitHiddenLoader(node) {
  if ((node?._bruxosLoaderLayout || LOADERS.has(loaderName(node))) && node.showAdvanced !== true)
    fitVisibleHeight(node);
}

function fitRestoredLoaders() {
  const nodes = app.graph?._nodes || [];
  for (const node of nodes) fitHiddenLoader(node);
}

function addLegacyAdvancedButton(node) {
  // Nodes 2.0 ja possui o NodeFooter nativo. No Nodes 1.0 a mesma funcao fica
  // apenas no menu de contexto; este botao torna o controle visivel no node.
  if (isVueNodes() || node._bruxosLegacyAdvancedButton) return;

  let button;
  const updateLabel = () => {
    const label = node.showAdvanced
      ? "Hide advanced inputs ^"
      : "Show advanced inputs v";
    if (button) {
      button.name = label;
      button.label = label;
    }
  };

  button = node.addWidget("button", "", null, () => {
    if (typeof node.toggleAdvanced === "function") {
      node.toggleAdvanced();
    } else {
      node.showAdvanced = !node.showAdvanced;
    }
    updateLabel();
    fitVisibleHeight(node);
  }, { serialize: false });
  button.serialize = false;
  button.options = button.options || {};
  button.options.canvasOnly = true;
  node._bruxosLegacyAdvancedButton = button;
  updateLabel();
}

app.registerExtension({
  name: "BruxosDoVFX.LoaderLegacyAdvancedButton",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!LOADERS.has(nodeData?.name)) return;

    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      this._bruxosLoaderLayout = true;
      // Aguarda os previews adicionados por outras extensoes terminarem, para
      // o botao ficar realmente no rodape sem reordenar a lista depois.
      queueMicrotask(() => {
        addLegacyAdvancedButton(this);
        // Intercepta tambem o NodeFooter nativo do Nodes 2.0. A implementacao
        // original apenas alterna showAdvanced; agora a moldura acompanha.
        if (!this._bruxosAdvancedResizeHook && typeof this.toggleAdvanced === "function") {
          const toggle = this.toggleAdvanced;
          this.toggleAdvanced = function () {
            const value = toggle.apply(this, arguments);
            fitVisibleHeight(this);
            return value;
          };
          this._bruxosAdvancedResizeHook = true;
        }
      });
      return result;
    };

    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = configured?.apply(this, arguments);
      const button = this._bruxosLegacyAdvancedButton;
      if (button) {
        const label = this.showAdvanced
          ? "Hide advanced inputs ^"
          : "Show advanced inputs v";
        button.name = label;
        button.label = label;
      }
      // Se advanced esta fechado, nao restaure a altura gigante de uma versao
      // anterior do node. Se esta aberto, preserve o tamanho salvo pelo usuario.
      if (!this.showAdvanced) fitVisibleHeight(this);
      return result;
    };

    // O serializer atual grava widgets_values usando o indice visual e deixa
    // buracos para DOMWidgets serialize:false. Como no Nodes 2.0 o preview foi
    // colocado antes dos advanced, compacte a lista para o loader sequencial
    // restaurar cada valor no input correto.
    const serialized = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (info) {
      const result = serialized?.apply(this, arguments);
      if (Array.isArray(this.widgets) && Array.isArray(info?.widgets_values)) {
        info.widgets_values = this.widgets
          .filter((item) => item?.serialize !== false)
          .map((item) => {
            const value = item?.value;
            return value != null && typeof value === "object"
              ? JSON.parse(JSON.stringify(value))
              : (value ?? null);
          });
      }
      // O Nodes 2 grava `size` antes/de forma independente dos widgets. Se a
      // workflow veio de uma versao antiga, nao perpetue a altura vazia no
      // proximo F5: salve a altura natural quando os advanced estao fechados.
      if (this.showAdvanced !== true && Array.isArray(info?.size)) {
        const measured = this.computeSize?.();
        if (measured) info.size[1] = Math.max(120, Number(measured[1]) || 0);
      }
      return result;
    };
  },

  afterConfigureGraph() {
    // Neste hook o ComfyUI ja terminou de aplicar `node.size` do JSON. Alguns
    // DOMWidgets/NodeFooter terminam o layout em ticks posteriores, por isso
    // repetimos apenas durante a abertura (nunca durante resize manual).
    fitRestoredLoaders();
    for (const delay of [80, 250, 700]) setTimeout(fitRestoredLoaders, delay);
  },
});
