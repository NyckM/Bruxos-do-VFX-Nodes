import { app } from "../../scripts/app.js";

// Exibe o texto recebido no node "Mostrar Texto (Bruxos)".
console.log("[Bruxos] mostrar-texto carregado");

function setText(node, textList) {
  const val = (textList || []).join("\n");
  let w = node.widgets && node.widgets.find((x) => x.name === "bruxos_text_out");
  if (!w) {
    const el = document.createElement("textarea");
    el.readOnly = true;
    el.style.cssText =
      "width:100%;box-sizing:border-box;min-height:60px;resize:vertical;" +
      "background:#181818;color:#ddd;border:1px solid #333;border-radius:6px;" +
      "font-family:monospace;font-size:11px;padding:6px;";
    w = node.addDOMWidget("bruxos_text_out", "text", el, { serialize: false });
    w._el = el;
  }
  w._el.value = val;
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "BruxosDoVFX.ShowText",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData && nodeData.name === "BruxosShowText") {
      const onExec = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        if (onExec) onExec.apply(this, arguments);
        if (message && message.text) setText(this, message.text);
      };
    }
  },
});
