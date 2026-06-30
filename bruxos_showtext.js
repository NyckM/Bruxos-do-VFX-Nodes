import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Faz o node Prompt Guide (Bruxos) se comportar como o do Deno:
//  - ao trocar 'model', filtra 'task' e 'negative_preset' para esse modelo
//  - ao trocar 'task', auto-preenche 'system_prompt' com o texto do preset
//  - ao trocar 'negative_preset', auto-preenche 'negative_prompt'

let PRESETS = null;
async function loadPresets() {
  if (PRESETS) return PRESETS;
  try {
    const r = await api.fetchApi("/bruxos/prompt_presets");
    if (r.status === 200) PRESETS = await r.json();
  } catch (e) {
    console.warn("[Bruxos] nao consegui carregar presets:", e);
  }
  return PRESETS;
}

function getWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function setComboOptions(widget, values, keepIfPossible) {
  if (!widget) return;
  widget.options = widget.options || {};
  widget.options.values = values;
  if (!(keepIfPossible && values.includes(widget.value))) {
    widget.value = values[0];
  }
}

function applyModel(node, model, fillTexts) {
  if (!PRESETS || !PRESETS.presets[model]) return;
  const p = PRESETS.presets[model];
  const taskNames = Object.keys(p.tasks);
  const negNames = Object.keys(p.negatives);
  const taskW = getWidget(node, "task");
  const negW = getWidget(node, "negative_preset");
  setComboOptions(taskW, taskNames, true);
  setComboOptions(negW, negNames, true);
  if (fillTexts) {
    fillSystem(node, model, taskW?.value);
    fillNegative(node, model, negW?.value);
  }
  node.setDirtyCanvas(true, true);
}

function fillSystem(node, model, task) {
  if (!PRESETS) return;
  const p = PRESETS.presets[model];
  if (!p) return;
  const sysW = getWidget(node, "system_prompt");
  if (sysW && task in p.tasks) {
    sysW.value = p.tasks[task] || "";
  }
}

function fillNegative(node, model, negName) {
  if (!PRESETS) return;
  const p = PRESETS.presets[model];
  if (!p) return;
  const negW = getWidget(node, "negative_prompt");
  if (negW && negName in p.negatives) {
    negW.value = p.negatives[negName] || "";
  }
}

app.registerExtension({
  name: "BruxosDoVFX.PromptGuide",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "BruxosPromptGuide") return;
    await loadPresets();

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const node = this;

      const modelW = getWidget(node, "model");
      const taskW = getWidget(node, "task");
      const negW = getWidget(node, "negative_preset");

      const wrap = (w, fn) => {
        if (!w) return;
        const prev = w.callback;
        w.callback = function () {
          const res = prev ? prev.apply(this, arguments) : undefined;
          try { fn(); } catch (e) { console.warn("[Bruxos]", e); }
          return res;
        };
      };

      wrap(modelW, () => applyModel(node, modelW.value, true));
      wrap(taskW, () => fillSystem(node, modelW?.value, taskW.value));
      wrap(negW, () => fillNegative(node, modelW?.value, negW.value));

      // estado inicial (sem sobrescrever textos que o usuario salvou no workflow)
      loadPresets().then(() => applyModel(node, modelW?.value, false));

      return r;
    };
  },
});
