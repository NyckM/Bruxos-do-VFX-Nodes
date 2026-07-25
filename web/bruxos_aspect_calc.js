import { app } from "../../scripts/app.js";

const round16 = (v) => Math.max(16, Math.floor(Number(v) / 16 + 0.5) * 16);

function find(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function update(node, driver) {
  if (node._bruxosAspectUpdating) return;
  const width = find(node, "width");
  const height = find(node, "height");
  const mode = find(node, "calculate_from");
  if (!width || !height) return;
  node._bruxosAspectUpdating = true;
  try {
    const from = driver || mode?.value || "largura";
    if (from === "altura") {
      height.value = round16(height.value);
      width.value = round16(height.value * 16 / 9);
    } else {
      width.value = round16(width.value);
      height.value = round16(width.value * 9 / 16);
    }
    node.title = `Calculadora 16:9 · ${width.value}×${height.value} (Bruxos)`;
    node.setDirtyCanvas?.(true, true);
  } finally {
    node._bruxosAspectUpdating = false;
  }
}

app.registerExtension({
  name: "BruxosDoVFX.Aspect16x9Calculator",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "BruxosAspect16x9") return;
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      for (const name of ["width", "height", "calculate_from"]) {
        const w = find(this, name);
        if (!w) continue;
        const original = w.callback;
        w.callback = (...args) => {
          const r = original?.apply(w, args);
          const mode = find(this, "calculate_from")?.value;
          if (name === "calculate_from") update(this, mode);
          else if ((name === "width" && mode === "largura") ||
                   (name === "height" && mode === "altura")) update(this, mode);
          return r;
        };
      }
      update(this);
      return result;
    };
    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigure?.apply(this, arguments);
      setTimeout(() => update(this), 0);
      return result;
    };
  },
});
