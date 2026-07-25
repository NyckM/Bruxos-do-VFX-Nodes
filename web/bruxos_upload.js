import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Botao de upload no node Load Video (Bruxos), espelhando o padrao do VHS.
console.log("[Bruxos] extensao de upload de video carregada");

function chainCallback(object, property, callback) {
  if (object == undefined) return;
  if (property in object) {
    const orig = object[property];
    object[property] = function () {
      const r = orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r;
    };
  } else {
    object[property] = callback;
  }
}

async function uploadFile(file) {
  try {
    const body = new FormData();
    body.append("image", file);
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (resp.status !== 200) {
      alert("[Bruxos] upload falhou: " + resp.status + " - " + resp.statusText);
      return null;
    }
    const data = await resp.json();
    return data.subfolder ? data.subfolder + "/" + data.name : data.name;
  } catch (e) {
    alert("[Bruxos] erro no upload: " + e);
    return null;
  }
}

function addUploadButton(nodeType, widgetName, opts) {
  opts = opts || {};
  const accept = opts.accept ||
    "video/webm,video/mp4,video/x-matroska,image/gif,video/quicktime,.mp4,.mov,.mkv,.avi,.webm,.gif,.m4v,.wmv,.flv";
  const label = opts.label || "📁 escolher vídeo (upload)";
  chainCallback(nodeType.prototype, "onNodeCreated", function () {
    const node = this;
    const pathWidget = node.widgets?.find((w) => w.name === widgetName);

    const fileInput = document.createElement("input");
    chainCallback(node, "onRemoved", () => fileInput?.remove());

    async function doUpload(file) {
      const name = await uploadFile(file);
      if (!name) return false;
      if (pathWidget) {
        pathWidget.options = pathWidget.options || {};
        pathWidget.options.values = pathWidget.options.values || [];
        if (!pathWidget.options.values.includes(name)) pathWidget.options.values.push(name);
        pathWidget.value = name;
        if (pathWidget.callback) pathWidget.callback(name);
      }
      app.graph.setDirtyCanvas(true, true);
      return true;
    }

    Object.assign(fileInput, {
      type: "file",
      accept,
      style: "display: none",
      onchange: async () => {
        if (fileInput.files.length) await doUpload(fileInput.files[0]);
      },
    });

    // drag-and-drop direto no node
    node.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
    node.onDragDrop = async function (e) {
      if (!e?.dataTransfer?.types?.includes?.("Files")) return false;
      const item = e.dataTransfer?.files?.[0];
      if (item) return await doUpload(item);
      return false;
    };

    document.body.append(fileInput);

    const uploadWidget = node.addWidget("button", label, "upload", () => {
      app.canvas.node_widget = null;
      fileInput.click();
    });
    // ComfyUI 0.28 ignora options.serialize -> forca por propriedade tambem,
    // senao o botao entra em widgets_values ("upload") e desloca os valores.
    uploadWidget.options = uploadWidget.options || {};
    uploadWidget.options.serialize = false;
    uploadWidget.serialize = false;
    uploadWidget.serializeValue = () => undefined;
  });
}

app.registerExtension({
  name: "BruxosDoVFX.LoadVideoUpload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === "BruxosLoadVideo") {
      addUploadButton(nodeType, "video");
    } else if (nodeData?.name === "BruxosLoadImage") {
      addUploadButton(nodeType, "image", {
        accept: "image/png,image/jpeg,image/webp,image/bmp,image/gif,image/tiff,.png,.jpg,.jpeg,.webp,.bmp,.gif,.tif,.tiff",
        label: "\uD83D\uDCC1 escolher imagem (upload)",
      });
    }
  },
});
