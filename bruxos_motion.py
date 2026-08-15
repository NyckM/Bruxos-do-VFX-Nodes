"""Editor e utilitarios de movimento dos Bruxos.

O conceito de hold-map/time-smear foi inspirado pelo Motion Lab do
ComfyUI-MAINodes, copyright (c) 2026 MatlowAI, distribuido sob licença MIT.
Esta implementacao e independente, com contrato e interface dos Bruxos.
"""
import json
import math
import os
import uuid

import torch
import torch.nn.functional as F


CAT = "Bruxos do VFX/Motion"


def _profile_payload(value, length):
    if not value or not str(value).strip():
        return {"v": 1, "length": length, "values": [0.0] * length,
                "suggested_holds": [1] * length}
    data = json.loads(value)
    if int(data.get("length", length)) != length:
        raise ValueError(f"motion_profile cobre {data.get('length')} frames, mas o video tem {length}")
    return data


def _hold_payload(value, length):
    data = json.loads(value or "{}")
    holds = [max(1, int(round(x))) for x in data.get("holds", [])]
    if len(holds) != length:
        raise ValueError(f"hold_map cobre {len(holds)} frames, mas a entrada tem {length}")
    return data, holds


def _segments(holds):
    out, start = [], None
    for i, h in enumerate(holds + [1]):
        if h > 1 and start is None:
            start = i
        elif h <= 1 and start is not None:
            out.append([start, i - 1, max(holds[start:i])])
            start = None
    return out


def _raster_strokes(strokes, height, width):
    mask = torch.zeros((height, width), dtype=torch.float32)
    yy = torch.arange(height, dtype=torch.float32)[:, None]
    xx = torch.arange(width, dtype=torch.float32)[None, :]
    for stroke in strokes or []:
        erase = stroke.get("t") == "erase"
        radius = max(1.0, float(stroke.get("r", .035)) * width)
        points = stroke.get("pts") or []
        for a, b in zip(points, points[1:] or points):
            x0, y0 = float(a[0]) * width, float(a[1]) * height
            x1, y1 = float(b[0]) * width, float(b[1]) * height
            steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / max(radius * .45, 1)))
            for k in range(steps + 1):
                t = k / steps
                cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                disc = (xx - cx).square() + (yy - cy).square() <= radius * radius
                mask[disc] = 0.0 if erase else 1.0
        if len(points) == 1:
            cx, cy = float(points[0][0]) * width, float(points[0][1]) * height
            disc = (xx - cx).square() + (yy - cy).square() <= radius * radius
            mask[disc] = 0.0 if erase else 1.0
    return mask


class BruxosMotionAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "sensitivity": ("FLOAT", {"default": .55, "min": 0.05, "max": .95, "step": .01}),
            "max_hold": ("INT", {"default": 4, "min": 2, "max": 8}),
            "smooth_frames": ("INT", {"default": 3, "min": 1, "max": 15}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("motion_profile", "suggested_hold_map", "report")
    FUNCTION = "analyze"
    CATEGORY = CAT
    DESCRIPTION = "Analisa mudanca temporal e sugere onde desacelerar movimentos bruscos."

    def analyze(self, images, sensitivity=.55, max_hold=4, smooth_frames=3):
        x = images.detach().float().cpu()
        n = int(x.shape[0])
        if n < 2:
            vals = torch.zeros(n)
        else:
            gray = x[..., :3].mean(-1)[:, None]
            h, w = gray.shape[-2:]
            scale = min(1.0, 96.0 / max(h, w))
            gray = F.interpolate(gray, size=(max(8, round(h * scale)), max(8, round(w * scale))),
                                 mode="bilinear", align_corners=False)
            velocity = (gray[1:] - gray[:-1]).abs().mean((1, 2, 3))
            # Mudanca da energia de movimento: favorece aceleracoes/reversoes,
            # mas preserva uma parcela da velocidade para movimentos muito rapidos.
            accel = torch.cat((velocity[:1], (velocity[1:] - velocity[:-1]).abs()))
            raw = .65 * accel + .35 * velocity
            vals = torch.cat((raw[:1], raw))[:n]
            k = max(1, int(smooth_frames))
            if k > 1:
                vals = F.avg_pool1d(vals[None, None], k, stride=1, padding=k // 2)[0, 0, :n]
            lo, hi = torch.quantile(vals, .05), torch.quantile(vals, .95)
            vals = ((vals - lo) / max(float(hi - lo), 1e-8)).clamp(0, 1)
        threshold = 1.0 - float(sensitivity)
        holds = [1 + int(round(max(0.0, (float(v) - threshold) / max(1e-6, 1-threshold))
                             * (int(max_hold) - 1))) for v in vals]
        payload = {"v": 1, "length": n, "values": [round(float(v), 5) for v in vals],
                   "suggested_holds": holds, "threshold": threshold}
        hold = {"v": 1, "world_len": n, "holds": holds, "source": "Bruxos Motion Analyzer"}
        segs = _segments(holds)
        report = f"{n} frames; {sum(h > 1 for h in holds)} marcados; {len(segs)} segmentos; {sum(holds)} frames com smear"
        return (json.dumps(payload), json.dumps(hold), report)


class BruxosMotionTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "editor_state": ("STRING", {"default": '{"v":1,"blocks":[]}', "multiline": True}),
        }, "optional": {
            "motion_profile": ("STRING", {"default": "", "forceInput": True}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": .01}),
            "paint_resolution": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64}),
            "use_suggestions_when_empty": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("STRING", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("hold_map", "motion_mask", "mask_preview", "report")
    FUNCTION = "compile"
    CATEGORY = CAT
    OUTPUT_NODE = True
    DESCRIPTION = "Timeline visual Nodes 2.0: blocos temporais, hold e pintura de mascara por frame."

    def _thumbs(self, images):
        try:
            import folder_paths
            from PIL import Image
            root = os.path.join(folder_paths.get_temp_directory(), "bruxos_motion")
            os.makedirs(root, exist_ok=True)
            tag = uuid.uuid4().hex[:8]
            arr = (images.clamp(0, 1) * 255).byte().cpu().numpy()
            refs = []
            for i, frame in enumerate(arr):
                im = Image.fromarray(frame[..., :3])
                im.thumbnail((160, 100))
                name = f"{tag}_{i:04d}.jpg"
                im.save(os.path.join(root, name), quality=80)
                refs.append({"filename": name, "subfolder": "bruxos_motion", "type": "temp"})
            return refs
        except Exception:
            return []

    def compile(self, images, editor_state, motion_profile="", fps=24.0,
                paint_resolution=512, use_suggestions_when_empty=True):
        x = images.detach().float().cpu()
        n, h, w, _ = x.shape
        state = json.loads(editor_state or "{}")
        blocks = state.get("blocks") or []
        profile = _profile_payload(motion_profile, n)
        holds = ([max(1, int(v)) for v in profile.get("suggested_holds", [1] * n)]
                 if use_suggestions_when_empty and not blocks else [1] * n)
        ph = max(1, round(h * min(int(paint_resolution), w) / w))
        pw = min(int(paint_resolution), w)
        mask = torch.zeros((n, ph, pw), dtype=torch.float32)
        for block in blocks:
            a = max(0, int(block.get("start", 0)))
            z = min(n - 1, int(block.get("end", a)))
            # hold=0 na interface significa "usar a sugestao do analyzer";
            # sem analyzer, usa 4 como valor seguro/visivel para o teste.
            block_hold = int(block.get("hold", 0))
            static = block.get("static_strokes") or []
            per_frame = block.get("strokes") or {}
            for f in range(a, z + 1):
                hold = block_hold if block_hold > 0 else max(
                    1, int((profile.get("suggested_holds") or [4] * n)[f]))
                if block_hold <= 0 and hold <= 1:
                    hold = 4
                holds[f] = max(holds[f], hold)
                strokes = static + (per_frame.get(str(f)) or [])
                mask[f] = _raster_strokes(strokes, ph, pw) if strokes else 1.0
        mask_full = F.interpolate(mask[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        overlay = x.clone()
        alpha = mask_full[..., None] * .48
        red = torch.zeros_like(overlay); red[..., 0] = 1.0
        overlay = overlay * (1 - alpha) + red * alpha
        payload = {"v": 1, "world_len": n, "holds": holds,
                   "segments": _segments(holds), "fps": float(fps)}
        report = (f"{n}f -> {sum(holds)}f ({sum(holds)/max(n,1):.2f}x); "
                  f"{len(payload['segments'])} segmentos; {int((mask_full > 0).sum())} pixels-frame ativos")
        ui = {"bruxos_motion_frames": self._thumbs(x),
              "bruxos_motion_profile": profile.get("values", []),
              "bruxos_motion_length": [int(n)], "bruxos_motion_fps": [float(fps)],
              "bruxos_motion_report": [report]}
        return {"ui": ui, "result": (json.dumps(payload), mask_full, overlay, report)}


class BruxosMotionTimeSmear:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "hold_map": ("STRING", {"forceInput": True})},
                "optional": {"pad_h3_17k5": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "hold_map_used", "frame_count")
    FUNCTION = "apply"
    CATEGORY = CAT

    def apply(self, images, hold_map, pad_h3_17k5=False):
        _, holds = _hold_payload(hold_map, int(images.shape[0]))
        indices = torch.repeat_interleave(torch.arange(len(holds)), torch.tensor(holds))
        out = images.index_select(0, indices.to(images.device))
        if pad_h3_17k5:
            target = max(22, int(math.ceil((int(out.shape[0]) - 5) / 17)) * 17 + 5)
            if target > int(out.shape[0]):
                extra = target - int(out.shape[0])
                out = torch.cat((out, out[-1:].expand(extra, -1, -1, -1)), 0)
                holds[-1] += extra
        used = {"v": 1, "world_len": len(holds), "holds": holds}
        return (out.contiguous(), json.dumps(used), int(out.shape[0]))


class BruxosMotionRecover:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "hold_map": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "recover"
    CATEGORY = CAT

    def recover(self, images, hold_map):
        data = json.loads(hold_map)
        holds = [max(1, int(v)) for v in data.get("holds", [])]
        starts, cursor = [], 0
        for hold in holds:
            starts.append(min(cursor, int(images.shape[0]) - 1)); cursor += hold
        out = images.index_select(0, torch.tensor(starts, device=images.device))
        return (out.contiguous(), int(out.shape[0]))


NODE_CLASS_MAPPINGS = {
    "BruxosMotionAnalyzer": BruxosMotionAnalyzer,
    "BruxosMotionTimeline": BruxosMotionTimeline,
    "BruxosMotionTimeSmear": BruxosMotionTimeSmear,
    "BruxosMotionRecover": BruxosMotionRecover,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosMotionAnalyzer": "Motion Analyzer (Bruxos)",
    "BruxosMotionTimeline": "Motion Timeline (Bruxos)",
    "BruxosMotionTimeSmear": "Motion Time Smear (Bruxos)",
    "BruxosMotionRecover": "Motion Recover (Bruxos)",
}
