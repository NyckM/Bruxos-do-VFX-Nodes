# -*- coding: utf-8 -*-
"""
Bruxos Tracking Export
======================
Exporta uma CAMERA_TRAJECTORY pro VFX. 100% logica propria (sem modelo):
  - JSON universal (poses 4x4 + intrinsics)
  - Nuke .chan (padrao de camera em composicao: frame tx ty tz rx ry rz)

Tolerante ao formato: aceita 'matrices' OU 'poses'/'translations'+'rotations',
porque os trackers do pacote variam a chave.
"""

import os
import json
import math

try:
    import numpy as np
except Exception:
    np = None

try:
    import folder_paths as _fp
except Exception:
    _fp = None


def _out_dir():
    if _fp is not None:
        try:
            return _fp.get_output_directory()
        except Exception:
            pass
    return os.path.join(os.getcwd(), "output")


def _to_np(x):
    if np is None:
        return x
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _matrices_from_traj(traj):
    """Normaliza varios formatos -> array (T,4,4)."""
    if isinstance(traj, dict):
        for k in ("matrices", "poses"):
            if k in traj and traj[k] is not None:
                m = _to_np(traj[k])
                if m.ndim == 3 and m.shape[-2:] == (4, 4):
                    return m
        # translations + rotations(quaternion) -> matrizes
        if "translations" in traj and "rotations" in traj:
            t = _to_np(traj["translations"])
            q = _to_np(traj["rotations"])
            T = t.shape[0]
            out = np.repeat(np.eye(4)[None], T, axis=0)
            for i in range(T):
                out[i, :3, :3] = _quat_to_mat(q[i])
                out[i, :3, 3] = t[i]
            return out
    m = _to_np(traj)
    if m.ndim == 3 and m.shape[-2:] == (4, 4):
        return m
    raise ValueError("Trajetoria sem 'matrices'/'poses' reconheciveis.")


def _quat_to_mat(q):
    # q = [qw, qx, qy, qz]
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = math.sqrt(w*w + x*x + y*y + z*z) or 1.0
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=float)


def _mat_to_euler_xyz_deg(R):
    # XYZ intrinseco -> graus
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    d = 180.0 / math.pi
    return rx*d, ry*d, rz*d


class BruxosTrackingExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trajectory": ("CAMERA_TRAJECTORY", {"tooltip": "Saida do Camera Tracker (Bruxos)."}),
                "formato": (["json", "nuke_chan", "ambos"], {"default": "ambos",
                    "tooltip": "json = universal (poses 4x4). nuke_chan = camera pro Nuke/composicao (frame tx ty tz rx ry rz)."}),
                "nome_arquivo": ("STRING", {"default": "bruxos_camera",
                    "tooltip": "Nome base do arquivo (sem extensao). Salva na pasta output do ComfyUI."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    OUTPUT_TOOLTIPS = ("Caminho do(s) arquivo(s) exportado(s).",)
    FUNCTION = "export"
    CATEGORY = "Bruxos do VFX/Tracking"
    DESCRIPTION = ("Exporta a trajetoria da camera pro VFX: JSON universal e/ou Nuke .chan "
                   "(padrao de camera em composicao). Salva na pasta output do ComfyUI.")

    def export(self, trajectory, formato="ambos", nome_arquivo="bruxos_camera"):
        if np is None:
            raise RuntimeError("numpy indisponivel.")
        mats = _matrices_from_traj(trajectory)
        T = mats.shape[0]
        out_dir = _out_dir()
        os.makedirs(out_dir, exist_ok=True)
        salvos = []

        if formato in ("json", "ambos"):
            data = {"format": "bruxos-camera", "frames": int(T),
                    "matrices": mats.tolist()}
            if isinstance(trajectory, dict) and "intrinsics" in trajectory:
                try:
                    data["intrinsics"] = _to_np(trajectory["intrinsics"]).tolist()
                except Exception:
                    pass
            p = os.path.join(out_dir, f"{nome_arquivo}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            salvos.append(p)
            print(f"[Bruxos Tracking Export] JSON -> {p}", flush=True)

        if formato in ("nuke_chan", "ambos"):
            p = os.path.join(out_dir, f"{nome_arquivo}.chan")
            with open(p, "w", encoding="utf-8") as f:
                for i in range(T):
                    M = mats[i]
                    tx, ty, tz = M[0, 3], M[1, 3], M[2, 3]
                    rx, ry, rz = _mat_to_euler_xyz_deg(M[:3, :3])
                    # Nuke .chan: frame tx ty tz rx ry rz  (1-based frame)
                    f.write(f"{i+1} {tx:.6f} {ty:.6f} {tz:.6f} {rx:.6f} {ry:.6f} {rz:.6f}\n")
            salvos.append(p)
            print(f"[Bruxos Tracking Export] Nuke .chan -> {p}", flush=True)

        return (" | ".join(salvos),)


NODE_CLASS_MAPPINGS = {"BruxosTrackingExport": BruxosTrackingExport}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosTrackingExport": "Tracking Export (Bruxos)"}


class BruxosPointsExport:
    """Exporta POINT_TRACKS ou OBJECT_TRACKS pro VFX (JSON com trajetorias 2D)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks": ("POINT_TRACKS,OBJECT_TRACKS", {"tooltip": "Saida do Point Tracker OU Object Tracker."}),
                "nome_arquivo": ("STRING", {"default": "bruxos_tracks",
                    "tooltip": "Nome base (sem extensao). Salva na pasta output do ComfyUI."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    OUTPUT_TOOLTIPS = ("Caminho do JSON exportado.",)
    FUNCTION = "export"
    CATEGORY = "Bruxos do VFX/Tracking"
    DESCRIPTION = ("Exporta trajetorias de PONTOS/OBJETO pro VFX (JSON: [frame][ponto] = x,y "
                   "+ visibilidade). Levar pro After Effects/Nuke como tracking 2D.")

    def export(self, tracks, nome_arquivo="bruxos_tracks"):
        vis = None
        n_obj = None
        if isinstance(tracks, dict):
            vis = tracks.get("visibility", None)
            n_obj = tracks.get("n_objects", None)
            tracks = tracks.get("tracks", tracks.get("object_tracks", None))
        arr = _to_np(tracks)
        if arr.ndim == 4:
            arr = arr[0]
        # arr agora [T, N, 2+]
        T, N = int(arr.shape[0]), int(arr.shape[1])
        visN = _to_np(vis) if vis is not None else None
        if visN is not None and visN.ndim == 3:
            visN = visN[0]

        data = {
            "format": "bruxos-points",
            "frames": T, "n_points": N,
            "n_objects": (int(n_obj) if n_obj is not None else None),
            "tracks": [],
        }
        for t in range(T):
            frame_pts = []
            for i in range(N):
                x = float(arr[t, i, 0]); y = float(arr[t, i, 1])
                v = 1
                if visN is not None:
                    try:
                        v = int(bool(visN[t, i]))
                    except Exception:
                        v = 1
                frame_pts.append({"x": x, "y": y, "v": v})
            data["tracks"].append(frame_pts)

        out_dir = _out_dir()
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"{nome_arquivo}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Bruxos Points Export] {T} frames x {N} pontos -> {p}", flush=True)
        return (p,)


NODE_CLASS_MAPPINGS["BruxosPointsExport"] = BruxosPointsExport
NODE_DISPLAY_NAME_MAPPINGS["BruxosPointsExport"] = "Points Export (Bruxos)"
