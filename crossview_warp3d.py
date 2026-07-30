"""BruxosCrossViewWarp3D - CrossView Warp com preview 3D real (Bruxos do VFX).

Mesma entrada/saida do node "CrossView Warp" (ComfyUI-CrossViewWarp): frames + depth ->
warp (video de controle, buracos magenta) + orbit_view (diagrama da camera). A matematica
de reprojecao/orbit/keyframes abaixo e um port -- com pequenos ajustes de nomes -- do
crossview_warp_node.py do pacote ComfyUI-CrossViewWarp.

    NOTICE: portado de ComfyUI-CrossViewWarp (crossview_warp_node.py),
    licenca Apache License 2.0. Ver LICENSE em custom_nodes/ComfyUI-CrossViewWarp.

O que este node ACRESCENTA em cima do CrossViewWarp original:

  1. Um viewer 3D de verdade embutido no proprio node (web/bruxos_crossview3d.js -- um
     widget so, sem iframe/HTML separado), no lugar do gizmo 2D abstrato -- reconstroi a
     cena a partir de frames+depth como uma
     NUVEM DE PONTOS colorida (ao vivo, recalculada no navegador a cada frame de preview) OU
     como um "gaussian splat" sintetico (mesmo formato .ply usado pelo SHARP/comfyui-GaussianViewer),
     escolhido pelo widget 'render_mode' -- da pra alternar entre os dois pra comparar.
  2. Uma TIMELINE visual de keyframes (playhead arrastavel, clique pra adicionar, pips
     arrastaveis) no lugar do campo de texto JSON puro. O widget 'keyframes' continua
     existindo por baixo, no MESMO formato JSON do CrossViewWarp (so a edicao fica visual;
     workflows/kf lists continuam compativeis entre os dois nodes).

Escopo do preview 3D (importante, documentado tambem no tooltip do node):
  - A CAMERA (azimuth/elevation/distance/pivot) e 100% ao vivo: arrastar no viewer 3D
    atualiza os widgets na hora, sem rodar o node de novo.
  - A GEOMETRIA (depth_ratio, smooth_depth, invert_depth, hfov) e uma FOTO da ultima
    execucao: like o gizmo antigo, muda so quando o node roda de novo. Por isso o preview
    manda um numero pequeno de frames amostrados (nao o clipe inteiro) como PNG de
    profundidade (16-bit) + RGB, e o navegador reprojeta eles com a camera atual.
"""

import json
import logging
import os
import time
import shutil
from pathlib import Path

import numpy as np
import torch

try:
    from comfy.utils import ProgressBar
except Exception:  # standalone import (fora do ComfyUI)
    ProgressBar = None

try:
    import folder_paths
except Exception:
    folder_paths = None

# Rotas opcionais para o botao "Importar Gaussian" do widget web.
# O arquivo fica em ComfyUI/input/bruxos_gaussians e pode ser reaberto ao carregar o workflow.
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/bruxos/upload_gaussian")
    async def bruxos_upload_gaussian(request):
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "file":
            return web.json_response({"error": "campo 'file' ausente"}, status=400)
        original = os.path.basename(field.filename or "scene.ply")
        ext = os.path.splitext(original)[1].lower()
        if ext not in (".ply", ".splat", ".ksplat"):
            return web.json_response({"error": "use .ply, .splat ou .ksplat"}, status=400)
        root = folder_paths.get_input_directory() if folder_paths is not None else os.getcwd()
        out_dir = os.path.join(root, "bruxos_gaussians")
        os.makedirs(out_dir, exist_ok=True)
        safe = f"{int(time.time()*1000)}_{original}"
        out_path = os.path.join(out_dir, safe)
        with open(out_path, "wb") as f:
            while True:
                chunk = await field.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        rel = os.path.relpath(out_path, root).replace("\\", "/")
        return web.json_response({"filename": rel, "display_name": original})

    @PromptServer.instance.routes.get("/bruxos/gaussian")
    async def bruxos_get_gaussian(request):
        rel = request.query.get("filename", "")
        root = folder_paths.get_input_directory() if folder_paths is not None else os.getcwd()
        path = os.path.abspath(os.path.join(root, rel))
        if not path.startswith(os.path.abspath(root) + os.sep) or not os.path.isfile(path):
            return web.Response(status=404, text="Gaussian nao encontrado")
        return web.FileResponse(path)
except Exception:
    # Fora do ComfyUI (ou durante testes standalone), as rotas simplesmente nao existem.
    pass

MAGENTA = np.array([255, 0, 255], dtype=np.uint8)
SH_C0 = 0.28209479177387814  # sqrt(1/(4*pi)) -- coeficiente SH grau 0 (cor -> f_dc)

# Faixa de distancia da camera. Alargada (era 0.2..3.0) porque o view 3D estava
# "travando" perto dos limites -- agora da pra chegar bem mais perto e bem mais
# longe do pivo. Mantido como constante pra INPUT_TYPES e _sample_path baterem.
_DIST_MIN, _DIST_MAX = 0.05, 12.0


# =============================================================================
# Matematica de warp/orbit/keyframes -- port do ComfyUI-CrossViewWarp (Apache-2.0)
# =============================================================================

def _dist_scale(d):
    d = float(d)
    if d <= 1.0:
        return 0.45 + 0.55 * d
    return 1.0 + 0.25 * (d - 1.0)


def _warp_frame(rgb_ref, depth_ref, C_ref, C_tgt, fx_pix, splat, cx, cy):
    H, W = depth_ref.shape
    fy_pix = fx_pix
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_ref
    fin = np.isfinite(z) & (z > 0)
    thr = np.percentile(z[fin], 99.5) if fin.any() else 0
    fin = fin & (z < thr)
    Xc = np.stack([(u - W / 2.0) / fx_pix * z, (v - H / 2.0) / fy_pix * z, z], -1).reshape(-1, 3)
    Xw = (C_ref[:3, :3] @ Xc.T).T + C_ref[:3, 3]
    Ci = np.linalg.inv(C_tgt)
    Xd = (Ci[:3, :3] @ Xw.T).T + Ci[:3, 3]
    zt = Xd[:, 2]
    ui = np.round(Xd[:, 0] / zt * fx_pix + cx).astype(int)
    vi = np.round(Xd[:, 1] / zt * fy_pix + cy).astype(int)
    val = fin.ravel() & (zt > 0)
    order = np.argsort(-zt)
    sel = order[val[order]]
    tu0 = ui[sel]
    tv0 = vi[sel]
    cols = np.ascontiguousarray(rgb_ref.reshape(-1, 3)[sel])
    warp = np.tile(MAGENTA, (H, W, 1))
    flat = warp.reshape(-1, 3)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            tu = tu0 + dx
            tv = tv0 + dy
            ok = (tu >= 0) & (tu < W) & (tv >= 0) & (tv < H)
            flat[tv[ok] * W + tu[ok]] = cols[ok]
    return warp.astype(np.uint8)


def _look_at(eye, target, world_down=np.array([0.0, 1.0, 0.0])):
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    right = np.cross(world_down, f)
    right = right / (np.linalg.norm(right) + 1e-9)
    down = np.cross(f, right)
    C = np.eye(4)
    C[:3, 0], C[:3, 1], C[:3, 2], C[:3, 3] = right, down, f, eye
    return C


def _rodrigues(axis, ang):
    a = axis / (np.linalg.norm(axis) + 1e-9)
    c, s = np.cos(ang), np.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _depth_to_z(depth_bhw, invert, ratio=4.0):
    d = depth_bhw.astype(np.float64)
    if invert:
        d = -d
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    dn = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
    r = max(float(ratio), 1.01)
    return 1.0 / (1.0 / r + (1.0 - 1.0 / r) * dn)


def _wrap_deg(a):
    return ((a + 180.0) % 360.0) - 180.0


def _ease(t, mode):
    if mode == "ease_in_out":
        return 0.5 - 0.5 * np.cos(np.pi * t)
    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    return t


def _unwrap_seq(degs):
    out = [float(degs[0])]
    for d in degs[1:]:
        out.append(out[-1] + _wrap_deg(float(d) - out[-1]))
    return out


def _catmull(p0, p1, p2, p3, u):
    return 0.5 * ((2.0 * p1)
                  + (-p0 + p2) * u
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u * u
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u * u * u)


def _seg_value(vals, seg, u, smooth):
    p1, p2 = vals[seg], vals[seg + 1]
    if not smooth or len(vals) < 3:
        return p1 + (p2 - p1) * u
    p0 = vals[seg - 1] if seg > 0 else p1 + (p1 - p2)
    p3 = vals[seg + 2] if seg + 2 < len(vals) else p2 + (p2 - p1)
    return _catmull(p0, p1, p2, p3, u)


def _parse_keyframes(raw, frame_count):
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw or raw in ("[]", "null"):
        return []
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            "BruxosCrossViewWarp3D: 'keyframes' nao e um JSON valido (%s)." % exc
        ) from None
    if not isinstance(data, list):
        raise ValueError("BruxosCrossViewWarp3D: 'keyframes' precisa ser uma lista JSON.")
    out = []
    for i, kf in enumerate(data):
        if not isinstance(kf, dict):
            raise ValueError("BruxosCrossViewWarp3D: keyframe #%d nao e um objeto." % i)
        try:
            f = int(round(float(kf["f"])))
            az, el, dist = float(kf["az"]), float(kf["el"]), float(kf["dist"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "BruxosCrossViewWarp3D: keyframe #%d precisa de 'f', 'az', 'el', 'dist' numericos." % i
            ) from None
        if f < 1:
            raise ValueError("BruxosCrossViewWarp3D: keyframe #%d no frame %d (comeca em 1)." % (i, f))
        if f > frame_count:
            raise ValueError(
                "BruxosCrossViewWarp3D: keyframe #%d no frame %d, mas o clipe tem %d frames." % (i, f, frame_count))
        out.append((f, az, el, dist))
    out.sort(key=lambda k: k[0])
    seen = [k[0] for k in out]
    if len(set(seen)) != len(seen):
        raise ValueError("BruxosCrossViewWarp3D: dois keyframes no mesmo frame.")
    return out


def _prepare_path(kfs):
    return {
        "f": [k[0] for k in kfs],
        "az": _unwrap_seq([k[1] for k in kfs]),
        "el": [k[2] for k in kfs],
        "dist": [k[3] for k in kfs],
    }


def _sample_path(path, frame, easing, smooth):
    fs = path["f"]
    if frame <= fs[0]:
        return path["az"][0], path["el"][0], path["dist"][0]
    if frame >= fs[-1]:
        return path["az"][-1], path["el"][-1], path["dist"][-1]
    seg = 0
    for i in range(len(fs) - 1):
        if fs[i] <= frame <= fs[i + 1]:
            seg = i
            break
    u = _ease((frame - fs[seg]) / float(fs[seg + 1] - fs[seg]), easing)
    az = _wrap_deg(_seg_value(path["az"], seg, u, smooth))
    el = float(np.clip(_seg_value(path["el"], seg, u, smooth), -90.0, 90.0))
    dist = float(np.clip(_seg_value(path["dist"], seg, u, smooth), _DIST_MIN, _DIST_MAX))
    return az, el, dist


def _orbit_C_tgt(az_deg, el_deg, dist, pivot):
    R_orbit = _rot_y(np.radians(-az_deg)) @ _rot_x(np.radians(-el_deg))
    eye = pivot + dist * (R_orbit @ (-pivot))
    return _look_at(eye, pivot)


def _orbit_view_image(azimuth, elevation, distance, size=512, kfs=None, smooth=False):
    """Diagrama simples (documenta o setup de camera); versao enxuta do gizmo do
    CrossViewWarp -- so o essencial pra registrar a pose/trajeto, sem toda a UI de
    zonas de cobertura (isso agora vive no viewer 3D interativo do node)."""
    from PIL import Image, ImageDraw
    W = H = size
    cx, cy = W / 2.0, H / 2.0
    R = size * 0.32
    img = Image.new("RGB", (W, H), (27, 27, 31))
    d = ImageDraw.Draw(img)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(120, 124, 138), width=2)

    def pt(a_deg, e_deg):
        a, e = np.radians(a_deg), np.radians(e_deg)
        x = np.cos(e) * np.sin(a)
        y = np.sin(e)
        return cx + R * x, cy - R * y

    if kfs and len(kfs) >= 2:
        az_un = _unwrap_seq([kf[1] for kf in kfs])
        els = [kf[2] for kf in kfs]
        SUB = 24
        prev = None
        for seg in range(len(kfs) - 1):
            for s in range(SUB + 1):
                u = s / SUB
                a = _wrap_deg(_seg_value(az_un, seg, u, smooth))
                e = _seg_value(els, seg, u, smooth)
                x1, y1 = pt(a, e)
                if prev is not None:
                    d.line([prev[0], prev[1], x1, y1], fill=(95, 206, 128), width=3)
                prev = (x1, y1)
        for f_no, kaz, kel, kdist in kfs:
            kx, ky = pt(kaz, kel)
            d.ellipse([kx - 6, ky - 6, kx + 6, ky + 6], fill=(95, 206, 128), outline=(255, 255, 255))
        label = f"{len(kfs)} keyframes  f{kfs[0][0]}-{kfs[-1][0]}"
    else:
        x, y = pt(azimuth, elevation)
        d.line([cx, cy, x, y], fill=(120, 190, 255), width=2)
        d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(120, 190, 255), outline=(255, 255, 255))
        label = f"az {azimuth:+.0f}  el {elevation:+.0f}  dist {distance:.2f}x"

    d.text((8, 6), label, fill=(255, 255, 100))
    d.text((8, H - 20), "BruxosCrossViewWarp3D -- abra o viewer 3D no node p/ orbitar de verdade",
           fill=(160, 160, 170))
    return np.asarray(img, dtype=np.uint8)


# =============================================================================
# Extras deste node: PLY sintetico (formato SHARP/3DGS) + PNGs de preview
# =============================================================================

def _write_gaussian_ply(path, xyz, rgb01, scale_world, opacity=0.92):
    """Escreve um .ply binario no MESMO layout que o SHARP/comfyui-GaussianViewer
    esperam (x,y,z, f_dc_0-2, opacity, scale_0-2, rot_0-3), so que com gaussians
    ISOTROPICAS sinteticas (um "gaussian splat" aproximado a partir de depth+rgb,
    sem rede neural) -- serve pra COMPARAR contra a nuvem de pontos crua, nao e
    uma reconstrucao 3DGS de verdade.

    xyz: (N,3) float64. rgb01: (N,3) float em [0,1]. scale_world: (N,) ou escalar,
    desvio-padrao da gaussian em unidades de mundo (mesmas de xyz).
    """
    N = int(xyz.shape[0])
    if N == 0:
        raise ValueError("BruxosCrossViewWarp3D: nenhum ponto valido pra exportar no .ply")

    f_dc = (np.clip(rgb01, 0.0, 1.0).astype(np.float64) - 0.5) / SH_C0

    opac = np.clip(float(opacity), 1e-4, 1.0 - 1e-4)
    opacity_logit = np.full((N, 1), np.log(opac / (1.0 - opac)), dtype=np.float64)

    if np.isscalar(scale_world):
        s = np.full((N, 3), float(scale_world), dtype=np.float64)
    else:
        s = np.tile(np.asarray(scale_world, dtype=np.float64).reshape(-1, 1), (1, 3))
    scale_logit = np.log(np.clip(s, 1e-6, None))

    quat = np.zeros((N, 4), dtype=np.float64)
    quat[:, 0] = 1.0  # identidade (w,x,y,z) -- isotropico, rotacao nao importa

    attrs = np.concatenate([xyz, f_dc, opacity_logit, scale_logit, quat], axis=1).astype(np.float32)

    names = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    for n in names:
        header += f"property float {n}\n"
    header += "end_header\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(attrs).tobytes())


def _save_png_rgb(rgb_uint8, path):
    from PIL import Image
    Image.fromarray(rgb_uint8, mode="RGB").save(path, compress_level=3)


def _save_png_depth16(z01, path):
    """z01: float [0,1] (ja normalizado) -> PNG RGB comum com os 16 bits EMPACOTADOS
    em 2 canais de 8 bits (R = byte alto, G = byte baixo, B = 0). Um PNG real de
    16-bit-por-canal ("I;16") nao da pra ler no navegador com precisao -- canvas/
    getImageData sempre trunca pra 8 bits por canal -- entao o pacote R+G e o jeito
    que garante os 65536 niveis chegarem inteiros no JS (reconstroi com r*256+g)."""
    from PIL import Image
    z16 = np.clip(z01 * 65535.0, 0, 65535).astype(np.uint32)
    hi = (z16 >> 8).astype(np.uint8)
    lo = (z16 & 0xFF).astype(np.uint8)
    packed = np.stack([hi, lo, np.zeros_like(hi)], axis=-1)
    Image.fromarray(packed, mode="RGB").save(path, compress_level=3)


def _pick_preview_indices(B, n_preview):
    if B <= 1:
        return [0]
    n = max(1, min(int(n_preview), B))
    if n == 1:
        return [0]
    idxs = sorted(set(int(round(i * (B - 1) / (n - 1))) for i in range(n)))
    return idxs


class BruxosCrossViewWarp3D:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_mode": (["inputs", "import_gaussian"], {"default": "inputs",
                    "tooltip": "inputs = reconstrui frames+depth. import_gaussian = abre .ply/.splat/.ksplat e dispensa as entradas."}),
                "azimuth": ("FLOAT", {"default": -30.0, "min": -180.0, "max": 180.0, "step": 1.0, "tooltip": "Rotação horizontal da câmera ao redor da cena. Valores negativos giram para a esquerda e positivos para a direita."}),
                "elevation": ("FLOAT", {"default": 20.0, "min": -90.0, "max": 90.0, "step": 1.0, "tooltip": "Rotação vertical da câmera. Valores positivos olham de cima e negativos olham de baixo."}),
                "distance": ("FLOAT", {"default": 1.0, "min": _DIST_MIN, "max": _DIST_MAX, "step": 0.05,
                    "tooltip": "Zoom/dolly do viewport. A roda do mouse atualiza este valor."}),
                "hfov": ("FLOAT", {"default": 50.0, "min": 20.0, "max": 120.0, "step": 1.0, "tooltip": "Campo de visão horizontal. Valores baixos fecham a lente; valores altos criam uma visão mais grande-angular."}),
                "head_bias": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.02, "tooltip": "Desloca o enquadramento vertical do warp para compensar a posição da cabeça ou do assunto."}),
                "depth_ratio": ("FLOAT", {"default": 6.0, "min": 1.5, "max": 1000.0, "step": 0.5, "tooltip": "Controla a separação entre regiões próximas e distantes do mapa de profundidade. Valores maiores aumentam o efeito 3D."}),
                "smooth_depth": ("BOOLEAN", {"default": False, "tooltip": "Suaviza o mapa de profundidade para reduzir ruído, tremulação e superfícies quebradas."}),
                "invert_depth": ("BOOLEAN", {"default": False, "tooltip": "Inverte o mapa de profundidade quando perto e longe estão interpretados ao contrário."}),
            },
            "optional": {
                "frames": ("IMAGE", {"tooltip": "Obrigatorio somente em source_mode=inputs."}),
                "depth": ("IMAGE", {"tooltip": "Obrigatorio somente em source_mode=inputs."}),
                "gaussian_file": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Preenchido automaticamente pelo botao Importar Gaussian do viewport."}),
                "roll_lock": ("BOOLEAN", {"default": True, "tooltip": "Mantém a câmera nivelada, evitando que o horizonte incline durante a órbita."}),
                "pivot_override": ("BOOLEAN", {"default": True, "tooltip": "Usa os valores manuais de pivô como centro de rotação da câmera."}),
                "pivot_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01, "tooltip": "Move o centro de rotação para a esquerda ou para a direita."}),
                "pivot_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01, "tooltip": "Move o centro de rotação para cima ou para baixo."}),
                "pivot_z": ("FLOAT", {"default": 1.05, "min": 0.01, "max": 1000.0, "step": 0.01, "tooltip": "Move o centro de rotação para frente ou para trás na profundidade."}),
                "use_keyframes": ("BOOLEAN", {"default": False, "tooltip": "Ativa a animação de câmera usando os keyframes criados na timeline."}),
                "frame_count": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "Define a duração da animação em frames. Em Gaussian importado, use por exemplo 81 para animar a câmera por 81 frames."}),
                "keyframes": ("STRING", {"default": "", "multiline": False, "tooltip": "Dados internos da timeline em JSON. Normalmente você edita pelos botões Key e pela barra de tempo."}),
                "interp_motion": (["linear", "ease_in_out", "ease_in", "ease_out"], {"default": "linear", "tooltip": "Define a aceleração entre keyframes: linear, suave no início e fim, entrada suave ou saída suave."}),
                "interpolation": (["linear", "smooth"], {"default": "linear", "tooltip": "Define o formato do trajeto da câmera. Linear cria segmentos retos; smooth suaviza a curva entre os keyframes."}),
                "render_mode": (["pointcloud", "gaussian"], {"default": "gaussian",
                    "tooltip": "pointcloud mostra pontos quadrados simples. gaussian usa splats suaves renderizados pela GPU (WebGL), mais próximos da aparência 3DGS."}),
                "preview_quality": (["leve", "equilibrado", "completo"], {"default": "equilibrado",
                    "tooltip": "Qualidade do visualizador GPU. Leve reduz resolução e quantidade de splats; Equilibrado combina velocidade e qualidade; Completo usa resolução total e mais splats."}),
                "point_size": ("FLOAT", {"default": 1.6, "min": 0.3, "max": 12.0, "step": 0.1, "tooltip": "Multiplica o tamanho visual dos pontos ou splats no preview. Não altera o arquivo original."}),
                "preview_frames": ("INT", {"default": 6, "min": 1, "max": 16, "step": 1, "tooltip": "Quantidade de frames enviados ao visualizador ao reconstruir frames+depth. Mais frames usam mais memória."}),
                "play_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                    "tooltip": "FPS do botao Play da timeline no viewport."}),
                "loop_playback": ("BOOLEAN", {"default": True, "tooltip": "Quando ativo, o Play volta ao primeiro frame após chegar ao final."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("warp", "orbit_view")
    OUTPUT_TOOLTIPS = (
        "O video de controle do warp (magenta = buracos de desoclusao).",
        "Diagrama simples da pose/trajeto de camera (o viewer de verdade e o widget 3D do node).",
    )
    FUNCTION = "build"
    CATEGORY = "Bruxos do VFX/CrossView"
    OUTPUT_NODE = True

    def build(self, source_mode, azimuth, elevation, distance, hfov, head_bias, depth_ratio,
              smooth_depth, invert_depth, frames=None, depth=None, gaussian_file="",
              roll_lock=True, pivot_override=True, pivot_x=0.0, pivot_y=0.0, pivot_z=1.05,
              use_keyframes=False, frame_count=0, keyframes="", interp_motion="linear",
              interpolation="linear", render_mode="gaussian", preview_quality="equilibrado", point_size=1.6,
              preview_frames=6, play_fps=24.0, loop_playback=True):
        if source_mode == "import_gaussian":
            if not gaussian_file:
                raise ValueError("BruxosCrossViewWarp3D: clique em 'Importar Gaussian' no viewport.")
            root = folder_paths.get_input_directory() if folder_paths is not None else os.getcwd()
            full = os.path.abspath(os.path.join(root, gaussian_file))
            if not full.startswith(os.path.abspath(root) + os.sep) or not os.path.isfile(full):
                raise ValueError(f"BruxosCrossViewWarp3D: arquivo Gaussian nao encontrado: {gaussian_file}")
            total = max(1, int(frame_count) if frame_count else 1)
            meta = {
                "source_mode": "import_gaussian", "gaussian_file": gaussian_file,
                "frame_count": total, "sampled_frames": [1],
                "render_mode": render_mode, "preview_quality": preview_quality, "point_size": float(point_size),
                "play_fps": float(play_fps), "loop_playback": bool(loop_playback),
                "pivot": [float(pivot_x), float(pivot_y), float(pivot_z)],
            }
            ui_payload = {"bx_cv3d_import": [gaussian_file], "bx_cv3d_meta": [json.dumps(meta)]}
            black = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            orbit = _orbit_view_image(azimuth, elevation, distance)
            orbit_t = torch.from_numpy(orbit.astype(np.float32) / 255.0)[None]
            return {"ui": ui_payload, "result": (black, orbit_t)}

        if frames is None or depth is None:
            raise ValueError("BruxosCrossViewWarp3D: em source_mode=inputs, conecte frames e depth.")
        rgb = (frames.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        B, H, W = rgb.shape[:3]
        depth_bhw = depth.clamp(0, 1).mean(dim=-1).cpu().numpy()
        if smooth_depth:
            import cv2
            for i in range(B):
                d32 = cv2.medianBlur(depth_bhw[i].astype(np.float32), 3)
                try:
                    d32 = cv2.ximgproc.guidedFilter(rgb[i], d32, radius=8, eps=1e-3)
                except Exception:
                    d32 = cv2.bilateralFilter(d32, 9, 0.1, 9.0)
                depth_bhw[i] = d32
        z = _depth_to_z(depth_bhw, invert_depth, depth_ratio)  # [B,H,W], range ~[1, depth_ratio]

        fx = W / (2.0 * np.tan(np.radians(hfov) / 2.0))
        cc = W / 2.0
        cch = H / 2.0

        z0 = z[0]
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        central = np.zeros_like(z0, bool)
        central[H // 8: 4 * H // 5, W // 5: 4 * W // 5] = True
        fin = np.isfinite(z0) & (z0 > 0) & central
        fin &= z0 < np.percentile(z0[fin], 95.0)
        fg = fin & (z0 < np.percentile(z0[fin], 50.0))
        Xw0 = np.stack([(uu - cc) / fx * z0, (vv - cch) / fx * z0, z0], -1)
        pivot = np.median(Xw0[fin], axis=0)
        if pivot_override:
            pivot = np.array([pivot_x, pivot_y, pivot_z], dtype=np.float64)

        C_ref = np.eye(4)

        if use_keyframes and frame_count and frame_count != B:
            logging.warning(
                "BruxosCrossViewWarp3D: frame_count e %d mas o clipe tem %d frames.",
                frame_count, B)
        kfs = _parse_keyframes(keyframes, B) if use_keyframes else []
        path = _prepare_path(kfs) if kfs else None
        keyframing = bool(len(kfs) >= 2 and B > 1)
        smooth_path = (interpolation == "smooth")
        if keyframing:
            mid_az, mid_el, mid_dist = _sample_path(path, (B + 1) // 2, interp_motion, smooth_path)
        elif kfs:
            mid_az, mid_el, mid_dist = kfs[0][1], kfs[0][2], kfs[0][3]
        else:
            mid_az, mid_el, mid_dist = azimuth, elevation, distance
        C_tgt = _orbit_C_tgt(mid_az, mid_el, mid_dist, pivot)

        fgc = Xw0[fg].mean(0)
        pu = cc + fgc[0] / fgc[2] * fx
        subj = fin & (np.abs(z0 - fgc[2]) < 0.3 * fgc[2]) & (np.abs(uu - pu) < 0.3 * W)
        if subj.sum() < 500:
            subj = fg

        def _lean(P2):
            P2 = P2 - P2.mean(0)
            _ev, V2 = np.linalg.eigh(P2.T @ P2)
            v2 = V2[:, -1]
            if v2[1] > 0:
                v2 = -v2
            dom = np.sqrt(_ev[-1] / max(_ev[-2], 1e-9))
            ang = np.arctan2(v2[0], -v2[1])
            if dom < 1.3 or abs(ang) > np.radians(45):
                return None
            return ang

        def _lean_in(C):
            Ci2 = np.linalg.inv(C)
            Xd2 = (Ci2[:3, :3] @ Xw0[subj].T).T + Ci2[:3, 3]
            m2 = Xd2[:, 2] > 0
            return _lean(np.stack([Xd2[m2, 0] / Xd2[m2, 2], Xd2[m2, 1] / Xd2[m2, 2]], -1))

        def _wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        th_src = _lean(np.stack([uu[subj].astype(np.float64), vv[subj].astype(np.float64)], -1))
        th_tgt = _lean_in(C_tgt)
        applied_droll = 0.0
        if roll_lock and th_src is not None and th_tgt is not None:
            droll = float(np.clip(_wrap(th_tgt - th_src), -np.radians(35), np.radians(35)))
            err0 = abs(_wrap(th_tgt - th_src))
            for cand in (droll, -droll):
                C_try = C_tgt.copy()
                C_try[:3, :3] = _rodrigues(C_tgt[:3, 2], cand) @ C_tgt[:3, :3]
                th_try = _lean_in(C_try)
                if th_try is not None and abs(_wrap(th_try - th_src)) < err0 - 1e-6:
                    applied_droll = cand
                    C_tgt = C_try
                    break

        cx_eff = cc
        cy_eff = cch + head_bias * H

        pbar = ProgressBar(B) if ProgressBar is not None else None
        warp_frames = []
        for i in range(B):
            if keyframing:
                fr_az, fr_el, fr_dist = _sample_path(path, i + 1, interp_motion, smooth_path)
                C_tgt_i = _orbit_C_tgt(fr_az, fr_el, fr_dist, pivot)
                if applied_droll != 0.0:
                    C_tgt_i = C_tgt_i.copy()
                    C_tgt_i[:3, :3] = _rodrigues(C_tgt_i[:3, 2], applied_droll) @ C_tgt_i[:3, :3]
            else:
                C_tgt_i = C_tgt
            warp_frames.append(_warp_frame(rgb[i], z[i], C_ref, C_tgt_i, fx, 2, cx_eff, cy_eff))
            if pbar is not None:
                pbar.update(1)
        warp = np.stack(warp_frames, 0)

        warp_t = torch.from_numpy(warp.astype(np.float32) / 255.0)
        orbit = _orbit_view_image(mid_az, mid_el, mid_dist,
                                   kfs=kfs if keyframing else None, smooth=smooth_path)
        orbit_t = torch.from_numpy(orbit.astype(np.float32) / 255.0)[None]

        ui_payload = self._build_preview_payload(
            rgb, z, W, H, B, fx, cc, cch, hfov, pivot, frame_count,
            render_mode, preview_quality, point_size, preview_frames, play_fps, loop_playback)

        return {"ui": ui_payload, "result": (warp_t, orbit_t)}

    # -- preview 3D: PNGs (rgb+depth16) e/ou .ply sinteticos p/ o widget JS --------
    def _build_preview_payload(self, rgb, z, W, H, B, fx, cc, cch, hfov, pivot,
                                frame_count, render_mode, preview_quality, point_size, preview_frames,
                                play_fps=24.0, loop_playback=True):
        try:
            tmp_dir = folder_paths.get_temp_directory() if folder_paths is not None else None
        except Exception:
            tmp_dir = None
        if not tmp_dir:
            logging.warning("BruxosCrossViewWarp3D: folder_paths indisponivel -- preview 3D desativado.")
            return {}
        os.makedirs(tmp_dir, exist_ok=True)

        MAXSIDE = 512  # downsample do preview (nao do output 'warp', que fica intacto)
        idxs = _pick_preview_indices(B, preview_frames)
        z_lo, z_hi = float(z.min()), float(z.max())
        z_span = max(z_hi - z_lo, 1e-6)

        stamp = f"{int(time.time() * 1000)}_{id(self) & 0xffff:04x}"
        frames_out, depth_out, ply_out = [], [], []

        for n, idx in enumerate(idxs):
            scale = min(1.0, MAXSIDE / max(W, H))
            w2, h2 = max(1, int(round(W * scale))), max(1, int(round(H * scale)))

            rgb_i = rgb[idx]
            z_i = z[idx]
            if scale < 1.0:
                import cv2
                rgb_i = cv2.resize(rgb_i, (w2, h2), interpolation=cv2.INTER_AREA)
                z_i = cv2.resize(z_i.astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
            z01 = (z_i - z_lo) / z_span

            fname_rgb = f"bx_cv3d_{stamp}_{n}_rgb.png"
            fname_z = f"bx_cv3d_{stamp}_{n}_z16.png"
            try:
                _save_png_rgb(rgb_i, os.path.join(tmp_dir, fname_rgb))
                _save_png_depth16(z01, os.path.join(tmp_dir, fname_z))
                frames_out.append({"filename": fname_rgb, "subfolder": "", "type": "temp"})
                depth_out.append({"filename": fname_z, "subfolder": "", "type": "temp"})
            except Exception as e:
                logging.warning("BruxosCrossViewWarp3D: falha salvando preview PNG: %s", e)
                continue

            if render_mode == "gaussian":
                try:
                    fname_ply = f"bx_cv3d_{stamp}_{n}.ply"
                    self._write_ply_for_frame(
                        rgb_i, z_i, w2, h2, fx * scale, point_size,
                        os.path.join(tmp_dir, fname_ply))
                    ply_out.append({"filename": fname_ply, "subfolder": "", "type": "temp"})
                except Exception as e:
                    logging.warning("BruxosCrossViewWarp3D: falha gerando .ply: %s", e)

        meta = {
            "W": W, "H": H, "B": B, "fx": float(fx), "hfov": float(hfov),
            "pivot": [float(pivot[0]), float(pivot[1]), float(pivot[2])],
            "z_lo": z_lo, "z_hi": z_hi,
            "frame_count": int(frame_count) if frame_count else B,
            "sampled_frames": [int(i) + 1 for i in idxs],  # 1-based, casa com 'keyframes'
            "source_mode": "inputs",
            "render_mode": render_mode,
            "point_size": float(point_size),
            "play_fps": float(play_fps),
            "loop_playback": bool(loop_playback),
        }
        return {
            "bx_cv3d_frames": frames_out,
            "bx_cv3d_depth": depth_out,
            "bx_cv3d_ply": ply_out,
            "bx_cv3d_meta": [json.dumps(meta)],
        }

    def _write_ply_for_frame(self, rgb_i, z_i, w2, h2, fx_scaled, point_size, out_path):
        cc2, cch2 = w2 / 2.0, h2 / 2.0
        uu, vv = np.meshgrid(np.arange(w2), np.arange(h2))
        X = (uu - cc2) / fx_scaled * z_i
        Y = (vv - cch2) / fx_scaled * z_i
        xyz = np.stack([X, Y, z_i], -1).reshape(-1, 3).astype(np.float64)
        colors = (rgb_i.reshape(-1, 3).astype(np.float64)) / 255.0
        # gaussian isotropica ~ 'point_size' pixels-fonte de raio, em unidades de mundo
        scale_world = np.clip(point_size * (z_i.reshape(-1) / fx_scaled), 1e-5, None)
        _write_gaussian_ply(out_path, xyz, colors, scale_world)


NODE_CLASS_MAPPINGS = {"BruxosCrossViewWarp3D": BruxosCrossViewWarp3D}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosCrossViewWarp3D": "Ângulos 3D / Gaussian (Bruxos)"}
