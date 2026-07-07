# -*- coding: utf-8 -*-
"""
ComfyUI-Bruxos-FaceFusion — nodes de face swap local (ONNX) pro pipeline Bruxos do VFX.
Reconstrução dos nodes de huygiatrng/Facefusion_comfyui, estilo Bruxos:
- 100% local (sem modo API / httpx / comfy_api_nodes)
- IMAGE batch in/out (compatível com BruxosLoadVideo/BruxosSaveVideo e Bernini)
- Saída MASK dos rostos (liga direto no region_mask do Bernini Infinity)
- Filtro de conteúdo do upstream mantido ativo
"""
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import torch
import cv2

from .facefusion.engine import (
    detect_faces, get_local_swapper, get_face_occluder,
    get_face_parser, tensor_to_cv2, cv2_to_tensor, find_matching_faces,
)
from .facefusion.filtro import analyse_frame, blur_frame

SWAPPER_MODELS = [
    'hyperswap_1a_256', 'hyperswap_1b_256', 'hyperswap_1c_256',
    'ghost_1_256', 'ghost_2_256', 'ghost_3_256',
    'hififace_unofficial_256',
    'inswapper_128', 'inswapper_128_fp16',
    'blendswap_256', 'simswap_256', 'simswap_unofficial_512', 'uniface_256',
]
DETECTOR_MODELS = ['scrfd', 'retinaface', 'yolo_face', 'yunet', 'many']
SORT_ORDERS = ['large-small', 'small-large', 'left-right', 'right-left',
               'top-bottom', 'bottom-top', 'best-worst', 'worst-best']
PIXEL_BOOSTS = ['256x256', '512x512', '768x768', '1024x1024']
OCCLUDERS = ['none', 'xseg_1', 'xseg_2', 'xseg_3']
PARSERS = ['none', 'bisenet_resnet_18', 'bisenet_resnet_34']


def _parse_csv(text: str) -> Optional[List[str]]:
    items = [t.strip() for t in (text or '').split(',') if t.strip()]
    return items or None


def _parse_padding(text: str) -> Tuple[int, int, int, int]:
    try:
        vals = [int(v.strip()) for v in (text or '0,0,0,0').split(',')]
        while len(vals) < 4:
            vals.append(0)
        return tuple(vals[:4])
    except Exception:
        return (0, 0, 0, 0)


def _faces_to_mask(faces, h: int, w: int, grow: int, blur: int) -> np.ndarray:
    """Máscara float32 [H,W] a partir dos bboxes dos rostos (com grow + feather)."""
    mask = np.zeros((h, w), dtype=np.float32)
    for f in faces:
        x1, y1, x2, y2 = [int(round(float(v))) for v in f['bbox'][:4]]
        x1, y1 = max(0, x1 - grow), max(0, y1 - grow)
        x2, y2 = min(w, x2 + grow), min(h, y2 + grow)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    if blur > 0:
        k = blur * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def _nsfw_gate(frames_cv2: List[np.ndarray], source_cv2: Optional[np.ndarray]) -> bool:
    """Amostra 1º/meio/último frame + source. True = bloquear (igual ao upstream)."""
    if source_cv2 is not None and analyse_frame(source_cv2):
        return True
    if frames_cv2:
        n = len(frames_cv2)
        for idx in sorted({0, n // 2, n - 1}):
            if analyse_frame(frames_cv2[idx]):
                return True
    return False


class BruxosFaceFusionSwap:
    """Swap de rosto local (imagem única ou batch/vídeo), com máscaras combináveis."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'source_face': ('IMAGE', {'tooltip': 'Imagem do rosto NOVO (quem vai entrar). Basta 1 foto nitida, de frente. So o 1o rosto detectado nela e usado (use face_position se tiver mais de um).'}),
                'target_images': ('IMAGE', {'tooltip': 'Video/imagem ALVO (onde o rosto sera trocado). Aceita 1 frame ou um batch inteiro (video). O tamanho da saida acompanha este.'}),
                'face_swapper_model': (SWAPPER_MODELS, {'default': 'hyperswap_1c_256', 'tooltip': 'Modelo de troca. hyperswap_1c_256 = melhor semelhanca (recomendado); inswapper_128_fp16 = mais rapido/leve; ghost_* = Apache-2.0 (uso comercial). Baixa sozinho no 1o uso.'}),
                'face_detector_model': (DETECTOR_MODELS, {'default': 'scrfd', 'tooltip': 'Detector de rostos. scrfd e um bom padrao. yolo_face/retinaface como alternativas; many roda varios e junta (mais lento).'}),
                'pixel_boost': (PIXEL_BOOSTS, {'default': '512x512', 'tooltip': 'Resolucao interna do rosto trocado. Maior = mais detalhe e nitidez, porem mais lento e mais VRAM. 512x512 e um bom meio-termo.'}),
                'face_mask_blur': ('FLOAT', {'default': 0.3, 'min': 0.0, 'max': 1.0, 'step': 0.05, 'tooltip': 'Suaviza a borda da mascara do rosto colado (0..1). Maior = emenda mais macia com a pele em volta; muito alto pode borrar o contorno.'}),
                'face_selector_mode': (['one', 'many', 'reference'], {'default': 'one', 'tooltip': 'Quais rostos do alvo trocar. one = so um (o de face_position); many = todos os rostos do frame; reference = so quem parecer com reference_face (bom p/ cena com varias pessoas).'}),
                'face_position': ('INT', {'default': 0, 'min': 0, 'max': 99, 'tooltip': '[modo one] Indice do rosto a trocar, na ordem de sort_order. 0 = o primeiro (ex.: o maior). Use se houver mais de um rosto no frame.'}),
                'sort_order': (SORT_ORDERS, {'default': 'large-small', 'tooltip': 'Ordem em que os rostos sao listados (define quem e o indice 0). large-small = do maior pro menor (mais estavel); left-right, etc. p/ posicao fixa em cena.'}),
                'score_threshold': ('FLOAT', {'default': 0.3, 'min': 0.05, 'max': 1.0, 'step': 0.05, 'tooltip': 'Confianca minima pra aceitar um rosto (0..1). Abaixe (0.3-0.4) se rostos nao forem detectados; suba se estiver pegando rosto falso.'}),
                'reference_distance': ('FLOAT', {'default': 0.6, 'min': 0.05, 'max': 1.5, 'step': 0.05, 'tooltip': '[modo reference] Quao parecido com o reference_face o rosto precisa ser p/ ser trocado. Menor = mais rigoroso (so quem e muito igual); maior = mais permissivo.'}),
                'use_occlusion_mask': ('BOOLEAN', {'default': False, 'tooltip': 'Liga a mascara de OCLUSAO (xseg): preserva o que passa na frente do rosto (mao, cabelo, microfone). Ligue se algo cobre o rosto na cena.'}),
                'face_occluder_model': (OCCLUDERS, {'default': 'xseg_1', 'tooltip': 'Modelo de oclusao usado quando use_occlusion_mask esta ligado. xseg_1 costuma bastar.'}),
                'use_region_mask': ('BOOLEAN', {'default': False, 'tooltip': 'Liga a mascara por REGIAO (bisenet): recorta so partes do rosto (pele, nariz, boca...) definidas em face_mask_regions. Melhora a borda no cabelo/testa.'}),
                'face_parser_model': (PARSERS, {'default': 'bisenet_resnet_34', 'tooltip': 'Modelo de segmentacao facial usado quando use_region_mask esta ligado. bisenet_resnet_34 = melhor; _18 = mais leve.'}),
                'use_area_mask': ('BOOLEAN', {'default': False, 'tooltip': 'Liga a mascara por AREA: limita a troca a zonas amplas do rosto (upper-face/lower-face/mouth) definidas em face_mask_areas.'}),
                'face_mask_areas': ('STRING', {'default': 'upper-face,lower-face,mouth', 'tooltip': '[use_area_mask] Areas do rosto a incluir, separadas por virgula: upper-face, lower-face, mouth.'}),
                'face_mask_regions': ('STRING', {'default': 'skin,nose,mouth,upper-lip,lower-lip', 'tooltip': '[use_region_mask] Regioes do bisenet a incluir, separadas por virgula (ex.: skin, nose, mouth, upper-lip, lower-lip, eyes, brows, hair).'}),
                'face_mask_padding': ('STRING', {'default': '0,0,0,0', 'tooltip': 'Folga da mascara nas bordas, em % do rosto: top,right,bottom,left. Ex.: 0,0,0,0. Aumente p/ pegar mais testa/queixo.'}),
                'out_mask_grow': ('INT', {'default': 0, 'min': 0, 'max': 256, 'tooltip': 'So afeta a saida MASK (slot face_mask), nao o swap. Dilata a caixa dos rostos em pixels antes de gerar a mascara.'}),
                'out_mask_blur': ('INT', {'default': 6, 'min': 0, 'max': 128, 'tooltip': 'So afeta a saida MASK (slot face_mask). Suaviza a borda (feather) da mascara de rostos que sai pro Bernini/compose.'}),
                'max_workers': ('INT', {'default': 4, 'min': 1, 'max': 16, 'tooltip': 'Quantos frames processar em paralelo (video). Mais = mais rapido, porem mais VRAM/CPU. Baixe p/ 1-2 se estourar memoria.'}),
                'device': (['auto', 'cuda', 'cpu'], {'default': 'auto', 'tooltip': 'Onde rodar os modelos ONNX. auto = usa a GPU (CUDA) se o onnxruntime tiver; senao CPU. cuda = forca GPU (se nao houver, avisa no console e usa CPU). cpu = forca CPU. Se cair na CPU sem querer, e conflito de pacote onnxruntime (veja o aviso no console).'}),
            },
            'optional': {
                'reference_face': ('IMAGE', {'tooltip': '[modo reference] Rosto de referencia p/ escolher QUEM trocar no alvo. Se vazio, usa o proprio source_face como referencia.'}),
            },
        }

    RETURN_TYPES = ('IMAGE', 'MASK', 'INT')
    RETURN_NAMES = ('images', 'face_mask', 'frames_with_face')
    OUTPUT_TOOLTIPS = (
        'Video/imagem com o rosto trocado (mesmo tamanho do target_images).',
        'Mascara dos rostos por frame (branco = rosto). Liga direto no region_mask do Bernini Infinity ou num compose.',
        'Quantos frames tiveram rosto trocado (util p/ conferir se o detector pegou o alvo).',
    )
    FUNCTION = 'process'
    CATEGORY = 'Bruxos do VFX/Face'
    DESCRIPTION = (
        'Troca de rosto LOCAL (ONNX, sem API). Poe o rosto de source_face nos target_images (1 frame ou video inteiro). '
        'Mascaras combinaveis: box (sempre) + oclusao (xseg, preserva mao/cabelo na frente) + regiao (bisenet, borda melhor) + area. '
        'Modos de selecao: one (um rosto), many (todos), reference (so quem parece com reference_face). '
        'Sai tambem uma MASK dos rostos pronta p/ ligar no region_mask do Bernini Infinity. '
        'Requer onnxruntime-gpu/opencv; os modelos .onnx baixam sozinhos no 1o uso em models/facefusion/.'
    )

    def process(self, source_face, target_images, face_swapper_model, face_detector_model,
                pixel_boost, face_mask_blur, face_selector_mode, face_position, sort_order,
                score_threshold, reference_distance, use_occlusion_mask, face_occluder_model,
                use_region_mask, face_parser_model, use_area_mask, face_mask_areas,
                face_mask_regions, face_mask_padding, out_mask_grow, out_mask_blur,
                max_workers, device='auto', reference_face=None):

        from .facefusion.engine.runtime import set_device
        set_device(device)

        if target_images.dim() == 3:
            target_images = target_images.unsqueeze(0)
        n, h, w = target_images.shape[0], target_images.shape[1], target_images.shape[2]

        source_cv2 = tensor_to_cv2(source_face[0:1] if source_face.dim() == 4 else source_face)
        frames_cv2 = [tensor_to_cv2(target_images[i:i + 1]) for i in range(n)]

        empty_mask = torch.zeros((n, h, w), dtype=torch.float32)

        # Filtro de conteúdo (mesma amostragem do upstream: 1º/meio/último + source)
        if _nsfw_gate(frames_cv2, source_cv2):
            print('[Bruxos FaceFusion] Conteúdo bloqueado pelo filtro — saída borrada.')
            blurred = [cv2_to_tensor(blur_frame(f)).squeeze(0)[..., :3] for f in frames_cv2]
            return (torch.stack(blurred), empty_mask, 0)

        # Tipos de máscara de blend (box sempre; extras combináveis)
        mask_types = ['box']
        occluder_model = face_occluder_model if (use_occlusion_mask and face_occluder_model != 'none') else None
        parser_model = face_parser_model if (use_region_mask and face_parser_model != 'none') else None
        if occluder_model:
            mask_types.append('occlusion')
        if parser_model:
            mask_types.append('region')
        if use_area_mask:
            mask_types.append('area')

        areas = _parse_csv(face_mask_areas)
        regions = _parse_csv(face_mask_regions)
        padding = _parse_padding(face_mask_padding)

        # Rosto(s) de referência (selector 'reference')
        ref_faces = None
        if face_selector_mode == 'reference':
            ref_img = reference_face if reference_face is not None else source_face
            ref_cv2 = tensor_to_cv2(ref_img[0:1] if ref_img.dim() == 4 else ref_img)
            ref_detected = detect_faces(ref_cv2, score_threshold, sort_order, face_detector_model)
            if not ref_detected:
                print('[Bruxos FaceFusion] Nenhum rosto na referência — caindo pra modo "one".')
                face_selector_mode = 'one'
            else:
                ref_faces = ref_detected

        # Pré-carrega swapper/occluder/parser 1x (thread-safe pro executor)
        swapper = get_local_swapper(face_swapper_model)
        swapper.initialize()
        occluder = get_face_occluder(occluder_model) if occluder_model else None
        parser = get_face_parser(parser_model) if parser_model else None

        found_flags = [0] * n
        masks = [None] * n

        # Rosto fonte detectado UMA vez (é constante em todos os frames)
        src_faces = detect_faces(source_cv2, score_threshold, sort_order, face_detector_model)
        if not src_faces:
            print('[Bruxos FaceFusion] Nenhum rosto detectado na imagem fonte — retornando frames originais.')
            return (target_images, empty_mask, 0)
        src = src_faces[min(face_position, len(src_faces) - 1)]

        def _work(i):
            frame = frames_cv2[i]
            try:
                tgt_faces = detect_faces(frame, score_threshold, sort_order, face_detector_model)
                if face_selector_mode == 'reference':
                    used = find_matching_faces(ref_faces[0], tgt_faces, reference_distance)
                elif face_selector_mode == 'one':
                    used = [tgt_faces[min(face_position, len(tgt_faces) - 1)]] if tgt_faces else []
                else:  # many
                    used = tgt_faces
                result = frame.copy()
                for tf in used:
                    result = swapper.swap_face(src, tf, result, pixel_boost, face_mask_blur,
                                               occluder, parser, source_cv2,
                                               mask_types, areas, regions, padding)
                found_flags[i] = 1 if used else 0
                masks[i] = _faces_to_mask(used, h, w, out_mask_grow, out_mask_blur)
                return i, result
            except Exception as e:
                print(f'[Bruxos FaceFusion] Erro no frame {i}: {e}')
                masks[i] = np.zeros((h, w), dtype=np.float32)
                return i, frame

        results = [None] * n
        if n == 1 or max_workers <= 1:
            for i in range(n):
                idx, res = _work(i)
                results[idx] = res
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for idx, res in ex.map(_work, range(n)):
                    results[idx] = res

        out_images = torch.stack([cv2_to_tensor(r).squeeze(0)[..., :3] for r in results])
        out_masks = torch.from_numpy(np.stack(masks)).float()
        total = int(sum(found_flags))
        print(f'[Bruxos FaceFusion] {total}/{n} frames com rosto trocado '
              f'(modelo={face_swapper_model}, boost={pixel_boost}, masks={"+".join(mask_types)}).')
        return (out_images, out_masks, total)


class BruxosFaceFusionDetector:
    """Detecta rostos: preview com caixas, MASK por frame (pro Bernini) e contagem."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'images': ('IMAGE', {'tooltip': 'Video/imagem a analisar. Detecta rostos frame a frame.'}),
                'face_detector_model': (DETECTOR_MODELS, {'default': 'scrfd', 'tooltip': 'Detector de rostos. scrfd e um bom padrao. yolo_face/retinaface como alternativas; many roda varios (mais lento).'}),
                'score_threshold': ('FLOAT', {'default': 0.3, 'min': 0.05, 'max': 1.0, 'step': 0.05, 'tooltip': 'Confianca minima pra aceitar um rosto (0..1). Abaixe se nao detectar; suba se pegar rosto falso.'}),
                'sort_order': (SORT_ORDERS, {'default': 'large-small', 'tooltip': 'Ordem dos rostos (define o indice de cada um). large-small = do maior pro menor.'}),
                'max_faces': ('INT', {'default': 0, 'min': 0, 'max': 64, 'tooltip': 'Limita quantos rostos considerar por frame. 0 = todos.'}),
                'mask_grow': ('INT', {'default': 0, 'min': 0, 'max': 256, 'tooltip': 'Dilata a caixa dos rostos (px) antes de gerar a mascara de saida.'}),
                'mask_blur': ('INT', {'default': 6, 'min': 0, 'max': 128, 'tooltip': 'Suaviza a borda (feather) da mascara de rostos, em pixels.'}),
                'device': (['auto', 'cuda', 'cpu'], {'default': 'auto', 'tooltip': 'Onde rodar o detector ONNX. auto = GPU (CUDA) se disponivel, senao CPU. cuda = forca GPU (avisa no console se nao houver). cpu = forca CPU.'}),
            },
        }

    RETURN_TYPES = ('IMAGE', 'MASK', 'INT', 'STRING', 'BBOX')
    RETURN_NAMES = ('preview', 'face_mask', 'faces_first_frame', 'info', 'face_bboxes')
    OUTPUT_TOOLTIPS = (
        'Preview com caixas verdes e landmarks roxos sobre cada rosto (mais o indice e o score).',
        'Mascara dos rostos por frame (branco = rosto). Pronta p/ o region_mask do Bernini.',
        'Quantos rostos foram achados no PRIMEIRO frame (util p/ calibrar face_position no Swap).',
        'Relatorio por frame (quantos rostos em cada um).',
        'Caixa do rosto principal por frame (x1,y1,x2,y2). Liga no Face Crop Expand e no FaceStitchUpscale. Em frames sem rosto, repete a ultima caixa (nao pula).',
    )
    FUNCTION = 'process'
    CATEGORY = 'Bruxos do VFX/Face'
    DESCRIPTION = (
        'Detecta rostos e mostra caixas + landmarks, sem trocar nada. Use p/ achar o face_position certo, '
        'calibrar score_threshold antes do Swap, ou gerar uma MASK de rostos pro Bernini. '
        'Saidas: preview (IMAGE), face_mask (MASK por frame), faces_first_frame (INT), info (relatorio).'
    )

    def process(self, images, face_detector_model, score_threshold, sort_order,
                max_faces, mask_grow, mask_blur, device='auto'):
        from .facefusion.engine.runtime import set_device
        set_device(device)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        n, h, w = images.shape[0], images.shape[1], images.shape[2]

        previews, masks, lines = [], [], []
        face_bboxes = []          # 1 bbox por frame (rosto principal), p/ Crop/Stitch
        last_box = None           # carry-forward: repete a ultima caixa em frames sem rosto
        first_count = 0
        for i in range(n):
            frame = tensor_to_cv2(images[i:i + 1])
            faces = detect_faces(frame, score_threshold, sort_order, face_detector_model)
            if max_faces > 0:
                faces = faces[:max_faces]
            if i == 0:
                first_count = len(faces)
            vis = frame.copy()
            for j, f in enumerate(faces):
                x1, y1, x2, y2 = [int(round(float(v))) for v in f['bbox'][:4]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (94, 197, 34), 2)  # verde Bruxos (BGR)
                cv2.putText(vis, f'{j}:{f["score"]:.2f}', (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (94, 197, 34), 1, cv2.LINE_AA)
                if f.get('landmarks') is not None:
                    for (lx, ly) in np.asarray(f['landmarks']).reshape(-1, 2):
                        cv2.circle(vis, (int(lx), int(ly)), 2, (247, 85, 168), -1)  # roxo
            # bbox do rosto principal (o 1o apos o sort) p/ este frame
            if faces:
                bx = [float(v) for v in faces[0]['bbox'][:4]]
                last_box = (bx[0], bx[1], bx[2], bx[3])
            # se nao achou rosto neste frame, repete a ultima caixa (nao pula/some)
            if last_box is not None:
                face_bboxes.append(last_box)
            else:
                face_bboxes.append((0.0, 0.0, float(w), float(h)))  # fallback: frame inteiro
            previews.append(cv2_to_tensor(vis).squeeze(0)[..., :3])
            masks.append(_faces_to_mask(faces, h, w, mask_grow, mask_blur))
            if i < 32:
                lines.append(f'frame {i}: {len(faces)} rosto(s)')
        if n > 32:
            lines.append(f'... (+{n - 32} frames)')

        return (torch.stack(previews),
                torch.from_numpy(np.stack(masks)).float(),
                first_count,
                '\n'.join(lines),
                face_bboxes)


NODE_CLASS_MAPPINGS = {
    'BruxosFaceFusionSwap': BruxosFaceFusionSwap,
    'BruxosFaceFusionDetector': BruxosFaceFusionDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'BruxosFaceFusionSwap': 'FaceFusion Swap (Bruxos)',
    'BruxosFaceFusionDetector': 'FaceFusion Detectar Rostos (Bruxos)',
}
