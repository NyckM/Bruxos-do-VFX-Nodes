"""
Engine local do FaceFusion (vendorizado de huygiatrng/Facefusion_comfyui, MIT-style — ver LICENSE upstream).
Somente inferência LOCAL via ONNX. Nenhuma chamada de API externa.
"""
from .swap_local import swap_faces_local
from .detection.detector import detect_faces, get_face_detector
from .models import get_local_swapper, get_face_occluder, get_face_parser, MODEL_CONFIGS, MODEL_URLS
from .utils import (
    tensor_to_cv2, cv2_to_tensor, sort_faces_by_order, find_matching_faces,
    get_average_embedding, VisionFrame, Face,
)

__all__ = [
    'swap_faces_local', 'detect_faces', 'get_face_detector',
    'get_local_swapper', 'get_face_occluder', 'get_face_parser',
    'MODEL_CONFIGS', 'MODEL_URLS',
    'tensor_to_cv2', 'cv2_to_tensor', 'sort_faces_by_order', 'find_matching_faces',
    'get_average_embedding', 'VisionFrame', 'Face',
]
