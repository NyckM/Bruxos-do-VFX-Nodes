"""
Wrapper do filtro de conteúdo (NSFW). Mantido intacto e sempre ativo — igual ao upstream.
Modelos ficam em models/facefusion/content_filter.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CF_PATH = os.path.join(_HERE, 'content_filter')
if _CF_PATH not in sys.path:
    sys.path.insert(0, _CF_PATH)

CONTENT_FILTER_AVAILABLE = False
_filter = None


def _get_filter():
    global _filter, CONTENT_FILTER_AVAILABLE
    if _filter is None:
        try:
            import content_filter as _cf_mod  # content_filter.py (dir no sys.path)
            from .engine.utils import get_models_dir
            inst = _cf_mod.ContentFilter(models_dir=os.path.join(get_models_dir(), 'content_filter'))
            _cf_mod._filter_instance = inst  # funcoes globais usam nossa instancia
            _filter = _cf_mod
            CONTENT_FILTER_AVAILABLE = True
        except Exception as e:
            print(f"[Bruxos FaceFusion] Filtro de conteúdo indisponível: {e}")
            _filter = False
    return _filter or None


def analyse_frame(frame) -> bool:
    f = _get_filter()
    if f is None:
        return False
    try:
        return bool(f.analyse_frame(frame))
    except Exception as e:
        print(f"[Bruxos FaceFusion] Erro no filtro de conteúdo: {e}")
        return False


def blur_frame(frame, blur_amount: int = 99):
    f = _get_filter()
    if f is not None:
        try:
            return f.blur_frame(frame, blur_amount)
        except Exception:
            pass
    try:
        import cv2
        k = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
        return cv2.GaussianBlur(frame, (k, k), 0)
    except Exception:
        return frame
