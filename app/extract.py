# from typing import Dict
# from langdetect import detect
# from paddleocr import PaddleOCR
# from pdfminer.high_level import extract_text
# from pdf2image import convert_from_path

# # Pre-initialize small set of OCR models (lazy initialize if you prefer)
# # Use 'multi' if you want one big model (heavier). Keep minimal models to save disk/RAM.
# _OCR_INSTANCES = {}

# def get_ocr_model(lang_hint: str = "en"):
#     """
#     Return a PaddleOCR instance for the given language hint.
#     Cache instances to avoid reloading heavy models repeatedly.
#     """
#     # normalize hint
#     key = lang_hint if lang_hint in ("en", "hi", "multi") else "multi"
#     if key not in _OCR_INSTANCES:
#         # initialize; choose 'multi' if unknown
#         _OCR_INSTANCES[key] = PaddleOCR(use_angle_cls=True, lang=key)
#     return _OCR_INSTANCES[key]

# def detect_lang_safe(text: str) -> str:
#     try:
#         return detect(text)
#     except Exception:
#         return "und"

# def extract_text_from_pdf(path: str, lang_hint: str = None) -> Dict[int, Dict]:
#     """
#     Extract text from a PDF file.

#     Returns:
#         pages: dict mapping page_num -> { 'text': str, 'is_scanned': bool, 'lang': str }
#     """
#     pages = {}

#     # 1) Try digital text extraction (fast)
#     try:
#         raw = extract_text(path)
#     except Exception:
#         raw = ""

#     if raw and len(raw.strip()) > 100:
#         page_texts = raw.split('\f')
#         for i, p in enumerate(page_texts):
#             text = p.strip()
#             if text:
#                 pages[i] = {"text": text, "is_scanned": False, "lang": detect_lang_safe(text)}
#         return pages

#     # 2) Fallback: OCR pages (scanned PDF). Render pages to images first.
#     pil_pages = convert_from_path(path, dpi=300)
#     # get an OCR model - prefer lang_hint if provided else fallback to multi
#     ocr_model = get_ocr_model(lang_hint or "multi")
#     for i, img in enumerate(pil_pages):
#         try:
#             res = ocr_model.ocr(img, det=True, rec=True)
#             text = "\n".join([line[-1][0] for block in res for line in block if line])
#         except Exception:
#             # fallback to empty
#             text = ""
#         pages[i] = {"text": text, "is_scanned": True, "lang": detect_lang_safe(text) if text else (lang_hint or "und")}
#     return pages

from typing import Dict
from langdetect import detect
from paddleocr import PaddleOCR
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path

_OCR_INSTANCES = {}


def get_ocr_model(lang_hint: str = "en"):
    """Return cached PaddleOCR model."""
    key = lang_hint if lang_hint in ("en", "hi", "multi") else "multi"
    if key not in _OCR_INSTANCES:
        _OCR_INSTANCES[key] = PaddleOCR(use_angle_cls=True, lang=key)
    return _OCR_INSTANCES[key]


def detect_lang_safe(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "und"


def extract_text_from_pdf(path: str, lang_hint: str = None) -> Dict[int, Dict]:
    """
    Extract text from a PDF file.
    - Tries digital extraction first
    - Falls back to OCR if needed
    - Returns {} instead of None
    """
    pages = {}

    # 1️⃣ Try digital text extraction
    try:
        raw = extract_text(path)
    except Exception:
        raw = ""

    if raw and len(raw.strip()) > 100:
        page_texts = raw.split("\f")
        for i, p in enumerate(page_texts):
            text = p.strip()
            if text:
                pages[i] = {
                    "text": text,
                    "is_scanned": False,
                    "lang": detect_lang_safe(text)
                }
        if pages:
            return pages  # ✅ digital extraction successful

    # 2️⃣ Fallback to OCR
    try:
        pil_pages = convert_from_path(path, dpi=200)
        ocr_model = get_ocr_model(lang_hint or "multi")

        for i, img in enumerate(pil_pages):
            try:
                res = ocr_model.ocr(img, det=True, rec=True)
                text = "\n".join([line[-1][0] for block in res for line in block if line])
            except Exception:
                text = ""

            pages[i] = {
                "text": text,
                "is_scanned": True,
                "lang": detect_lang_safe(text) if text else (lang_hint or "und")
            }
    except Exception:
        return {}  # 🛡️ never None

    return pages or {}  # 🛡️ always return dict
