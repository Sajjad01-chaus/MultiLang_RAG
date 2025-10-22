import os
from typing import Dict, Optional
import numpy as np
from langdetect import detect
from paddleocr import PaddleOCR
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import pytesseract  # ✅ Import first!
from pytesseract import Output

# ✅ Set Tesseract path (after import)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ POPPLER PATH
POPPLER_PATH = r"C:\Users\ABC\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"

# --- OCR Model Cache ---
_OCR_INSTANCES = {}

# --- Language Maps ---
LANG_CODE_MAP = {
    "english": "en",
    "eng": "en",
    "hindi": "hi",
    "hin": "hi",
    "bengali": "ben",
    "ben": "ben",
    "tamil": "ta",
    "telugu": "te",
    "marathi": "mr",
    "urdu": "ur",
    "chinese": "ch",
    "japanese": "jp",
    "korean": "ko",
    "arabic": "ar",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "russian": "ru",
    "portuguese": "pt",
}

TESS_LANG_MAP = {
    "ben": "ben",
    "hi": "hin",
    "en": "eng",
    "ta": "tam",
    "te": "tel",
    "mr": "mar",
    "ur": "urd",
    "ch": "chi_sim",
    "jp": "jpn",
    "ko": "kor",
    "ar": "ara",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "ru": "rus",
    "pt": "por",
}

def detect_lang_safe(text: str) -> str:
    """Safely detect language."""
    try:
        return detect(text)
    except Exception:
        return "und"

def get_ocr_model(lang_hint: str):
    """Return cached PaddleOCR model for supported languages."""
    lang_code = LANG_CODE_MAP.get(lang_hint.lower(), None)
    if not lang_code:
        raise ValueError(f"Unsupported language '{lang_hint}' for PaddleOCR.")

    if lang_code not in _OCR_INSTANCES:
        print(f"[OCR] Loading PaddleOCR model for '{lang_code}'...")
        _OCR_INSTANCES[lang_code] = PaddleOCR(use_angle_cls=True, lang=lang_code, show_log=False)
    return _OCR_INSTANCES[lang_code]

# --- Core Extraction Function ---
def extract_text_from_pdf(path: str, lang_hint: Optional[str] = None) -> Dict[int, Dict]:
    pages = {}
    print(f"[EXTRACT] Starting extraction for {path}")
    print(f"[EXTRACT] Language hint: {lang_hint or 'auto'}")

    # Try digital text extraction first
    try:
        raw_text = extract_text(path)
        if raw_text and len(raw_text.strip()) > 100:
            print(f"[EXTRACT] Extracted {len(raw_text)} chars via digital method ✅")
            page_texts = raw_text.split("\f")
            for i, txt in enumerate(page_texts):
                if txt.strip():
                    pages[i] = {
                        "text": txt.strip(),
                        "is_scanned": False,
                        "lang": detect_lang_safe(txt)
                    }
            return pages
    except Exception as e:
        print(f"[WARN] Digital extraction failed: {e}")

    print("[EXTRACT] Falling back to OCR (scanned document detected).")

    if not lang_hint:
        raise ValueError("Scanned document detected. Please provide a language hint (e.g., 'hindi', 'bengali', 'urdu').")

    # Convert PDF → Images
    if os.name == "nt":
        images = convert_from_path(path, dpi=200, poppler_path=POPPLER_PATH)
    else:
        images = convert_from_path(path, dpi=200)

    print(f"[EXTRACT] Converted PDF into {len(images)} images.")

    # Try PaddleOCR first
    paddle_success = False
    try:
        ocr = get_ocr_model(lang_hint)
        paddle_success = True
    except Exception as e:
        print(f"[WARN] PaddleOCR not available for {lang_hint}: {e}")

    for i, img in enumerate(images):
        text = ""
        try:
            if paddle_success:
                arr = np.array(img)
                ocr_result = ocr.ocr(arr, cls=True)
                if ocr_result and ocr_result[0]:
                    text = "\n".join([line[1][0] for line in ocr_result[0]])
                    print(f"[OCR] PaddleOCR extracted {len(text)} chars from page {i+1}.")
            if not text.strip():
                raise Exception("Empty text from PaddleOCR, fallback to Tesseract.")
        except Exception as e:
            print(f"[FALLBACK] Using Tesseract for page {i+1}: {e}")
            tess_lang = TESS_LANG_MAP.get(LANG_CODE_MAP.get(lang_hint.lower(), ""), "eng")
            text = pytesseract.image_to_string(img, lang=tess_lang, config="--psm 6")
            print(f"[TESSERACT] Extracted {len(text)} chars from page {i+1}.")

        if text.strip():
            pages[i] = {
                "text": text.strip(),
                "is_scanned": True,
                "lang": detect_lang_safe(text) if text else lang_hint
            }

    if not pages:
        raise ValueError("No text could be extracted from scanned document. Try providing clearer scans or correct language hint.")

    print(f"[EXTRACT] Completed OCR extraction for {len(pages)} pages ✅")
    return pages