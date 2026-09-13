"""OCR-based rotation detection."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

DEFAULT_MIN_CONFIDENCE = 0.1
DEFAULT_LANGUAGES = ("en",)
ROTATION_ANGLES = (0, 90, 180, 270)

@lru_cache(maxsize=4)
def _easyocr_reader(languages: tuple[str, ...]):
    import easyocr
    return easyocr.Reader(list(languages), gpu=False, verbose=False)

def _score_ocr_result(result: list[Any], min_confidence: float) -> dict[str, Any]:
    confidences, texts = [], []
    total_length = 0
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) < 3: continue
        text = str(item[1]).strip()
        try:
            confidence = float(item[2])
        except (TypeError, ValueError):
            continue
        
        if not text or confidence < min_confidence:
            continue
        
        confidences.append(confidence)
        texts.append(text)
        total_length += len(text)
    return {"score": round(sum(confidences) + total_length * 0.01, 6), "count": len(confidences), "mean_confidence": round(sum(confidences) / len(confidences), 6) if confidences else 0.0, "texts": texts}

def analyze_image_rotation(image: str | Path | Image.Image, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE, languages: tuple[str, ...] | list[str] = DEFAULT_LANGUAGES) -> dict[str, Any]:
    """
    Analyze an image for text rotation using OCR.
    Result structure:
    {
        "ocr_has_rotation": bool,  # True if text is rotated (90 or 270 degrees)
        "ocr_best_angle": int,      # Best rotation angle (0, 90, 180, 270)
        "ocr_min_confidence": float, # Minimum confidence threshold used
        "ocr_rotation_scores": dict[str, dict[str, Any]], # Scores for each angle
    }
    """
    pil_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
    reader = _easyocr_reader(tuple(languages))
    
    scores = {}
    for angle in ROTATION_ANGLES:
        rotated = pil_image.rotate(angle, expand=True) if angle else pil_image.copy()
        ocr_result = reader.readtext(np.array(rotated), detail=1)
        scores[str(angle)] = _score_ocr_result(ocr_result, min_confidence)
    
    # Get the best angle which has the highest score (identify most text)
    best_angle = max(ROTATION_ANGLES, key=lambda angle: (scores[str(angle)]["score"], scores[str(angle)]["count"]))
    
    return {
        "ocr_has_rotation": best_angle in (90, 270),
        "ocr_best_angle": best_angle,
        "ocr_min_confidence": min_confidence,
        "ocr_rotation_scores": scores,
    }
