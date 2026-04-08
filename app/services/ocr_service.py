from typing import Any
import logging

import cv2
import numpy as np
from paddleocr import PaddleOCR

from app.config import settings
from app.models.schemas import PageResult, TextBoxResult


class OCRService:
    """Wraps PaddleOCR (v3) with lazy singleton-like pattern."""

    def __init__(self, lang: str = "ch", use_gpu: bool = False):
        # Suppress PaddleOCR's verbose startup logs
        logging.getLogger("ppocr").setLevel(logging.WARNING)
        logging.getLogger("paddlex").setLevel(logging.WARNING)

        kwargs: dict[str, Any] = {
            "use_textline_orientation": True,
            "lang": lang,
        }
        if use_gpu:
            kwargs["use_gpu"] = True
        self._ocr = PaddleOCR(**kwargs)

    def recognize_image(self, image_path: str) -> PageResult:
        """Run OCR on a single image file."""
        results = self._ocr.predict(image_path)
        return self._parse(results, page_index=0)

    def recognize_bytes(self, image_bytes: bytes) -> PageResult:
        """Run OCR on raw image bytes."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        results = self._ocr.predict(img)
        return self._parse(results, page_index=0)

    @staticmethod
    def _parse(results, page_index: int) -> PageResult:
        """Parse PaddleOCR v3 output into PageResult.

        v3 returns a list of OCRResult objects that behave like dicts:
          item["rec_texts"]  : list[str]
          item["rec_scores"] : list[float]
          item["rec_polys"]  : list of 4-point polygons (numpy arrays)
        """
        text_blocks: list[TextBoxResult] = []
        full_parts: list[str] = []

        if results:
            for item in results:
                texts = item.get("rec_texts", [])
                scores = item.get("rec_scores", [])
                polys = item.get("rec_polys", [])
                for i, text in enumerate(texts):
                    if not text.strip():
                        continue
                    confidence = float(scores[i]) if i < len(scores) else 0.0
                    poly = polys[i] if i < len(polys) else []
                    box = poly.tolist() if hasattr(poly, 'tolist') else poly
                    text_blocks.append(
                        TextBoxResult(
                            text=text,
                            confidence=round(confidence, 4),
                            bbox=[(int(x), int(y)) for x, y in box],
                        )
                    )
                    full_parts.append(text)

        return PageResult(
            page_index=page_index,
            text_blocks=text_blocks,
            full_text="\n".join(full_parts),
        )


# Lazy-loaded global instance
_service: OCRService | None = None


def get_ocr_service() -> OCRService:
    global _service
    if _service is None:
        _service = OCRService(
            lang=settings.ocr_lang,
            use_gpu=False,
        )
    return _service
