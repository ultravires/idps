from typing import Any
import logging
import threading
import time

import cv2
import numpy as np
from paddleocr import PaddleOCR

from app.config import settings
from app.models.schemas import PageResult, TextBoxResult

logger = logging.getLogger(__name__)


class OCRService:
    """Wraps PaddleOCR (v3) with lazy singleton-like pattern."""

    def __init__(
        self,
        lang: str = "ch",
        use_gpu: bool = False,
        det_model_name: str | None = None,
        rec_model_name: str | None = None,
        det_limit_side_len: int = 2000,
    ):
        # Suppress PaddleOCR's verbose startup logs
        logging.getLogger("ppocr").setLevel(logging.WARNING)
        logging.getLogger("paddlex").setLevel(logging.WARNING)

        kwargs: dict[str, Any] = {
            "use_textline_orientation": True,
            "lang": lang,
        }
        if use_gpu:
            kwargs["device"] = "gpu"
        if det_model_name:
            kwargs["text_detection_model_name"] = det_model_name
        if rec_model_name:
            kwargs["text_recognition_model_name"] = rec_model_name
        self._det_limit_side_len = det_limit_side_len
        self._ocr = PaddleOCR(**kwargs)

    def _predict(self, image) -> Any:
        """Run OCR with the detection input size capped.

        The default pipeline only shrinks images above 4000px, which makes
        CPU detection take minutes on large photos. Capping the long side
        keeps detection fast; recognition still crops from the original
        resolution, so accuracy is largely preserved.
        """
        start = time.perf_counter()
        results = self._ocr.predict(
            image,
            text_det_limit_type="max",
            text_det_limit_side_len=self._det_limit_side_len,
        )
        elapsed = time.perf_counter() - start
        logger.info("OCR predict finished in %.2fs", elapsed)
        return results

    def recognize_image(self, image_path: str) -> PageResult:
        """Run OCR on a single image file."""
        return self._parse(self._predict(image_path), page_index=0)

    def recognize_bytes(self, image_bytes: bytes) -> PageResult:
        """Run OCR on raw image bytes."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        h, w = img.shape[:2]
        logger.info("Running OCR on %dx%d image", w, h)
        return self._parse(self._predict(img), page_index=0)

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
_service_lock = threading.Lock()


def get_ocr_service() -> OCRService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = OCRService(
                    lang=settings.ocr_lang,
                    use_gpu=False,
                    det_model_name=settings.ocr_det_model_name,
                    rec_model_name=settings.ocr_rec_model_name,
                    det_limit_side_len=settings.ocr_det_limit_side_len,
                )
    return _service
