"""
OCR Engine Module
Wraps EasyOCR and Tesseract for text extraction with confidence scores.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class OCRResult:
    """Container for a single OCR text detection result."""

    def __init__(self, text: str, confidence: float, bbox: Optional[List] = None):
        self.text = text.strip()
        self.confidence = confidence  # 0.0 - 1.0
        self.bbox = bbox  # Bounding box [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    def __repr__(self):
        return f"OCRResult(text={self.text!r}, confidence={self.confidence:.2f})"


class OCREngine:
    """
    Multi-engine OCR wrapper supporting EasyOCR (primary) and Tesseract (fallback).
    """

    def __init__(self, engine: str = "easyocr", languages: List[str] = None):
        self.engine_name = engine
        self.languages = languages or ["en"]
        self._reader = None
        self._init_engine()

    def _init_engine(self):
        """Initialize the selected OCR engine."""
        if self.engine_name == "easyocr":
            try:
                import easyocr
                self._reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
                logger.info("EasyOCR engine initialized")
            except ImportError:
                logger.warning("EasyOCR not available, falling back to Tesseract")
                self.engine_name = "tesseract"
                self._init_engine()

        elif self.engine_name == "tesseract":
            try:
                import pytesseract
                self._reader = pytesseract
                logger.info("Tesseract engine initialized")
            except ImportError:
                raise RuntimeError("Neither EasyOCR nor Tesseract is available. "
                                   "Install with: pip install easyocr OR pytesseract")

    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """
        Extract text from a preprocessed image.
        Returns a list of OCRResult objects sorted top-to-bottom.
        """
        if self.engine_name == "easyocr":
            return self._extract_easyocr(image)
        elif self.engine_name == "tesseract":
            return self._extract_tesseract(image)

    def _extract_easyocr(self, image: np.ndarray) -> List[OCRResult]:
        """Run EasyOCR and return structured results."""
        try:
            raw_results = self._reader.readtext(image, detail=1, paragraph=False)
            results = []
            for bbox, text, confidence in raw_results:
                if text.strip():
                    results.append(OCRResult(text=text, confidence=float(confidence), bbox=bbox))

            # Sort top-to-bottom by Y coordinate of bounding box
            results.sort(key=lambda r: r.bbox[0][1] if r.bbox else 0)
            return results
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []

    def _extract_tesseract(self, image: np.ndarray) -> List[OCRResult]:
        """Run Tesseract and return structured results with word-level confidence."""
        try:
            import pytesseract
            from PIL import Image

            pil_image = Image.fromarray(image)
            data = pytesseract.image_to_data(
                pil_image,
                config="--psm 6",
                output_type=pytesseract.Output.DICT
            )

            results = []
            n_boxes = len(data["level"])
            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf_raw = data["conf"][i]
                if text and conf_raw != -1:
                    confidence = max(0.0, min(1.0, float(conf_raw) / 100.0))
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    results.append(OCRResult(text=text, confidence=confidence, bbox=bbox))

            results.sort(key=lambda r: r.bbox[0][1] if r.bbox else 0)
            return results
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return []

    def get_full_text(self, results: List[OCRResult]) -> str:
        """Join all detected text into a single string (newline-separated)."""
        return "\n".join(r.text for r in results if r.text)

    def get_average_confidence(self, results: List[OCRResult]) -> float:
        """Compute mean confidence across all OCR results."""
        if not results:
            return 0.0
        return sum(r.confidence for r in results) / len(results)
