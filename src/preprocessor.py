"""
Image Preprocessing Module
Handles noise removal, skew correction, contrast enhancement for receipt images.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses receipt images for optimal OCR performance."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline:
        1. Convert to grayscale
        2. Denoise
        3. Correct skew
        4. Enhance contrast
        5. Binarize
        """
        logger.info("Starting image preprocessing")

        # Step 1: Grayscale
        gray = self._to_grayscale(image)

        # Step 2: Denoise
        denoised = self._denoise(gray)

        # Step 3: Skew correction
        deskewed = self._deskew(denoised)

        # Step 4: Contrast enhancement
        enhanced = self._enhance_contrast(deskewed)

        # Step 5: Binarization (Otsu's thresholding)
        binary = self._binarize(enhanced)

        logger.info("Preprocessing complete")
        return binary

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur + Non-local means denoising."""
        # Light Gaussian blur to smooth noise
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        # Non-local means for stronger denoising
        denoised = cv2.fastNlMeansDenoising(blurred, h=10, templateWindowSize=7, searchWindowSize=21)
        return denoised

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct skew angle using Hough transform."""
        try:
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                     minLineLength=100, maxLineGap=10)

            if lines is None or len(lines) == 0:
                return image

            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if -45 < angle < 45:
                        angles.append(angle)

            if not angles:
                return image

            median_angle = np.median(angles)
            if abs(median_angle) < 0.5:
                return image

            h, w = image.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            logger.debug(f"Deskewed by {median_angle:.2f} degrees")
            return rotated
        except Exception as e:
            logger.warning(f"Deskew failed: {e}, returning original")
            return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better handling of uneven lighting."""
        # Try Otsu first
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Also try adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        # Combine: use adaptive where Otsu fails (edge cases)
        combined = cv2.bitwise_and(otsu, adaptive)
        return combined

    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from disk."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image

    def preprocess_from_path(self, image_path: str) -> np.ndarray:
        """Load and preprocess an image from a file path."""
        image = self.load_image(image_path)
        return self.preprocess(image)
