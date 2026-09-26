"""Plate cropping and PaddleOCR recognition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .config import MIN_OCR_SCORE, OCR_SCALE, PLATE_PADDING

if TYPE_CHECKING:
    from paddleocr import TextRecognition
    from ultralytics.engine.results import Boxes


def usable_reading(text: str, score: float) -> bool:
    """Apply the same OCR quality gate to display and history."""
    return (
        bool(text.strip()) and bool(np.isfinite(score)) and MIN_OCR_SCORE <= score <= 1
    )


def crop_plate(
    image: np.ndarray,
    box: Boxes,
    padding: float = PLATE_PADDING,
    scale: int = OCR_SCALE,
) -> np.ndarray | None:
    """Crop, pad, and upscale a detected plate for OCR."""
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding

    left = max(round(x1 - pad_x), 0)
    top = max(round(y1 - pad_y), 0)
    right = min(round(x2 + pad_x), image_width)
    bottom = min(round(y2 + pad_y), image_height)

    if left >= right or top >= bottom:
        return None

    crop = image[top:bottom, left:right]
    return cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def recognize_plate(recognizer: TextRecognition, crop: np.ndarray) -> tuple[str, float]:
    """Return the recognized text and confidence for a plate crop."""
    predictions = recognizer.predict(input=crop, batch_size=1)
    if not predictions:
        return "", 0.0

    prediction = predictions[0]
    return str(prediction["rec_text"]).strip(), float(prediction["rec_score"])
