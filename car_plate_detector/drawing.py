"""Shared box and label styling for images and videos."""

from collections.abc import Sequence

import cv2
import numpy as np

from .config import (
    BOX_THICKNESS,
    CLASS_COLORS,
    DEFAULT_COLOR,
    LABEL_PADDING,
    LABEL_TEXT_COLOR,
)


def class_color(class_name: str) -> tuple[int, int, int]:
    """Use one color for a class's boxes and labels."""
    if "plate" in class_name:
        return CLASS_COLORS["car plate"]
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def car_label(
    text: str = "",
    score: float = 0.0,
    *,
    track_id: int | None = None,
    confirmed: bool | None = None,
) -> str:
    label = "Car" if track_id is None else f"Car #{track_id}"
    if text:
        score_name = "OCR" if confirmed is None else "OCR avg"
        label += f": {text} ({score_name}: {score:.2f})"
        if confirmed is False:
            label += " (unconfirmed)"
    return label


def draw_label(
    image: np.ndarray, box: Sequence[float], label: str, class_name: str
) -> None:
    """Draw wrapped white text on a solid background matching the box."""
    left, top, right, _ = box
    x = max(0, round(left))
    width = max(
        1, min(max(80, round(right - left)), image.shape[1] - x) - 2 * LABEL_PADDING
    )
    lines = [""]
    for word in label.split():
        candidate = f"{lines[-1]} {word}".strip()
        (text_width, _), _ = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        if lines[-1] and text_width > width:
            lines.append(word)
        else:
            lines[-1] = candidate
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2) for line in lines]
    text_height = max(size[0][1] for size in sizes)
    baseline = max(size[1] for size in sizes)
    line_height = text_height + baseline + LABEL_PADDING
    background_width = max(size[0][0] for size in sizes) + 2 * LABEL_PADDING
    background_height = len(lines) * line_height + LABEL_PADDING
    x = max(0, min(x, image.shape[1] - background_width))
    top = max(
        0, min(round(top) - background_height, image.shape[0] - background_height)
    )
    cv2.rectangle(
        image,
        (x, top),
        (x + background_width, top + background_height),
        class_color(class_name),
        cv2.FILLED,
    )
    y = top + LABEL_PADDING + text_height
    for line in lines:
        cv2.putText(
            image,
            line,
            (x + LABEL_PADDING, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            LABEL_TEXT_COLOR,
            2,
        )
        y += line_height


def draw_box(image: np.ndarray, box: Sequence[float], class_name: str) -> None:
    left, top, right, bottom = map(round, box)
    color = class_color(class_name)
    cv2.rectangle(image, (left, top), (right, bottom), color, BOX_THICKNESS)
