"""Annotate detector results and combine plate readings for video tracks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .drawing import car_label, draw_box, draw_label
from .ocr import crop_plate, recognize_plate, usable_reading
from .tracking import containing_cars, plate_owner

if TYPE_CHECKING:
    from paddleocr import TextRecognition
    from ultralytics.engine.results import Boxes, Results

    from .tracking import VideoPlateTracker


def select_car_detections(result: Results) -> Results:
    """Suppress cars losing a shared plate, unless they win another plate."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return result
    cars = {
        index: box.xyxy[0].tolist()
        for index, box in enumerate(boxes)
        if result.names[int(box.cls.item())].lower() == "car"
    }
    confidences = {index: float(boxes[index].conf.item()) for index in cars}
    candidates: set[int] = set()
    winners: set[int] = set()
    for box in boxes:
        if "plate" not in result.names[int(box.cls.item())].lower():
            continue
        plate = box.xyxy[0].tolist()
        candidates.update(containing_cars(plate, cars))
        owner = plate_owner(plate, cars, confidences)
        if owner is not None:
            winners.add(owner)
    suppressed = candidates - winners
    return result[[index for index in range(len(boxes)) if index not in suppressed]]


def select_plate_detections(
    boxes: Boxes,
    names: dict[int, str],
    cars: dict[int, list[float]],
    confidences: dict[int, float],
) -> set[int]:
    """Keep the highest YOLO-confidence plate per selected car."""
    selected = set()
    best: dict[int, tuple[int, float]] = {}
    for index, box in enumerate(boxes):
        if "plate" not in names[int(box.cls.item())].lower():
            selected.add(index)
            continue
        owner = plate_owner(box.xyxy[0].tolist(), cars, confidences)
        if owner is None:
            selected.add(index)
            continue
        confidence = float(box.conf.item())
        if owner not in best or confidence > best[owner][1]:
            best[owner] = (index, confidence)
    selected.update(index for index, _ in best.values())
    return selected


def annotate_result(
    result: Results,
    recognizer: TextRecognition,
    *,
    verbose: bool = False,
    tracking: VideoPlateTracker | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Use the same box styles and car-label placement for images and videos."""
    annotated = result.orig_img.copy()
    plate_texts: list[str] = []
    result = select_car_detections(result)
    boxes = result.boxes
    readings = []
    if tracking is not None:
        tracking.update(result)

    if boxes is None or len(boxes) == 0:
        if verbose:
            print("  No detections")
        return annotated, plate_texts

    cars = {
        index: box.xyxy[0].tolist()
        for index, box in enumerate(boxes)
        if result.names[int(box.cls.item())].lower() == "car"
    }
    confidences = {index: float(boxes[index].conf.item()) for index in cars}
    if tracking is not None:
        cars = tracking.association_boxes
        confidences = tracking.association_confidences
    selected = select_plate_detections(boxes, result.names, cars, confidences)
    image_readings = {}

    for index, box in enumerate(boxes):
        if index not in selected:
            continue
        class_name = result.names[int(box.cls.item())]
        normalized_name = class_name.lower()
        confidence = float(box.conf.item())
        coordinates = box.xyxy[0].tolist()
        if verbose:
            print(f"  {class_name}: {confidence:.2f}")
        draw_box(annotated, coordinates, normalized_name)

        if "plate" not in normalized_name:
            if normalized_name != "car":
                draw_label(annotated, coordinates, class_name, normalized_name)
            continue

        plate_label = "Plate"
        crop = crop_plate(result.orig_img, box)
        if crop is None:
            if verbose:
                print("    Skipped invalid plate crop")
        else:
            text, score = recognize_plate(recognizer, crop)
            if verbose:
                print(f"    Plate text: '{text}' (OCR score: {score:.2f})")
            if not usable_reading(text, score):
                text = ""
            owner = plate_owner(coordinates, cars, confidences)
            if tracking is not None:
                readings.append((coordinates, text, score))
            elif text and owner is not None:
                image_readings[owner] = (text, score)
            # Unassociated readings remain on the plate, never on an arbitrary car.
            if text and owner is None and tracking is None:
                plate_label = f"Plate: {text} (OCR: {score:.2f})"
                plate_texts.append(plate_label)
        draw_label(annotated, coordinates, plate_label, normalized_name)

    tracked_labels = {}
    if tracking is not None:
        tracking.add_readings(readings)
        tracked_labels = tracking.labels()
    for car_id, coordinates in cars.items():
        if tracking is not None:
            label, text = tracked_labels.get(car_id, (car_label(), ""))
        else:
            text, score = image_readings.get(car_id, ("", 0.0))
            label = car_label(text, score)
        draw_label(annotated, coordinates, label, "car")
        if text:
            plate_texts.append(label)

    return annotated, plate_texts
