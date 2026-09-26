"""Media discovery and sampled video playback."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .config import (
    DETECTION_CONFIDENCE,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    VIDEO_MAX_FPS,
)
from .pipeline import annotate_result
from .tracking import VideoPlateTracker

if TYPE_CHECKING:
    from paddleocr import TextRecognition
    from ultralytics import YOLO

    from .viewer import DetectorViewer


def find_media(directory: Path) -> list[Path]:
    """Return image and video files below *directory* in deterministic order."""
    if not directory.is_dir():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    )


def process_video(
    path: Path,
    detector: YOLO,
    recognizer: TextRecognition,
    viewer: DetectorViewer,
    *,
    verbose: bool = False,
) -> str:
    """Sample video frames for detection and OCR at up to VIDEO_MAX_FPS."""
    capture = cv2.VideoCapture(str(path))
    tracking = VideoPlateTracker()
    try:
        if not capture.isOpened():
            print(f"  Cannot open video: {path}")
            return "advance"

        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30
        processing_fps = min(fps, VIDEO_MAX_FPS)
        frame_seconds = 1 / processing_fps
        sample_interval = fps / processing_fps
        next_sample_frame = 0.0
        frame_number = 0
        started = monotonic()
        while True:
            if not capture.grab():
                print(f"  Video ended after {frame_number} source frame(s)")
                return "advance"

            frame_number += 1
            if frame_number - 1 + 1e-9 < next_sample_frame:
                continue
            next_sample_frame += sample_interval
            success, frame = capture.retrieve()
            if not success:
                print(f"  Cannot decode frame {frame_number}")
                return "advance"
            if verbose:
                print(f"  Frame {frame_number}")
            result = detector.predict(
                source=frame, conf=DETECTION_CONFIDENCE, verbose=False
            )[0]
            annotated, _ = annotate_result(
                result, recognizer, verbose=verbose, tracking=tracking
            )
            viewer.show(annotated)
            delay_ms = max(1, round((frame_seconds - (monotonic() - started)) * 1000))
            action = viewer.wait_for_action(timeout_ms=delay_ms, allow_pause=True)
            if action == "pause":
                action = viewer.wait_for_action(allow_pause=True)
                if action == "pause":
                    action = "tick"
            if action != "tick":
                return action
            started = monotonic()
    finally:
        capture.release()
