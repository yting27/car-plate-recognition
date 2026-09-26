"""Command-line setup and image/video application flow."""

import argparse

from .config import (
    MODEL_PATH,
    OCR_MODEL_DIR,
    OCR_MODEL_NAME,
    TEST_DIR,
    VIDEO_EXTENSIONS,
    VIDEO_MAX_FPS,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View car and license plate detections."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-frame detections and OCR details.",
    )
    args = parser.parse_args()

    import tkinter as tk

    from paddleocr import TextRecognition
    from ultralytics import YOLO

    from .media import find_media, process_video
    from .pipeline import annotate_result
    from .viewer import DetectorViewer

    media = find_media(TEST_DIR)
    if not media:
        print(f"No images or videos found in {TEST_DIR}")
        return

    detector = YOLO(str(MODEL_PATH))
    recognizer = TextRecognition(
        model_name=OCR_MODEL_NAME, model_dir=str(OCR_MODEL_DIR)
    )
    root = tk.Tk()
    viewer = DetectorViewer(root)

    print(f"Found {len(media)} image/video file(s) in {TEST_DIR}")
    try:
        for media_path in media:
            print(f"\nProcessing: {media_path.name}")
            # Video data
            if media_path.suffix.lower() in VIDEO_EXTENSIONS:
                print(
                    f"  Video processes at up to {VIDEO_MAX_FPS} FPS; "
                    "any key = next file, 'q'/ESC = quit"
                )
                if (
                    process_video(
                        media_path, detector, recognizer, viewer, verbose=args.verbose
                    )
                    == "quit"
                ):
                    break
                continue

            # Image data
            result = detector.predict(source=str(media_path), verbose=False)[0]
            annotated, _ = annotate_result(result, recognizer, verbose=args.verbose)
            viewer.show(annotated)

            print("  Any key = next image, 'q'/ESC = quit")
            if viewer.wait_for_action() == "quit":
                break
    finally:
        viewer.close()
