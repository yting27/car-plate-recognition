import argparse
from pathlib import Path
from time import monotonic
import tkinter as tk

import cv2
import numpy as np
from paddleocr import TextRecognition
from PIL import Image, ImageTk
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "dataset" / "test"
MODEL_PATH = BASE_DIR / "models" / "yolo26n-v1.pt"
OCR_MODEL_DIR = BASE_DIR / "models" / "PP-OCRv6_medium_rec"

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})
CLASS_COLORS = {
    "car": (255, 0, 0),
    "car plate": (0, 255, 0),
}
DEFAULT_COLOR = (0, 200, 255)
BOX_THICKNESS = 2
PLATE_PADDING = 0.1
OCR_SCALE = 2
VIDEO_MAX_FPS = 10


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


def recognize_plate(
    recognizer: TextRecognition, crop: np.ndarray
) -> tuple[str, float]:
    """Return the recognized text and confidence for a plate crop."""
    predictions = recognizer.predict(input=crop, batch_size=1)
    if not predictions:
        return "", 0.0

    prediction = predictions[0]
    return str(prediction["rec_text"]).strip(), float(prediction["rec_score"])


def annotate_result(
    result: Results, recognizer: TextRecognition, *, verbose: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Draw detections and recognize every detected license plate."""
    annotated = result.orig_img.copy()
    plate_texts: list[str] = []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        if verbose:
            print("  No detections")
        return annotated, plate_texts

    for box in boxes:
        class_name = result.names[int(box.cls.item())]
        normalized_name = class_name.lower()
        confidence = float(box.conf.item())
        if verbose:
            print(f"  {class_name}: {confidence:.2f}")

        x1, y1, x2, y2 = (round(value) for value in box.xyxy[0].tolist())
        color = CLASS_COLORS.get(normalized_name, DEFAULT_COLOR)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        if "plate" not in normalized_name:
            continue

        crop = crop_plate(result.orig_img, box)
        if crop is None:
            if verbose:
                print("    Skipped invalid plate crop")
            continue

        text, score = recognize_plate(recognizer, crop)
        if verbose:
            print(f"    Plate text: '{text}' (OCR score: {score:.2f})")
        if text:
            plate_texts.append(text)

    return annotated, plate_texts


class DetectorViewer:
    """Display detector results and handle keyboard navigation."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.action = tk.StringVar(root)
        self.photo: ImageTk.PhotoImage | None = None
        self.frame_image: Image.Image | None = None

        root.title("Car Plate Detector")
        width = min(1100, round(root.winfo_screenwidth() * 0.85))
        height = min(800, round(root.winfo_screenheight() * 0.85))
        root.geometry(f"{width}x{height}")
        root.minsize(min(480, width), min(320, height))
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.bind("<Key>", self._on_key)

        self.canvas = tk.Canvas(root, highlightthickness=0, background="black")
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.CENTER)
        self.canvas.bind("<Button-1>", self._focus_canvas)
        self.canvas.bind("<Configure>", self._render_frame)

        footer = tk.Frame(root)
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(
            footer,
            text="Plate number (select & copy):",
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(6, 0))
        self.entry = tk.Entry(
            footer,
            font=("Menlo", 16),
            state="readonly",
            readonlybackground="white",
            fg="black",
            disabledforeground="black",
        )
        self.entry.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.focus_set()

    def show(self, annotated: np.ndarray, plate_texts: list[str]) -> None:
        """Update the image and recognized plate text."""
        self._set_plate_text("   |   ".join(plate_texts))

        rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self.frame_image = Image.fromarray(rgb_image)
        self._render_frame()

    def _render_frame(self, event: tk.Event | None = None) -> None:
        """Fit the current frame to the canvas without changing window size."""
        if self.frame_image is None:
            return
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return

        image = self.frame_image.copy()
        image.thumbnail((width, height), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.coords(self.image_item, width / 2, height / 2)
        self.canvas.itemconfigure(self.image_item, image=self.photo)

    def wait_for_action(self, timeout_ms: int | None = None) -> str:
        """Handle UI events until a key is pressed or the next frame is due."""
        self.action.set("")
        timer = None
        if timeout_ms is not None:
            timer = self.root.after(timeout_ms, lambda: self.action.set("tick"))
        try:
            self.root.wait_variable(self.action)
            return self.action.get()
        finally:
            if timer is not None:
                self.root.after_cancel(timer)

    def quit(self) -> None:
        self.action.set("quit")

    def close(self) -> None:
        self.root.destroy()

    def _set_plate_text(self, text: str) -> None:
        self.entry.configure(state="normal")
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)
        self.entry.configure(state="readonly")
        self.entry.selection_range(0, tk.END)

    def _on_key(self, event: tk.Event) -> None:
        if event.keysym == "Escape":
            self.quit()
            return
        if event.widget is self.entry:
            return
        self.action.set("quit" if event.keysym in {"q", "Escape"} else "advance")

    def _focus_canvas(self, event: tk.Event) -> None:
        del event
        self.canvas.focus_set()


def process_video(
    path: Path, detector: YOLO, recognizer: TextRecognition, viewer: DetectorViewer,
    *, verbose: bool = False,
) -> str:
    """Sample video frames for detection and OCR at up to VIDEO_MAX_FPS."""
    capture = cv2.VideoCapture(str(path))
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
            result = detector.predict(source=frame, verbose=False)[0]
            annotated, plate_texts = annotate_result(result, recognizer, verbose=verbose)
            viewer.show(annotated, plate_texts)
            delay_ms = max(1, round((frame_seconds - (monotonic() - started)) * 1000))
            action = viewer.wait_for_action(timeout_ms=delay_ms)
            if action != "tick":
                return action
            started = monotonic()
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="View car and license plate detections.")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-frame detections and OCR details.",
    )
    args = parser.parse_args()

    media = find_media(TEST_DIR)
    if not media:
        print(f"No images or videos found in {TEST_DIR}")
        return

    detector = YOLO(str(MODEL_PATH))
    recognizer = TextRecognition(
        model_name="PP-OCRv6_medium_rec", model_dir=str(OCR_MODEL_DIR)
    )
    root = tk.Tk()
    viewer = DetectorViewer(root)

    print(f"Found {len(media)} image/video file(s) in {TEST_DIR}")
    try:
        for media_path in media:
            print(f"\nProcessing: {media_path.name}")
            if media_path.suffix.lower() in VIDEO_EXTENSIONS:
                print(
                    f"  Video processes at up to {VIDEO_MAX_FPS} FPS; "
                    "any key = next file, 'q'/ESC = quit"
                )
                if process_video(
                    media_path, detector, recognizer, viewer, verbose=args.verbose
                ) == "quit":
                    break
                continue

            result = detector.predict(source=str(media_path), verbose=False)[0]
            annotated, plate_texts = annotate_result(result, recognizer, verbose=args.verbose)
            viewer.show(annotated, plate_texts)

            print(
                "  Select the plate number and Cmd+C/Ctrl+C to copy; "
                "any key = next image, 'q'/ESC = quit"
            )
            if viewer.wait_for_action() == "quit":
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
