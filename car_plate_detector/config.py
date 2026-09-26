"""Shared paths, rendering settings, and recognition thresholds."""

from pathlib import Path

# Dir path
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "dataset" / "test"
MODEL_PATH = BASE_DIR / "models" / "yolo26n-v1.pt"
OCR_MODEL_DIR = BASE_DIR / "models" / "PP-OCRv6_medium_rec"

# Media settings
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
LABEL_TEXT_COLOR = (255, 255, 255)
LABEL_PADDING = 4
PLATE_PADDING = 0.1
OCR_SCALE = 2
VIDEO_MAX_FPS = 10

# Tracker
TRACK_BUFFER = 30  # Processed frames, not source frames.
HISTORY_SIZE = 10
MIN_OCR_SCORE = 0.8
CONFIRM_READS = 3


OCR_MODEL_NAME = "PP-OCRv6_medium_rec"
CAR_DETECTION_CONFIDENCE = 0.4
PLATE_DETECTION_CONFIDENCE = 0.5
DETECTION_CONFIDENCE = min(CAR_DETECTION_CONFIDENCE, PLATE_DETECTION_CONFIDENCE)
