"""Per-video car tracking and plate-reading consensus."""

from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Boxes, Results
from ultralytics.trackers.byte_tracker import BYTETracker

from .config import CONFIRM_READS, HISTORY_SIZE, TRACK_BUFFER
from .drawing import car_label
from .ocr import usable_reading


def containing_cars(
    plate: Sequence[float], cars: Mapping[int, Sequence[float]]
) -> list[int]:
    """Return cars containing the plate center."""
    x1, y1, x2, y2 = plate
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [
        track_id
        for track_id, (left, top, right, bottom) in cars.items()
        if left <= cx <= right and top <= cy <= bottom
    ]


def plate_owner(
    plate: Sequence[float],
    cars: Mapping[int, Sequence[float]],
    confidences: Mapping[int, float],
) -> int | None:
    """Choose the highest-confidence car containing the plate center."""
    owners = containing_cars(plate, cars)
    return max(owners, key=confidences.__getitem__) if owners else None


class PlateHistory:
    """Maintain a bounded set of OCR votes for one car."""

    def __init__(self) -> None:
        self.readings: deque[tuple[str, float]] = deque(maxlen=HISTORY_SIZE)

    def add(self, text: str, score: float) -> None:
        text = "".join(text.upper().split()).replace("-", "")
        if usable_reading(text, score):
            self.readings.append((text, score))

    def result(self) -> tuple[str, bool, float]:
        """Return winning text, confirmation, and its mean PaddleOCR score."""
        if not self.readings:
            return "", False, 0.0
        counts = Counter(text for text, _ in self.readings)
        scores = defaultdict(float)
        for text, score in self.readings:
            scores[text] += score
        winner = max(counts, key=lambda text: (counts[text], scores[text]))
        confirmed = (
            counts[winner] >= CONFIRM_READS and counts[winner] > len(self.readings) / 2
        )
        return winner, confirmed, scores[winner] / counts[winner]


class VideoPlateTracker:
    """Own car tracks and OCR histories for a single video."""

    def __init__(self) -> None:
        self.tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                new_track_thresh=0.25,
                track_buffer=TRACK_BUFFER,
                match_thresh=0.8,
                fuse_score=True,
            )
        )
        self.histories: dict[int, PlateHistory] = {}
        self.cars: dict[int, list[float]] = {}
        self.association_boxes: dict[int, list[float]] = {}
        self.association_confidences: dict[int, float] = {}

    def update(self, result: Results) -> None:
        """Match cars in the current frame and remove expired histories."""
        boxes = result.boxes
        if boxes is None:
            boxes = Boxes(np.empty((0, 6), dtype=np.float32), result.orig_shape)
        else:
            boxes = boxes.cpu().numpy()
        car_mask = np.array(
            [result.names[int(cls)].lower() == "car" for cls in boxes.cls], dtype=bool
        )
        tracked = self.tracker.update(boxes[car_mask])
        # Associate plates against observed boxes, not smoothed predictions.
        car_boxes = boxes[car_mask]
        self.cars = {
            int(row[4]): car_boxes.xyxy[int(row[-1])].tolist() for row in tracked
        }
        ids_by_index = {int(row[-1]): int(row[4]) for row in tracked}
        self.association_boxes = {
            ids_by_index.get(index, -(index + 1)): box.tolist()
            for index, box in enumerate(car_boxes.xyxy)
        }
        self.association_confidences = {
            ids_by_index.get(index, -(index + 1)): float(score)
            for index, score in enumerate(car_boxes.conf)
        }
        alive = {
            track.track_id
            for track in self.tracker.tracked_stracks + self.tracker.lost_stracks
            if self.tracker.frame_id - track.end_frame <= TRACK_BUFFER
        }
        self.histories = {
            key: value for key, value in self.histories.items() if key in alive
        }

    def add_readings(
        self, readings: Iterable[tuple[Sequence[float], str, float]]
    ) -> None:
        """Associate plate readings and keep one vote per car for this frame."""
        # Duplicate plate boxes must not contribute multiple votes in one frame.
        best = {}
        for plate, text, score in readings:
            owner = plate_owner(
                plate, self.association_boxes, self.association_confidences
            )
            if (
                owner in self.cars
                and usable_reading(text, score)
                and (owner not in best or score > best[owner][1])
            ):
                best[owner] = (text, score)
        for owner, (text, score) in best.items():
            self.histories.setdefault(owner, PlateHistory()).add(text, score)

    def labels(self) -> dict[int, tuple[str, str]]:
        """Return display labels and selected plate text for visible cars."""
        labels = {}
        for track_id in self.cars:
            history = self.histories.get(track_id)
            text, confirmed, score = history.result() if history else ("", False, 0.0)
            if not confirmed:
                text = ""
            label = car_label(text, score, track_id=track_id, confirmed=confirmed)
            labels[track_id] = (label, text)
        return labels
