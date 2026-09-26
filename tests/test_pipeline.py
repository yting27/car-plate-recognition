"""Check OCR output and tracking integration without loading model weights."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import numpy as np
from ultralytics.engine.results import Results

from car_plate_detector.pipeline import (
    annotate_result,
    select_car_detections,
    select_plate_detections,
)
from car_plate_detector.tracking import VideoPlateTracker


def result():
    return Results(
        np.zeros((240, 320, 3), dtype=np.uint8),
        path="test",
        names={0: "car", 1: "car plate"},
        boxes=np.array(
            [[10, 10, 300, 220, 0.9, 0], [100, 150, 180, 180, 0.9, 1]], dtype=np.float32
        ),
    )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.ocr = Mock()
        self.ocr.predict.return_value = [{"rec_text": "ABC123", "rec_score": 0.92}]

    def test_image_confidence_and_verbose_output(self):
        output = io.StringIO()
        original = result()
        with redirect_stdout(output):
            image, labels = annotate_result(original, self.ocr)
        self.assertEqual(labels, ["Car: ABC123 (OCR: 0.92)"])
        self.assertEqual(output.getvalue(), "")
        self.assertTrue(np.any(image))
        self.assertFalse(np.any(original.orig_img))
        with redirect_stdout(output):
            annotate_result(original, self.ocr, verbose=True)
        self.assertIn("OCR score: 0.92", output.getvalue())

    def test_highest_yolo_plate_only_reaches_ocr_and_drawing(self):
        for video in (False, True):
            with self.subTest(video=video):
                detected = result()
                rows = np.array(
                    [
                        [10, 10, 300, 220, 0.9, 0],
                        [30, 150, 80, 180, 0.6, 1],
                        [100, 150, 180, 180, 0.95, 1],
                    ],
                    dtype=np.float32,
                )
                detected.update(boxes=rows)
                self.ocr.reset_mock()
                with (
                    patch("car_plate_detector.pipeline.draw_box") as draw_box,
                    patch(
                        "car_plate_detector.pipeline.crop_plate",
                        return_value=np.zeros((10, 10, 3)),
                    ) as crop,
                ):
                    annotate_result(
                        detected,
                        self.ocr,
                        tracking=VideoPlateTracker() if video else None,
                    )
                self.ocr.predict.assert_called_once()
                crop.assert_called_once()
                np.testing.assert_allclose(crop.call_args.args[1].xyxy[0], rows[2, :4])
                self.assertEqual(draw_box.call_count, 2)
                self.assertEqual(
                    draw_box.call_args_list[1].args[1], rows[2, :4].tolist()
                )

    def test_plate_selection_is_per_car_and_preserves_unassociated_boxes(self):
        detected = result()
        detected.update(
            boxes=np.array(
                [
                    [20, 20, 40, 40, 0.8, 1],
                    [25, 25, 45, 45, 0.8, 1],
                    [120, 20, 140, 40, 0.7, 1],
                    [125, 25, 145, 45, 0.9, 1],
                    [250, 20, 280, 40, 0.6, 1],
                ],
                dtype=np.float32,
            )
        )
        cars = {1: [0, 0, 100, 100], 2: [100, 0, 200, 100]}
        self.assertEqual(
            select_plate_detections(
                detected.boxes, detected.names, cars, {1: 0.9, 2: 0.9}
            ),
            {0, 3, 4},
        )

    def test_overlapping_cars_keep_highest_confidence_containing_plate(self):
        for video in (False, True):
            with self.subTest(video=video):
                detected = result()
                rows = np.array(
                    [
                        [10, 10, 300, 220, 0.7, 0],
                        [20, 20, 290, 210, 0.95, 0],
                        [0, 0, 50, 50, 0.99, 0],
                        [100, 150, 180, 180, 0.9, 1],
                    ],
                    dtype=np.float32,
                )
                detected.update(boxes=rows)
                tracker = VideoPlateTracker() if video else None
                with patch("car_plate_detector.pipeline.draw_box") as draw_box:
                    _, labels = annotate_result(detected, self.ocr, tracking=tracker)
                drawn_cars = [
                    call.args[1]
                    for call in draw_box.call_args_list
                    if call.args[2] == "car"
                ]
                self.assertEqual(
                    drawn_cars, [rows[1, :4].tolist(), rows[2, :4].tolist()]
                )
                if video:
                    self.assertEqual(labels, [])
                else:
                    self.assertEqual(len(labels), 1)
                    self.assertTrue(labels[0].startswith("Car"))
                    self.assertIn("ABC123", labels[0])
                if tracker is not None:
                    self.assertEqual(len(tracker.cars), 2)
                    self.assertEqual(len(tracker.histories), 1)

    def test_overlapping_cars_with_different_plates_are_preserved(self):
        detected = result()
        rows = np.array(
            [
                [0, 0, 200, 220, 0.9, 0],
                [100, 0, 300, 220, 0.8, 0],
                [120, 150, 140, 180, 0.9, 1],
                [240, 150, 270, 180, 0.9, 1],
            ],
            dtype=np.float32,
        )
        detected.update(boxes=rows)
        np.testing.assert_allclose(select_car_detections(detected).boxes.data, rows)

    def test_threshold_on_every_display_path(self):
        for video in (False, True):
            for associated in (False, True):
                for score in (0.899, 0.9, float("nan"), float("inf"), 1.1):
                    with self.subTest(video=video, associated=associated, score=score):
                        detected = result()
                        if not associated:
                            detected.boxes = detected.boxes[1:]
                        self.ocr.predict.return_value = [
                            {"rec_text": "ABC123", "rec_score": score}
                        ]
                        tracker = VideoPlateTracker() if video else None
                        with patch(
                            "car_plate_detector.pipeline.draw_label"
                        ) as draw_label:
                            for _ in range(3):
                                _, labels = annotate_result(
                                    detected, self.ocr, tracking=tracker
                                )
                        should_show = score == 0.9 and (not video or associated)
                        self.assertEqual(bool(labels), should_show)
                        if not should_show:
                            self.assertFalse(
                                any(
                                    "ABC123" in call.args[2]
                                    for call in draw_label.call_args_list
                                )
                            )
                        if tracker and score != 0.9:
                            self.assertFalse(tracker.histories)

    def test_video_confirmation_and_retained_reading(self):
        tracker = VideoPlateTracker()
        for index in range(3):
            _, labels = annotate_result(result(), self.ocr, tracking=tracker)
            if index < 2:
                self.assertEqual(labels, [])
            else:
                self.assertIn("ABC123 (OCR avg: 0.92)", labels[0])
        car_only = result()
        car_only.boxes = car_only.boxes[:1]
        _, labels = annotate_result(car_only, self.ocr, tracking=tracker)
        self.assertIn("ABC123 (OCR avg: 0.92)", labels[0])
        self.assertEqual(self.ocr.predict.call_count, 3)


if __name__ == "__main__":
    unittest.main()
