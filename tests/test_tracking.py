import unittest

import numpy as np
from ultralytics.engine.results import Results

from car_plate_detector.config import TRACK_BUFFER
from car_plate_detector.tracking import PlateHistory, VideoPlateTracker, plate_owner


def frame(rows):
    return Results(
        np.zeros((400, 600, 3), dtype=np.uint8),
        path="test",
        names={0: "car", 1: "car plate"},
        boxes=np.asarray(rows, dtype=np.float32).reshape(-1, 6),
    )


class TrackingTests(unittest.TestCase):
    def test_consensus_and_replacement(self):
        history = PlateHistory()
        history.add("noise", 0.2)
        self.assertEqual(history.result(), ("", False, 0.0))
        for text in ["abc 123", "ABC-123", "ABCI23", "ABC123"]:
            history.add(text, 0.9)
        self.assertEqual(history.result(), ("ABC123", True, 0.9))
        for _ in range(10):
            history.add("NEW456", 0.95)
        self.assertEqual(history.result()[:2], ("NEW456", True))
        self.assertAlmostEqual(history.result()[2], 0.95)

    def test_confidence_averages_only_winning_readings(self):
        history = PlateHistory()
        history.add("ABC123", 0.9)
        history.add("ABC123", 1.0)
        history.add("OTHER", 0.99)
        text, confirmed, score = history.result()
        self.assertEqual((text, confirmed), ("ABC123", False))
        self.assertAlmostEqual(score, 0.95)

    def test_overlapping_and_orphan_plates(self):
        self.assertIsNone(plate_owner([1, 1, 3, 3], {}, {}))
        self.assertEqual(
            plate_owner(
                [1, 1, 3, 3], {1: [0, 0, 5, 5], 2: [0, 0, 6, 6]}, {1: 0.8, 2: 0.95}
            ),
            2,
        )

    def test_tracking_history_recovery_and_reset(self):
        tracker = VideoPlateTracker()
        for step in range(4):
            tracker.update(
                frame(
                    [
                        [10 + step, 10, 210 + step, 210, 0.9, 0],
                        [350 - step, 10, 550 - step, 210, 0.9, 0],
                        [40, 150, 80, 170, 0.9, 1],
                    ]
                )
            )
            ids = list(tracker.cars)
            self.assertEqual(len(ids), 2)
            tracker.add_readings(
                [
                    ([40, 150, 80, 170], "ABC123", 0.9),
                    ([40, 150, 80, 170], "ABC123", 0.8),
                    ([400, 150, 440, 170], "XYZ789", 0.9),
                ]
            )
        first_id, second_id = ids
        self.assertEqual(len(tracker.histories[first_id].readings), 4)
        self.assertEqual(tracker.histories[first_id].result(), ("ABC123", True, 0.9))
        self.assertEqual(tracker.histories[second_id].result(), ("XYZ789", True, 0.9))
        tracker.update(frame([]))
        self.assertEqual(tracker.labels(), {})
        tracker.update(frame([[15, 10, 215, 210, 0.9, 0]]))
        self.assertIn(first_id, tracker.cars)
        self.assertIn("ABC123", tracker.labels()[first_id][0])
        for _ in range(TRACK_BUFFER + 2):
            tracker.update(frame([]))
        self.assertEqual(tracker.histories, {})
        fresh = VideoPlateTracker()
        fresh.update(frame([[10, 10, 210, 210, 0.9, 0]]))
        self.assertEqual(fresh.histories, {})


if __name__ == "__main__":
    unittest.main()
