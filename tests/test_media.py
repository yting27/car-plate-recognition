"""Video pause, resume, and exit behavior without a GUI or model weights."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from car_plate_detector.media import process_video


class VideoPlaybackTests(unittest.TestCase):
    def test_pause_resume_and_exit(self):
        for exit_action in ("advance", "quit"):
            with self.subTest(exit_action=exit_action):
                capture = Mock()
                capture.get.return_value = 10
                frame = np.zeros((20, 20, 3), dtype=np.uint8)
                capture.retrieve.return_value = (True, frame)
                detector = Mock()
                detector.predict.return_value = [Mock()]
                viewer = Mock()
                calls = 0

                def action(
                    capture=capture,
                    detector=detector,
                    exit_action=exit_action,
                    **kwargs,
                ):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        return "pause"
                    if calls == 2:
                        self.assertNotIn("timeout_ms", kwargs)
                        self.assertTrue(kwargs["allow_pause"])
                        self.assertEqual(capture.grab.call_count, 1)
                        self.assertEqual(detector.predict.call_count, 1)
                        return "pause"
                    return exit_action

                viewer.wait_for_action.side_effect = action
                with (
                    patch(
                        "car_plate_detector.media.cv2.VideoCapture",
                        return_value=capture,
                    ),
                    patch(
                        "car_plate_detector.media.annotate_result",
                        return_value=(frame, []),
                    ),
                ):
                    self.assertEqual(
                        process_video(Path("test.mp4"), detector, Mock(), viewer),
                        exit_action,
                    )
                self.assertEqual(detector.predict.call_count, 2)
                capture.release.assert_called_once()

    def test_exit_while_paused(self):
        for exit_action in ("advance", "quit"):
            with self.subTest(exit_action=exit_action):
                capture = Mock()
                capture.get.return_value = 10
                capture.retrieve.return_value = (True, np.zeros((20, 20, 3)))
                detector = Mock()
                detector.predict.return_value = [Mock()]
                viewer = Mock()
                viewer.wait_for_action.side_effect = ["pause", exit_action]
                with (
                    patch(
                        "car_plate_detector.media.cv2.VideoCapture",
                        return_value=capture,
                    ),
                    patch(
                        "car_plate_detector.media.annotate_result",
                        return_value=(np.zeros((20, 20, 3)), []),
                    ),
                ):
                    self.assertEqual(
                        process_video(Path("test.mp4"), detector, Mock(), viewer),
                        exit_action,
                    )
                self.assertEqual(capture.grab.call_count, 1)
                capture.release.assert_called_once()
