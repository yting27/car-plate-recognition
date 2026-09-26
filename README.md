# Malaysia Car Plate Detection

## Run detection

Place images or videos in `dataset/test/`, then run:

```bash
.venv/bin/python inference.py
```

Add `--verbose` to print frame numbers, detection confidence, and OCR details.

The viewer shows car/plate boxes and recognized plate text. Images wait for a
key press; videos play automatically with detection and OCR capped at 10 FPS.
Intermediate video frames are skipped, and slower hardware may reduce playback
speed. Adjust `VIDEO_MAX_FPS` in `car_plate_detector/config.py` to change the cap.

Click the image and press any key to advance to the next file, or press `q`/Escape
to quit. Plate text and OCR confidence appear directly on the image.
Images and videos share blue car boxes and green plate boxes. Labels use white
text on a solid background matching their box color.
Car labels sit above the car box and include the associated plate reading;
videos additionally show a track ID and confirmation status. In images, a plate outside all
car boxes shows its accepted OCR reading above its own box. In videos, plate
text is shown only after confirmation on a car track.
For both images and videos, when multiple plate detections belong to one car,
only the detection with the highest YOLO confidence is drawn and sent to OCR.
Equal scores keep the first detection. OCR confidence does not select the box.

Videos use ByteTrack to keep a car ID across frames. Plate centers must fall
inside a car box to contribute OCR readings. If several car boxes contain a
plate center, the highest-YOLO-confidence car wins (first detection on ties).
Competing car boxes are removed before tracking and drawing unless selected
for another plate. Cars without any plate candidates remain visible. Each car keeps its last
10 usable readings (OCR confidence at least 0.90); majority voting selects the
text, with summed confidence breaking ties. Three matching reads and a strict
majority mark the result as confirmed; until then, the plate text and OCR score
are hidden. All image and video readings below 0.90 are rejected.
Only one reading per car per processed frame contributes a vote.
The viewer shows PaddleOCR confidence as `OCR` for images and `OCR avg` for
videos. The video score averages only the recent readings matching the selected
text; it is not a probability that the combined result is correct.

Readings survive temporary plate disappearance while the car remains tracked.
Missing cars are hidden, and their history is retained for up to 30 processed
frames for recovery. IDs and history reset for each video. Images still use
single-image OCR. Settings are in `car_plate_detector/config.py`; tracking IDs can switch
when cars overlap or disappear, and repeated OCR mistakes can still win a vote.

Tracking requires `lap>=0.5.12` in addition to the existing dependencies.


## Code layout

```text
inference.py                 # Entry point: python inference.py [--verbose]
car_plate_detector/
    app.py                   # CLI, model loading, and application flow
    config.py                # Paths, FPS, OCR thresholds, and drawing settings
    drawing.py               # Shared box colors, label formatting, and placement
    media.py                 # File discovery and sampled video playback
    ocr.py                   # Plate cropping and PaddleOCR recognition
    pipeline.py              # Detection annotation and OCR/tracking integration
    tracking.py              # ByteTrack, plate association, and OCR voting
    viewer.py                # Tkinter display and keyboard controls
train.py                     # Standalone model training command
convert_polygons.py          # Standalone dataset conversion command
tests/                       # Tracking and pipeline regression tests
```

The application sends images directly to the annotation pipeline. For each
video, playback creates a fresh tracker and passes it to the same pipeline for
every sampled frame. The viewer displays the resulting annotated image.
All dataset and model paths are resolved from the project root.

Run the tests without opening the viewer or loading model weights:

```bash
.venv/bin/python -m unittest discover -v
```
