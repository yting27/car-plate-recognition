# Project guidance

Keep README.md focused on the dataset source, setup, and running inference.
Maintain implementation details here. Preserve the behavior below unless the
user requests a change.

## Start here

1. Read `git status --short` and the relevant diff before editing; preserve
   existing user changes and staged files.
2. Use the file map below, then inspect the affected functions and their callers.
   Search with `rg`; exclude `.venv`, datasets, and model binaries from broad
   searches unless investigating those assets specifically.
3. Implement shared behavior once, covering both image and video paths. Keep
   model loading in `app.py`, rendering in `drawing.py`, and Tkinter in `viewer.py`.
4. Run the checks appropriate to the change and report what was actually tested.
   Update this file when changing the behavior it describes.

## Environment and commands

Use `.venv/bin/python` for this workspace. The inspected environment uses Python
3.12.12, Ultralytics 8.4.163, PaddleOCR 3.7.0, PaddlePaddle 3.2.0, OpenCV
5.0.0.93, NumPy 2.3.5, Pillow 12.3.0, and lap 0.5.13. These are an environment
snapshot; verify installed versions before relying on library internals.
`requirements.txt` pins direct runtime dependencies to these working versions,
including the CPU PaddlePaddle package. It is not a complete transitive lockfile.
Python 3.12 on macOS arm64 is the validated environment; other platforms need
their own installation and inference checks. Tkinter comes from Python/the OS,
not a pip dependency.

```bash
uv pip install --python .venv/bin/python -r requirements.txt
git lfs pull
.venv/bin/python inference.py --help
.venv/bin/python inference.py
.venv/bin/python inference.py --verbose
```

`--help` parses arguments without importing the model libraries. The viewer
requires a working Tkinter installation and a desktop session. Headless tests
can exercise `annotate_result()` without constructing a viewer.

The local virtual environment may not include pip. Use
`uv pip install --python .venv/bin/python <package>` when adding a needed package;
do not silently switch to the system interpreter. Update `requirements.txt` when
adding or upgrading direct dependencies, then check resolution and the affected
runtime paths. ByteTrack's `lap` dependency is explicitly included.

For training options, use `.venv/bin/python train.py --help`. Training expects
the separately supplied `dataset/data.yaml`; committed test samples are not a
training dataset. Inspect annotation conversion with
`.venv/bin/python convert_polygons.py --dry-run` before using its in-place mode.

## Code organization

| File | Responsibility |
| --- | --- |
| `inference.py` | Thin entry point calling `app.main()` |
| `car_plate_detector/app.py` | CLI arguments, model loading, application flow |
| `car_plate_detector/config.py` | Shared paths, thresholds, FPS, drawing settings |
| `car_plate_detector/media.py` | Media discovery and sampled video playback |
| `car_plate_detector/ocr.py` | Plate cropping, PaddleOCR, shared OCR quality gate |
| `car_plate_detector/pipeline.py` | Detection selection, OCR/tracking integration, annotation |
| `car_plate_detector/tracking.py` | Car association, ByteTrack state, OCR history and voting |
| `car_plate_detector/drawing.py` | Shared box colors, label formatting and placement |
| `car_plate_detector/viewer.py` | Tkinter display, startup focus and keyboard navigation |
| `train.py` | Standalone YOLO training command |
| `convert_polygons.py` | Standalone dataset annotation conversion command |
| `tests/` | Tracking and pipeline regression tests |
| `requirements.txt` | Pinned direct runtime dependencies |

Resolve model and dataset paths from the project root, not the working directory.
Keep image and video annotation in the shared pipeline and styling in drawing.py.

## Where to implement changes

| Task | Start with | Check alongside it |
| --- | --- | --- |
| Change input discovery or FPS | `media.find_media`, `media.process_video` | Sampling intervals, end-of-file and capture release |
| Change car/plate selection | `pipeline.select_car_detections`, `pipeline.select_plate_detections` | `tracking.plate_owner`, overlap and duplicate tests |
| Change OCR acceptance | `ocr.usable_reading`, `config.MIN_OCR_SCORE` | Image labels, orphan plates, video history and confirmation |
| Change tracking or voting | `tracking.VideoPlateTracker`, `tracking.PlateHistory` | Lost-track recovery, per-video reset, score averaging |
| Change labels/colors | `drawing.car_label`, `drawing.draw_label`, `drawing.draw_box` | Both media types, wrapping and image-edge placement |
| Change focus/navigation | `viewer.DetectorViewer` | Startup focus, timer cleanup, next/quit actions |
| Change model paths or CLI | `config.py`, `app.main` | LFS downloads and lightweight `--help` |

The call flow is `inference.py -> app.main -> annotate_result` for images, and
`app.main -> process_video -> annotate_result` for video frames. Both finish
with `viewer.show(annotated)` and `viewer.wait_for_action(...)`.

## Data contracts and integration pitfalls

- YOLO boxes expose `xyxy`, `cls`, and `conf`. `xyxy[0].tolist()` gives
  `[left, top, right, bottom]`; `.item()` converts scalar tensors to Python values.
- Detection indices used during filtering are not persistent track IDs. Do not
  carry them between frames or index the original results with reindexed values.
- ByteTrack receives only car boxes as CPU NumPy-backed `Boxes`. In the installed
  version, output rows are `[x1, y1, x2, y2, id, score, class, detection_index]`.
  The final index refers to the car-only input. Recheck this contract on upgrades.
- `association_boxes` includes observed cars awaiting tracker confirmation,
  using negative temporary keys. Only IDs in `tracker.cars` may receive OCR
  history. `association_confidences` uses the same keys and current YOLO scores.
- `annotate_result()` returns `(annotated_image, plate_labels)` and draws on a
  copy. Keep the source image intact for cropping. The viewer consumes the image;
  tests also inspect returned labels.
- `recognize_plate()` returns raw `(text, score)` from PaddleOCR's `rec_text` and
  `rec_score`. Keep quality filtering separate so verbose diagnostics can still
  expose rejected readings.
- `PlateHistory.result()` returns `(text, confirmed, mean_score)`. UI gating
  belongs in the display path; unconfirmed history is still needed for voting.
- OpenCV frames are BGR; the viewer converts to RGB for Pillow. Preserve that
  conversion and retain `self.photo` to keep Tkinter's image reference alive.
- Video actions are `tick`, `pause`, `advance`, and `quit`. Preserve timer cancellation,
  `VideoCapture.release()` in `finally`, and viewer destruction on exit.

## Detection and association

The pipeline selects car detections before updating video tracking. It then
selects plate detections before drawing boxes or running OCR.

- A car contains a plate when the plate's center lies inside the car box. Full
  containment of the plate rectangle is not required.
- If multiple cars contain a plate center, the highest YOLO car confidence wins.
  Equal scores retain the first detection.
- Suppress losing car candidates unless they win another plate. Cars without
  any plate candidates remain visible.
- Each selected car keeps only its highest-YOLO-confidence plate detection.
  Equal scores retain the first plate. Other plate detections for that car are
  neither drawn nor sent to OCR.
- YOLO confidence selects boxes; PaddleOCR confidence evaluates recognized text.
- Images and videos use a 0.40 car detection threshold and a 0.50 plate detection
  threshold. YOLO's initial cutoff is the minimum of the two; `filter_detections`
  enforces class thresholds before car selection, tracking, drawing, and OCR.
- ByteTrack tracks cars only: high/new-track thresholds are 0.25, the low
  threshold is 0.10, match threshold is 0.8, and score fusion is enabled.
  With the current 0.40 car cutoff, weak detections below that cutoff never
  reach ByteTrack's low-confidence recovery stage.

## OCR and voting

Crop each selected plate with 10% padding, clamp it to the image boundaries,
and upscale it 2x for PaddleOCR. Skip invalid crops.

Use the shared `usable_reading()` gate for display and history: nonempty text,
a finite score, and `0.80 <= score <= 1.0`.

For videos, each car has its own history of up to 10 accepted readings. Normalize
history text to uppercase and remove whitespace and hyphens. At most one reading
per car per processed frame contributes a vote. The tracker also defensively
keeps the highest OCR score if duplicate readings reach it.

Select the text with the most votes; summed OCR confidence breaks vote ties.
Confirmation requires at least three matching reads and a strict majority of
the stored readings. Hide plate text and its score until confirmed.

`OCR avg` is the mean PaddleOCR score of stored readings matching the selected
text, not car detection confidence or a calibrated probability of correctness.
Images use the current crop's score, labeled `OCR`.

An image plate outside all car boxes can display an accepted reading on its own
box. Video plates without a car association show no OCR text because they have
no car history for confirmation. Raw OCR diagnostics remain available with
`--verbose`.

## Video lifecycle and viewer

Create fresh tracking state for each video. Sample at up to 10 FPS, skipping
intermediate source frames. Slower processing can slow playback; the current
loop does not dynamically discard additional frames to maintain real-time speed.

Retain histories during temporary disappearance for up to 30 processed frames,
not source frames. Missing cars are hidden; returning matched tracks can reuse
their histories. Track IDs can switch during occlusion and are not permanent
vehicle identities.

Use blue car boxes and green plate boxes (OpenCV uses BGR). Labels have white
text on solid backgrounds matching the box color and wrap to reduce overlap.
Associated plate text appears on the car label; video labels include track IDs.
There is no bottom text field. Raise and focus the viewer once on startup.
Only Enter (Return or keypad Enter) advances to the next file. Space pauses or
resumes videos; it does nothing for images. While paused, no frames are read and
no detection/OCR runs. Enter and window close still work; resuming resets the
frame timing. Other keys, including `q` and Escape, do nothing. Closing the window quits.

Keep per-frame detection and OCR logs behind `--verbose`. File-level progress
and errors remain visible by default.

## Models and sample files

The YOLO detector was trained on the Malaysian Car Plate Dataset linked in the
README. PaddleOCR uses the pretrained `PP-OCRv6_medium_rec` model; do not attribute
its training to that dataset.

Model binaries and sample MP4 files use Git LFS. Keep `inference.json`,
`inference.pdiparams`, and `inference.yml` together for OCR loading. The model's
`.cache` contains download metadata and is not required for inference.

Only `test*` sample files in `dataset/test/` are included in Git. Other samples,
training/validation data, caches, and generated output stay ignored. Inference
still discovers all supported local media in `dataset/test/`, including ignored
samples.

## Verification

Run regression tests without opening the GUI or loading model weights:

```bash
.venv/bin/python -m unittest discover -v
uvx ruff check car_plate_detector tests
git diff --check
```

For changes to model integration or rendering, use local sample media to check
the affected path. Distinguish automated tests from real-model or GUI checks.

`tests/test_pipeline.py` uses synthetic YOLO `Results` and mocked OCR to cover
selection before OCR/drawing, overlapping cars, confidence gates, and video
confirmation. `tests/test_tracking.py` exercises the installed ByteTrack with
synthetic detections, plus voting, recovery, expiration, and fresh state.

Extend these tests when changing those behaviors. Include score-boundary cases,
empty detections, multiple cars/plates, and image/video differences where relevant.
Do not load real weights in unit tests. For a model smoke check, load the configured
models, read `dataset/test/test1.jpeg` or frames from `test2.mp4`, and call
`annotate_result()` directly. Rendered output checks do not verify GUI focus or
keyboard behavior; check those with an actual Tkinter window.

For documentation-only changes, check the diff and links; model inference is
unnecessary. Keep generated validation images and logs out of Git. Before staging
assets, inspect `.gitignore`, `.gitattributes`, and `git lfs ls-files` so weights
and sample videos remain LFS objects rather than ordinary Git blobs.
