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
speed. Adjust `VIDEO_MAX_FPS` in `inference.py` to change the cap.

Click the image and press any key to advance to the next file, or press `q`/Escape
to quit. The plate text field supports selection and copying; Escape also quits
while that field has focus.
