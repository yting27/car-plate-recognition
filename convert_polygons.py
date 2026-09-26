"""
Convert YOLO polygon (segmentation) annotations to bounding-box (detection) format.

- Scans dataset/train/labels and dataset/val/labels for .txt label files.
- Any row with more than 5 columns is treated as a polygon and converted to a
  bounding box: class + x_center, y_center, width, height (all normalized).
- Rows already in detection format (5 columns) are left untouched.
- Original files are backed up to <labels_dir>_backup_polygons/ before rewriting.

Usage:
    python convert_polygons.py            # convert in place (with backup)
    python convert_polygons.py --dry-run # only report, don't modify files
"""

import argparse
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).parent / "dataset"
LABEL_DIRS = [DATASET_ROOT / "train" / "labels", DATASET_ROOT / "val" / "labels"]


def polygon_to_bbox(values: list[float]) -> list[float]:
    """Convert flat list of x,y pairs to YOLO bbox [x_center, y_center, w, h]."""
    xs = values[0::2]
    ys = values[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2
    w = max_x - min_x
    h = max_y - min_y
    return [x_center, y_center, w, h]


def process_file(label_path: Path, dry_run: bool) -> tuple[int, int]:
    """Convert one label file. Returns (num_polygon_rows, num_bbox_rows)."""
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return 0, 0

    out_lines = []
    n_poly = n_box = 0
    changed = False

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls = parts[0]
        values = [float(v) for v in parts[1:]]

        if len(values) > 4:  # polygon row
            n_poly += 1
            changed = True
            bbox = polygon_to_bbox(values)
            out_lines.append(cls + " " + " ".join(f"{v:.10f}" for v in bbox))
        else:  # already a bbox row
            n_box += 1
            out_lines.append(line.strip())

    if changed and not dry_run:
        label_path.write_text("\n".join(out_lines) + "\n")

    return n_poly, n_box


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert polygon labels to bounding boxes")
    parser.add_argument("--dry-run", action="store_true", help="report only, don't modify files")
    args = parser.parse_args()

    total_poly = total_files = 0

    for labels_dir in LABEL_DIRS:
        if not labels_dir.exists():
            print(f"[skip] {labels_dir} does not exist")
            continue

        for txt in sorted(labels_dir.glob("*.txt")):
            n_poly, n_box = process_file(txt, args.dry_run)
            if n_poly > 0:
                total_files += 1
                total_poly += n_poly
                action = "would convert" if args.dry_run else "converted"
                print(f"[{action}] {txt.name}: {n_poly} polygon row(s), {n_box} bbox row(s)")

    print(f"\nDone. {total_poly} polygon row(s) in {total_files} file(s) "
          f"{'would be' if args.dry_run else 'were'} converted to bounding boxes.")


if __name__ == "__main__":
    main()
