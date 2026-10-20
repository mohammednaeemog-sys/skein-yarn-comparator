"""Preview YOLO polygon-segmentation annotations over an image.

Usage (PowerShell, from project root):
    python preview_yolo_segmentation.py --image "data/raw_photos/IMAGE.jpg" --label "data/raw_photos/IMAGE.txt"

Each label line must be: class_id x1 y1 x2 y2 ...
Coordinates are normalized to [0, 1]. The script saves label_preview.jpg beside the image.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys
import cv2
import numpy as np

CLASS_NAMES = {0: "loop_fiber", 1: "protruding_fiber"}
COLORS = {0: (60, 200, 255), 1: (80, 220, 90)}  # BGR


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay YOLO polygon labels on an image.")
    parser.add_argument("--image", required=True, help="Path to original image")
    parser.add_argument("--label", required=True, help="Path to YOLO polygon .txt label")
    parser.add_argument("--output", default=None, help="Output image path (default: label_preview.jpg beside source image)")
    args = parser.parse_args()

    image_path, label_path = Path(args.image), Path(args.label)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}", file=sys.stderr)
        return 2
    if not label_path.is_file():
        print(f"ERROR: Could not find label file: {label_path}", file=sys.stderr)
        return 2

    h, w = image.shape[:2]
    overlay = image.copy()
    outline = image.copy()
    counts: Counter[int] = Counter()
    errors = []

    lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        bits = line.split()
        try:
            cls = int(bits[0])
            coords = np.asarray([float(v) for v in bits[1:]], dtype=np.float64)
        except (ValueError, IndexError):
            errors.append(f"line {line_no}: non-numeric class/coordinate")
            continue
        if len(coords) < 6 or len(coords) % 2:
            errors.append(f"line {line_no}: expected class plus at least 3 x/y pairs; got {len(coords)} coordinate values")
            continue
        pts = coords.reshape(-1, 2)
        if not np.isfinite(pts).all() or (pts < -0.001).any() or (pts > 1.001).any():
            errors.append(f"line {line_no}: coordinates are not finite normalized values in [0,1]")
            continue
        pts[:, 0] = np.clip(pts[:, 0] * w, 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1] * h, 0, h - 1)
        polygon = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        color = COLORS.get(cls, (255, 150, 60))
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(outline, [polygon], True, color, 2, cv2.LINE_AA)
        counts[cls] += 1
        x, y = polygon.reshape(-1, 2)[0]
        label = CLASS_NAMES.get(cls, f"class_{cls}")
        cv2.putText(outline, label, (int(x), max(20, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Blend fill over the original and then put the bright boundaries on top.
    blended = cv2.addWeighted(overlay, 0.32, image, 0.68, 0)
    # Add outlines and readable class tags after blending.
    for line_no, raw in enumerate(lines, start=1):
        bits = raw.strip().split()
        if not bits:
            continue
        try:
            cls = int(bits[0]); coords = np.asarray([float(v) for v in bits[1:]], dtype=np.float64)
            if len(coords) < 6 or len(coords) % 2 or not np.isfinite(coords).all() or (coords < -0.001).any() or (coords > 1.001).any():
                continue
            pts = coords.reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0] * w, 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1] * h, 0, h - 1)
            polygon = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            color = COLORS.get(cls, (255, 150, 60))
            cv2.polylines(blended, [polygon], True, color, 2, cv2.LINE_AA)
            x, y = polygon.reshape(-1, 2)[0]
            tag = CLASS_NAMES.get(cls, f"class_{cls}")
            cv2.putText(blended, tag, (int(x), max(20, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        except Exception:
            pass

    output_path = Path(args.output) if args.output else image_path.with_name("label_preview.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), blended):
        print(f"ERROR: Could not write output: {output_path}", file=sys.stderr)
        return 2

    print(f"Image: {image_path} ({w} x {h})")
    print(f"Label file: {label_path}")
    print(f"Polygons parsed: {sum(counts.values())}")
    for cls, count in sorted(counts.items()):
        print(f"  class {cls} ({CLASS_NAMES.get(cls, 'unknown')}): {count}")
    if errors:
        print(f"Warnings: {len(errors)}")
        for err in errors[:10]:
            print(f"  - {err}")
    print(f"Preview saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
