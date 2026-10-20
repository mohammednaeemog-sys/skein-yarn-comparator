"""Create a contact sheet to review many Skein yarn mask overlays at once.

Usage:
  python make_yarn_review_sheet.py --overlays data/yarn_masks/auto_review/overlays --out data/yarn_masks/auto_review/contact_sheet.jpg
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a contact sheet of yarn mask overlays")
    parser.add_argument("--overlays", type=Path, default=Path("data/yarn_masks/auto_review/overlays"))
    parser.add_argument("--out", type=Path, default=Path("data/yarn_masks/auto_review/contact_sheet.jpg"))
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--thumb-height", type=int, default=210)
    args = parser.parse_args()

    if not args.overlays.is_dir():
        print(f"Overlay folder not found: {args.overlays}")
        return 2
    files = sorted(p for p in args.overlays.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        print(f"No overlay images found in: {args.overlays}")
        return 2
    cols = max(1, args.columns)
    tw, th = max(160, args.thumb_width), max(120, args.thumb_height)
    cell_h = th + 50
    rows = (len(files) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * tw, 3), 245, dtype=np.uint8)
    for i, path in enumerate(files):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable overlay: {path.name}")
            continue
        h, w = image.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        thumb = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        x = (i % cols) * tw + (tw - nw) // 2
        y0 = (i // cols) * cell_h
        y = y0 + (th - nh) // 2
        sheet[y:y+nh, x:x+nw] = thumb
        name = path.stem.replace("_overlay", "")
        max_chars = max(20, int(tw / 8))
        if len(name) > max_chars:
            name = name[:max_chars-3] + "..."
        cv2.putText(sheet, f"{i+1}. {name}", ((i % cols) * tw + 8, y0 + th + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 45, 35), 1, cv2.LINE_AA)
        cv2.putText(sheet, "Green = proposed yarn-body mask", ((i % cols) * tw + 8, y0 + th + 41), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (75, 75, 75), 1, cv2.LINE_AA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), sheet):
        print(f"Could not write contact sheet: {args.out}")
        return 1
    print(f"Overlays included: {len(files)}")
    print(f"Contact sheet saved: {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
