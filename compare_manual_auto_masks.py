#!/usr/bin/env python3
"""Create a review contact sheet comparing manual and automatic yarn-body masks."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def key_from_mask(p: Path) -> str:
    stem = p.stem
    if stem.endswith('_mask'):
        stem = stem[:-5]
    if stem.endswith(' - Copy'):
        stem = stem[:-7]
    return stem

def find_image(stem: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        for ext in EXTS:
            p = d / f'{stem}{ext}'
            if p.is_file():
                return p
        # Fallback: prefix/stem match after copy suffix normalization.
        if d.exists():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in EXTS:
                    normalized = p.stem[:-7] if p.stem.endswith(' - Copy') else p.stem
                    if normalized == stem:
                        return p
    return None

def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    binary = mask > 0
    tint = np.zeros_like(image)
    tint[:] = color
    out[binary] = cv2.addWeighted(image, 0.42, tint, 0.58, 0)[binary]
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out

def resize_fit(im: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = im.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 246, np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='data/annotation_pilot_5', help='Folder with original images')
    ap.add_argument('--image-search', action='append', default=[], help='Additional image folder; repeatable')
    ap.add_argument('--manual', default='data/yarn_masks/manual/masks')
    ap.add_argument('--auto', default='data/yarn_masks/auto_review/masks')
    ap.add_argument('--out', default='data/yarn_masks/auto_review/manual_auto_comparison.jpg')
    args = ap.parse_args()
    image_dirs = [Path(args.images), *[Path(x) for x in args.image_search]]
    manual_dir, auto_dir = Path(args.manual), Path(args.auto)
    out_path = Path(args.out)
    if not manual_dir.exists() or not auto_dir.exists():
        ap.error('Manual or automatic mask directory not found. Check --manual and --auto.')
    manual = {key_from_mask(p): p for p in manual_dir.glob('*_mask.png')}
    auto = {key_from_mask(p): p for p in auto_dir.glob('*_mask.png')}
    common = sorted(set(manual) & set(auto))
    if not common:
        ap.error('No matching manual/automatic masks found. Check filenames and directories.')
    cell_w, img_h, header_h = 440, 300, 58
    headers = ['Original', 'Manual mask', 'Automatic mask', 'Disagreement (red=manual only, blue=auto only)']
    gap = 10
    sheet = np.full((len(common) * (img_h + header_h + gap) + gap, 4 * cell_w + 5 * gap, 3), 246, np.uint8)
    report = []
    for i, key in enumerate(common):
        mp, apath = manual[key], auto[key]
        image_path = find_image(key, image_dirs)
        if image_path is None:
            report.append(f'SKIP {key}: image not found')
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mm = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        am = cv2.imread(str(apath), cv2.IMREAD_GRAYSCALE)
        if image is None or mm is None or am is None or image.shape[:2] != mm.shape or mm.shape != am.shape:
            report.append(f'SKIP {key}: unreadable image/mask or dimensions differ')
            continue
        m, a = mm > 0, am > 0
        inter = int(np.logical_and(m, a).sum())
        union = int(np.logical_or(m, a).sum())
        dice = 2 * inter / (int(m.sum()) + int(a.sum())) if (m.sum() + a.sum()) else 1.0
        iou = inter / union if union else 1.0
        mw = np.count_nonzero(m, axis=0).astype(float)
        aw = np.count_nonzero(a, axis=0).astype(float)
        left, right = int(0.05 * image.shape[1]), int(0.95 * image.shape[1])
        valid = (mw[left:right] > 0) & (aw[left:right] > 0)
        wd = abs(float(np.median(mw[left:right][valid])) - float(np.median(aw[left:right][valid]))) if valid.any() else float('nan')
        orig = image.copy()
        cv2.putText(orig, key[:55], (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(orig, key[:55], (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .62, (20,20,20), 1, cv2.LINE_AA)
        manual_overlay = overlay(image, mm, (40, 200, 40))
        auto_overlay = overlay(image, am, (255, 160, 30))
        diff = np.zeros_like(image)
        overlap = np.logical_and(m, a)
        manual_only = np.logical_and(m, ~a)
        auto_only = np.logical_and(a, ~m)
        diff[overlap] = (130, 130, 130)
        diff[manual_only] = (0, 0, 255)  # red in BGR
        diff[auto_only] = (255, 80, 0)   # blue/orange contrasting region in BGR
        diff = cv2.addWeighted(image, .20, diff, .80, 0)
        cells = [orig, manual_overlay, auto_overlay, diff]
        row_y = gap + i * (img_h + header_h + gap)
        cv2.putText(sheet, f'{key[:44]}  | IoU {iou:.3f} Dice {dice:.3f} | median width delta {wd:.1f}px', (gap, row_y+23), cv2.FONT_HERSHEY_SIMPLEX, .55, (25,25,25), 1, cv2.LINE_AA)
        for j, (label, cell) in enumerate(zip(headers, cells)):
            x = gap + j * (cell_w + gap)
            cv2.putText(sheet, label[:52], (x+3, row_y+47), cv2.FONT_HERSHEY_SIMPLEX, .46, (40,40,40), 1, cv2.LINE_AA)
            fitted = resize_fit(cell, cell_w, img_h)
            sheet[row_y+header_h:row_y+header_h+img_h, x:x+cell_w] = fitted
        report.append(f'{key}: IoU={iou:.3f}, Dice={dice:.3f}, median_width_delta={wd:.1f}px')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), sheet):
        raise RuntimeError(f'Failed to save {out_path}')
    print(f'Matching pairs: {len(common)}')
    print(f'Comparison sheet saved: {out_path}')
    for line in report:
        print(line)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
