"""Save visual overlays comparing pilot reference masks with Skein segmentation methods.

Run from the inner Skein project folder, e.g.:
python preview_skein_masks.py --data-dir data\\yarn_masks\\pilot_v1 --dark-yarn --unetplusplus-checkpoint models\\pilot_unetplusplus_smoke.pth

This is for visual review only; it does not validate diameter accuracy.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from yarn_measurement import MeasurementConfig, segment_yarn
from yarn_ml import load_trained_model

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def draw_contours(image_rgb: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    out = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color_bgr, thickness, lineType=cv2.LINE_AA)
    return out


def label_panel(image_bgr: np.ndarray, label: str, width: int = 420, height: int = 315) -> np.ndarray:
    panel = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    header_h = 40
    canvas = np.full((height + header_h, width, 3), 248, dtype=np.uint8)
    canvas[header_h:, :] = panel
    cv2.putText(canvas, label[:52], (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (25, 25, 25), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/yarn_masks/pilot_v1"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/mask_review"))
    ap.add_argument("--dark-yarn", action="store_true", help="Yarn is darker than the background.")
    ap.add_argument("--unetplusplus-checkpoint", type=Path, default=None)
    ap.add_argument("--deeplabv3plus-checkpoint", type=Path, default=None)
    args = ap.parse_args()

    image_dir = args.data_dir / "images" / "test"
    mask_dir = args.data_dir / "masks" / "test"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"Expected held-out data at {image_dir} and {mask_dir}.")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f"No images found in {image_dir}.")

    models = {}
    for arch, checkpoint in (("unetplusplus", args.unetplusplus_checkpoint), ("deeplabv3plus", args.deeplabv3plus_checkpoint)):
        if checkpoint is not None:
            if not checkpoint.is_file():
                raise SystemExit(f"Checkpoint not found: {checkpoint}")
            models[arch] = load_trained_model(arch, path=checkpoint)

    methods = ["otsu", "sauvola", *models.keys()]
    cfg_base = dict(pixels_per_mm=300.0, min_width_mm=0.001, max_width_mm=10.0,
                    light_yarn_on_dark=not args.dark_yarn)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for image_path in images:
        truth_path = mask_dir / f"{image_path.stem}.png"
        if not truth_path.is_file():
            print(f"SKIP {image_path.name}: missing truth mask {truth_path.name}")
            continue
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        truth = (np.asarray(Image.open(truth_path).convert("L")) >= 128).astype(np.uint8) * 255
        if rgb.shape[:2] != truth.shape:
            print(f"SKIP {image_path.name}: image/mask dimensions differ")
            continue

        panels = [label_panel(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "ORIGINAL")]
        gt_overlay = draw_contours(rgb, truth, (0, 0, 255), 2)  # red reference edge
        panels.append(label_panel(gt_overlay, "REFERENCE EDGE (RED)"))
        for method in methods:
            cfg = MeasurementConfig(**cfg_base, segmentation_method=method)
            pred, _, warnings, diag = segment_yarn(rgb, cfg, model=models.get(method))
            if pred.shape[:2] != truth.shape:
                print(f"SKIP {image_path.name}/{method}: predicted shape {pred.shape} != reference {truth.shape}")
                continue
            overlay = draw_contours(rgb, truth, (0, 0, 255), 2)  # red reference
            overlay = draw_contours(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), pred, (0, 200, 0), 2)
            # draw_contours above outputs BGR. Convert once for display consistency is handled below.
            title = f"{method.upper()} (RED=REF, GREEN=PRED)"
            panel = label_panel(overlay, title)
            if warnings:
                cv2.putText(panel, f"warnings: {len(warnings)}", (10, panel.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 1, cv2.LINE_AA)
            panels.append(panel)
        # Align panels to the same height, then compose a single row for this test specimen.
        h = min(p.shape[0] for p in panels)
        panels = [cv2.resize(p, (420, h)) if p.shape[0] != h else p for p in panels]
        sheet = np.hstack(panels)
        out_path = args.output_dir / f"{image_path.stem}_comparison.jpg"
        cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        saved += 1
        print(f"Saved {out_path}")
    print(f"\nSaved {saved} comparison image(s) to: {args.output_dir}")
    print("Visual review only: these overlays do not validate physical diameter or 0.017 mm discrimination.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
