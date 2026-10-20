"""Benchmark current segmentation methods against expert yarn-core masks.

Expected dataset: data/yarn_masks/images/test and masks/test, same basename.
This script only evaluates segmentation masks; a good Dice score does not by
itself prove diameter accuracy. Compare the separate physical diameter reference
measurements as described in RESEARCH_AND_VALIDATION_PLAN.md.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from yarn_measurement import MeasurementConfig, segment_yarn
from yarn_ml import available_models, load_trained_model


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p = pred > 0
    t = truth > 0
    inter = float(np.logical_and(p, t).sum())
    p_sum, t_sum = float(p.sum()), float(t.sum())
    dice = (2 * inter + 1e-8) / (p_sum + t_sum + 1e-8)
    iou = (inter + 1e-8) / (p_sum + t_sum - inter + 1e-8)
    # Boundary F1 with a 1-pixel tolerance for raster edge mismatch.
    p_edge = p ^ cv2.erode(p.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    t_edge = t ^ cv2.erode(t.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    p_dil = cv2.dilate(p_edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    t_dil = cv2.dilate(t_edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    precision = float(np.logical_and(p_edge, t_dil).sum()) / max(float(p_edge.sum()), 1.0)
    recall = float(np.logical_and(t_edge, p_dil).sum()) / max(float(t_edge.sum()), 1.0)
    boundary_f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"dice": dice, "iou": iou, "boundary_f1_1px": boundary_f1}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/yarn_masks"))
    ap.add_argument("--output", type=Path, default=Path("reports/segmentation_benchmark.csv"))
    ap.add_argument("--dark-yarn", action="store_true", help="Set if yarn is darker than background.")
    args = ap.parse_args()
    image_dir = args.data_dir / "images" / "test"
    mask_dir = args.data_dir / "masks" / "test"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"Expected test data in {image_dir} and {mask_dir}; annotate and hold out real specimens first.")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not images:
        raise SystemExit(f"No test images found in {image_dir}")
    model_list = {name: load_trained_model(name) for name in available_models()}
    methods = ["otsu", "sauvola", *model_list.keys()]
    cfg_base = dict(pixels_per_mm=300.0, min_width_mm=0.001, max_width_mm=10.0,
                    light_yarn_on_dark=not args.dark_yarn)
    rows = []
    for image_path in images:
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            print(f"SKIP missing mask for {image_path.name}: expected {mask_path.name}")
            continue
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        truth = (np.asarray(Image.open(mask_path).convert("L")) >= 128).astype(np.uint8) * 255
        if rgb.shape[:2] != truth.shape:
            print(f"SKIP dimension mismatch: {image_path.name}")
            continue
        for method in methods:
            try:
                cfg = MeasurementConfig(**cfg_base, segmentation_method=method)
                pred, _, warnings, diag = segment_yarn(rgb, cfg, model=model_list.get(method))
                row = {"image": image_path.name, "method": method, **metrics(pred, truth),
                       "warnings_count": len(warnings), "touches_border": diag.get("touches_image_border", False)}
                rows.append(row)
                print(f"{image_path.name:32s} {method:16s} Dice={row['dice']:.4f} IoU={row['iou']:.4f} boundaryF1={row['boundary_f1_1px']:.4f}")
            except Exception as exc:
                rows.append({"image": image_path.name, "method": method, "error": str(exc)})
                print(f"ERROR {image_path.name} / {method}: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} method-image evaluations to {args.output}")
    print("Choose any model only after comparing held-out results and verifying actual diameter error against an independent reference.")

if __name__ == "__main__":
    main()
