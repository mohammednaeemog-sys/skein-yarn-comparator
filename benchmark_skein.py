"""Repeatable Skein segmentation benchmark with explicit local checkpoint paths.

Evaluates Otsu and Sauvola plus any selected local checkpoints against held-out
expert-reviewed masks. Segmentation scores do NOT establish physical diameter
accuracy. Keep test specimens independent of training/validation specimens.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np
from PIL import Image

from yarn_measurement import MeasurementConfig, segment_yarn
from yarn_ml import available_models, load_trained_model

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p = pred > 0
    t = truth > 0
    inter = float(np.logical_and(p, t).sum())
    p_sum, t_sum = float(p.sum()), float(t.sum())
    dice = (2.0 * inter + 1e-8) / (p_sum + t_sum + 1e-8)
    iou = (inter + 1e-8) / (p_sum + t_sum - inter + 1e-8)

    # Same 1-pixel tolerant boundary F1 calculation used by the bundled benchmark.
    kernel = np.ones((3, 3), np.uint8)
    p_edge = p ^ (cv2.erode(p.astype(np.uint8), kernel).astype(bool))
    t_edge = t ^ (cv2.erode(t.astype(np.uint8), kernel).astype(bool))
    p_dil = cv2.dilate(p_edge.astype(np.uint8), kernel) > 0
    t_dil = cv2.dilate(t_edge.astype(np.uint8), kernel) > 0
    precision = float(np.logical_and(p_edge, t_dil).sum()) / max(float(p_edge.sum()), 1.0)
    recall = float(np.logical_and(t_edge, p_dil).sum()) / max(float(t_edge.sum()), 1.0)
    bf1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"dice": dice, "iou": iou, "boundary_f1_1px": bf1}


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/yarn_masks/pilot_v1"))
    ap.add_argument("--output", type=Path, default=Path("reports/pilot_benchmark_with_unet.csv"))
    ap.add_argument("--summary-output", type=Path, default=None)
    ap.add_argument("--dark-yarn", action="store_true", help="Yarn is darker than the background.")
    ap.add_argument("--unetplusplus-checkpoint", type=Path, default=None,
                    help="Explicit checkpoint path; useful for evaluating a smoke-test checkpoint.")
    ap.add_argument("--deeplabv3plus-checkpoint", type=Path, default=None,
                    help="Explicit DeepLabV3+ checkpoint path.")
    args = ap.parse_args()

    image_dir = args.data_dir / "images" / "test"
    mask_dir = args.data_dir / "masks" / "test"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"Expected held-out data at {image_dir} and {mask_dir}.")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f"No supported test images found in {image_dir}.")

    explicit = {
        "unetplusplus": args.unetplusplus_checkpoint,
        "deeplabv3plus": args.deeplabv3plus_checkpoint,
    }
    auto_available = set(available_models())
    model_list = {}
    for arch, checkpoint in explicit.items():
        if checkpoint is not None:
            if not checkpoint.is_file():
                raise SystemExit(f"Checkpoint not found: {checkpoint}")
            model_list[arch] = load_trained_model(arch, path=checkpoint)
        elif arch in auto_available:
            model_list[arch] = load_trained_model(arch)

    methods = ["otsu", "sauvola", *model_list.keys()]
    # px/mm is only an internal plausibility range used by segment_yarn here; this
    # benchmark reports segmentation masks, not physical diameters. It is not calibration.
    cfg_base = dict(pixels_per_mm=300.0, min_width_mm=0.001, max_width_mm=10.0,
                    light_yarn_on_dark=not args.dark_yarn)
    rows = []
    for image_path in images:
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            print(f"SKIP {image_path.name}: missing truth mask {mask_path.name}")
            continue
        try:
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            truth = (np.asarray(Image.open(mask_path).convert("L")) >= 128).astype(np.uint8) * 255
        except Exception as exc:
            print(f"SKIP {image_path.name}: could not read pair: {exc}")
            continue
        if rgb.shape[:2] != truth.shape:
            print(f"SKIP {image_path.name}: image/mask dimensions differ")
            continue
        for method in methods:
            try:
                cfg = MeasurementConfig(**cfg_base, segmentation_method=method)
                pred, _, warnings, diag = segment_yarn(rgb, cfg, model=model_list.get(method))
                row = {
                    "image": image_path.name,
                    "method": method,
                    **metrics(pred, truth),
                    "warnings_count": len(warnings),
                    "touches_border": bool(diag.get("touches_image_border", False)),
                    "checkpoint": str(explicit.get(method) or "bundled default" if method in model_list else "n/a"),
                }
                rows.append(row)
                print(f"{image_path.name:36s} {method:16s} "
                      f"Dice={row['dice']:.4f} IoU={row['iou']:.4f} "
                      f"boundaryF1={row['boundary_f1_1px']:.4f}")
            except Exception as exc:
                rows.append({"image": image_path.name, "method": method, "error": f"{type(exc).__name__}: {exc}"})
                print(f"ERROR {image_path.name} / {method}: {type(exc).__name__}: {exc}")

    write_csv(args.output, rows)
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.csv")
    summary_rows = []
    methods_scored = sorted({str(r.get("method")) for r in rows if "dice" in r and "error" not in r})
    for method in methods_scored:
        method_rows = [r for r in rows if r.get("method") == method and "dice" in r and "error" not in r]
        summary_rows.append({
            "method": method,
            "n_images": len(method_rows),
            "mean_dice": mean(float(r["dice"]) for r in method_rows),
            "median_dice": median(float(r["dice"]) for r in method_rows),
            "mean_iou": mean(float(r["iou"]) for r in method_rows),
            "mean_boundary_f1_1px": mean(float(r["boundary_f1_1px"]) for r in method_rows),
            "images_with_warnings": sum(int(r.get("warnings_count", 0)) > 0 for r in method_rows),
        })
    write_csv(summary_path, summary_rows)
    print(f"\nSaved per-image report: {args.output}")
    print(f"Saved method summary:   {summary_path}")
    print("These are segmentation metrics only. They do not validate physical width or 0.017 mm discrimination.")
    if any("pilot" in str(p).lower() or "smoke" in str(p).lower() for p in [args.unetplusplus_checkpoint, args.deeplabv3plus_checkpoint] if p):
        print("WARNING: a pilot/smoke checkpoint was supplied; do not treat its score as model-selection evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
