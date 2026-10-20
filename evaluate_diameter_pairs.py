"""Evaluate estimated widths and pairwise decisions against independent references.

Input CSV needs columns: image_A_filename, image_B_filename,
reference_A_width_um, reference_B_width_um, scale_px_per_mm. It can be a copy of
validation_template.csv populated with independent measurements. Each row is a
paired comparison; document specimen/session IDs and avoid treating repeat images
from one specimen as independent source samples.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from yarn_measurement import MeasurementConfig, segment_yarn, estimate_widths, summarize_widths
from yarn_ml import available_models, load_trained_model


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def width_um(image: np.ndarray, scale_px_per_mm: float, method: str, dark_yarn: bool, model):
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cfg = MeasurementConfig(
        pixels_per_mm=scale_px_per_mm, min_width_mm=0.001, max_width_mm=10.0,
        light_yarn_on_dark=not dark_yarn, segmentation_method=method,
    )
    mask, processed, warnings, diagnostics = segment_yarn(image, cfg, model=model)
    widths, _, width_warnings = estimate_widths(mask, cfg, gray=processed)
    summary = summarize_widths(widths)
    return float(summary["median_mm"] * 1000.0), warnings + width_warnings, diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True, help="CSV filled with independently reference-measured pairs")
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--method", choices=["otsu", "sauvola", "unetplusplus", "deeplabv3plus"], default="otsu")
    p.add_argument("--decision-threshold-um", type=float, default=8.0, help="Pairs above this predicted difference are flagged DIFFERENT")
    p.add_argument("--target-difference-um", type=float, default=17.0, help="Reference differences at/above this are true positives for sensitivity")
    p.add_argument("--dark-yarn", action="store_true")
    p.add_argument("--output", type=Path, default=Path("reports/diameter_validation.csv"))
    args = p.parse_args()
    if args.decision_threshold_um < 0 or args.target_difference_um <= 0:
        raise SystemExit("Difference thresholds must be positive (decision may be zero).")
    model = load_trained_model(args.method) if args.method in {"unetplusplus", "deeplabv3plus"} else None
    if args.method in {"unetplusplus", "deeplabv3plus"} and args.method not in available_models():
        raise SystemExit(f"No local {args.method} checkpoint found. Train it first using ML_MODEL_SELECTION.md.")

    rows = []
    with args.csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"image_A_filename", "image_B_filename", "reference_A_width_um", "reference_B_width_um", "scale_px_per_mm"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        for index, row in enumerate(reader, start=1):
            if not row.get("image_A_filename") or row["image_A_filename"].startswith("#"):
                continue
            try:
                path_a = args.images_dir / row["image_A_filename"]
                path_b = args.images_dir / row["image_B_filename"]
                scale = float(row["scale_px_per_mm"])
                ref_a = float(row["reference_A_width_um"])
                ref_b = float(row["reference_B_width_um"])
                img_a, img_b = read_rgb(path_a), read_rgb(path_b)
                est_a, warn_a, diag_a = width_um(img_a, scale, args.method, args.dark_yarn, model)
                est_b, warn_b, diag_b = width_um(img_b, scale, args.method, args.dark_yarn, model)
                ref_diff = abs(ref_a - ref_b)
                est_diff = abs(est_a - est_b)
                true_different = ref_diff >= args.target_difference_um
                predicted_different = est_diff > args.decision_threshold_um
                rows.append({
                    **row,
                    "method": args.method,
                    "estimated_A_width_um": est_a,
                    "estimated_B_width_um": est_b,
                    "A_error_um": est_a - ref_a,
                    "B_error_um": est_b - ref_b,
                    "A_abs_error_um": abs(est_a - ref_a),
                    "B_abs_error_um": abs(est_b - ref_b),
                    "reference_difference_um_calculated": ref_diff,
                    "estimated_difference_um": est_diff,
                    "pair_difference_abs_error_um": abs(est_diff - ref_diff),
                    "true_different_at_target": true_different,
                    "predicted_different": predicted_different,
                    "correct_pair_screen": true_different == predicted_different,
                    "quality_warnings": " | ".join(warn_a + warn_b),
                    "touches_border": bool(diag_a.get("touches_image_border") or diag_b.get("touches_image_border")),
                })
            except Exception as exc:
                rows.append({**row, "method": args.method, "evaluation_error": str(exc)})
                print(f"Row {index}: evaluation failed: {exc}")

    if not rows:
        raise SystemExit("No usable validation rows were found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for r in rows for key in r})
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    ok = [r for r in rows if "evaluation_error" not in r]
    if ok:
        mae_a = float(np.mean([float(r["A_abs_error_um"]) for r in ok]))
        mae_b = float(np.mean([float(r["B_abs_error_um"]) for r in ok]))
        mae_pair = float(np.mean([float(r["pair_difference_abs_error_um"]) for r in ok]))
        positives = [r for r in ok if r["true_different_at_target"]]
        false_same = sum(not r["predicted_different"] for r in positives)
        sensitivity = 1.0 - false_same / len(positives) if positives else float("nan")
        print(f"Evaluated pairs: {len(ok)} / {len(rows)}")
        print(f"MAE width A: {mae_a:.2f} µm | width B: {mae_b:.2f} µm | pair-difference MAE: {mae_pair:.2f} µm")
        print(f"Sensitivity for reference differences >= {args.target_difference_um:.1f} µm: {sensitivity:.3f} (positives={len(positives)})")
        print(f"Output: {args.output}")
        print("Interpret only by independent physical specimen/source; repeated rows/captures are correlated.")

if __name__ == "__main__":
    main()
