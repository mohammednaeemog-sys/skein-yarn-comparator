"""Fast, human-reviewed yarn-body mask proposals for Skein.

Usage from the Skein project root:
  python fast_yarn_mask_review.py --images data/annotation_candidate --out data/yarn_masks/auto_review

This is a proposal tool, not a validated automatic annotator. It creates a classical
image-processing candidate mask, shows it over the original, and saves only after
an explicit human approval (Y). Review every saved overlay.

Controls:
  Y = approve/save current proposal and continue
  N = reject/skip current image
  O = toggle original / overlay
  Q or Esc = quit

Saved mask: white (255) = proposed main yarn body; black (0) = background.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_unique_images(images_dir: Path) -> list[Path]:
    paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS and not p.name.lower().startswith("label_preview")
    )
    seen: dict[str, Path] = {}
    unique: list[Path] = []
    for p in paths:
        digest = sha256_file(p)
        if digest in seen:
            print(f"Exact duplicate skipped: {p.name} (same bytes as {seen[digest].name})")
            continue
        seen[digest] = p
        unique.append(p)
    return unique


def proposal_mask(bgr: np.ndarray) -> tuple[np.ndarray | None, dict[str, object]]:
    """Propose the largest dark, mostly-horizontal body band using a robust classical heuristic."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Robustly suppress salt-and-pepper noise, then soften hair-scale detail.
    k = 9 if min(h, w) >= 9 else (min(h, w) // 2) * 2 + 1
    den = cv2.medianBlur(gray, k)
    smooth = cv2.GaussianBlur(den, (0, 0), sigmaX=max(1.2, w / 800.0), sigmaY=max(1.2, h / 800.0))

    # Main yarn is expected to occupy a broad, nearly horizontal dark band.
    row_profile = smooth.astype(np.float32).mean(axis=1)
    row_profile = cv2.GaussianBlur(row_profile.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=max(2.0, h / 220.0)).ravel()
    center = int(np.argmin(row_profile))
    pmin = float(row_profile[center])
    margin = max(10, int(h * 0.12))
    bg_samples = np.concatenate([row_profile[:margin], row_profile[-margin:]])
    background_level = float(np.median(bg_samples))
    contrast = background_level - pmin
    stats: dict[str, object] = {"center_y": center, "contrast": round(contrast, 2)}

    if contrast < 7:
        stats["warning"] = "weak horizontal-band contrast"
        return None, stats

    # Find the contiguous dark row band around the darkest row. Threshold is only
    # used to limit the search window; it is not claimed to be a physical edge.
    profile_threshold = pmin + 0.48 * contrast
    top = center
    bottom = center
    while top > 0 and row_profile[top - 1] <= profile_threshold:
        top -= 1
    while bottom < h - 1 and row_profile[bottom + 1] <= profile_threshold:
        bottom += 1

    pad = max(8, int(h * 0.012))
    top = max(0, top - pad)
    bottom = min(h - 1, bottom + pad)
    roi = smooth[top:bottom + 1, :]
    stats.update({"band_top": top, "band_bottom": bottom, "band_height": bottom - top + 1})

    if roi.shape[0] < max(8, int(h * 0.025)):
        stats["warning"] = "detected band is implausibly thin"
        return None, stats

    # Otsu separates the compact dark body from the lighter background in the
    # restricted band. Then closing fills internal texture gaps.
    _, local = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kh = max(15, min(61, (w // 45) | 1))
    kv = max(5, min(15, (h // 100) | 1))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kh, kv))
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    # Gentle horizontal opening removes isolated hair fragments without attempting
    # to define the exact yarn edge from a single intensity threshold.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, w // 100), 3))
    local = cv2.morphologyEx(local, cv2.MORPH_OPEN, open_kernel, iterations=1)

    full = np.zeros((h, w), dtype=np.uint8)
    full[top:bottom + 1, :] = local
    n, labels, component_stats, _ = cv2.connectedComponentsWithStats((full > 0).astype(np.uint8), 8)
    candidates = []
    for idx in range(1, n):
        x, y, cw, ch, area = component_stats[idx]
        if cw >= 0.45 * w and ch >= 0.025 * h and area >= 0.01 * w * h:
            # Preference for a long component close to the central dark row.
            cy = y + ch / 2
            score = float(area) * (cw / w) / (1 + abs(cy - center) / max(1, h * 0.1))
            candidates.append((score, idx))

    if not candidates:
        stats["warning"] = "no convincing wide body component; manually annotate or skip"
        return None, stats

    _, best_idx = max(candidates)
    mask = np.where(labels == best_idx, 255, 0).astype(np.uint8)

    # Fill small interior holes and smooth tiny staircase defects. Do not expand the
    # external contour aggressively because that would bias diameter upward.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        stats["warning"] = "no external contour"
        return None, stats
    largest = max(contours, key=cv2.contourArea)
    mask[:] = 0
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        stats["warning"] = "empty proposal"
        return None, stats
    xspan = int(xs.max() - xs.min() + 1)
    yspan = int(ys.max() - ys.min() + 1)
    fg = float(np.mean(mask > 0) * 100)
    border_touch = bool(xs.min() == 0 or xs.max() == w - 1 or ys.min() == 0 or ys.max() == h - 1)
    warnings = []
    if xspan < 0.75 * w:
        warnings.append("body does not span most of the image width")
    if yspan > 0.45 * h:
        warnings.append("mask unusually tall; possible flyaway fibre included")
    if fg < 2 or fg > 45:
        warnings.append("foreground area is unusual")
    if border_touch:
        warnings.append("mask touches image border / crop may be present")
    stats.update({
        "xspan_pct": round(100 * xspan / w, 1),
        "yspan_pct": round(100 * yspan / h, 1),
        "foreground_pct": round(fg, 2),
        "border_touch": border_touch,
        "warning": "; ".join(warnings) if warnings else "heuristic proposal; inspect boundary carefully",
    })
    return mask, stats


def make_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    tint = np.zeros_like(out)
    tint[:] = (50, 220, 70)
    blended = cv2.addWeighted(out, 0.68, tint, 0.32, 0)
    out[mask > 0] = blended[mask > 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, (0, 220, 50), 2, cv2.LINE_AA)
    return out


def add_banner(img: np.ndarray, title: str, detail: str, width_limit: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, width_limit / w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    banner_h = 92
    banner = np.full((banner_h, img.shape[1], 3), (28, 39, 32), dtype=np.uint8)
    cv2.putText(banner, title[:150], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (245, 245, 235), 1, cv2.LINE_AA)
    cv2.putText(banner, detail[:150], (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 220, 190), 1, cv2.LINE_AA)
    cv2.putText(banner, "Y=approve/save | N=reject | O=original/overlay | Q=quit", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 235), 1, cv2.LINE_AA)
    return np.vstack([banner, img])


def main() -> int:
    parser = argparse.ArgumentParser(description="Propose and human-review yarn-body masks in batches.")
    parser.add_argument("--images", type=Path, default=Path("data/annotation_candidate"))
    parser.add_argument("--out", type=Path, default=Path("data/yarn_masks/auto_review"))
    parser.add_argument("--display-width", type=int, default=1200)
    parser.add_argument("--manual-masks", type=Path, default=Path("data/yarn_masks/manual/masks"), help="Skip images that already have a matching manually saved mask")
    args = parser.parse_args()

    if not args.images.is_dir():
        print(f"Image folder does not exist: {args.images}", file=sys.stderr)
        return 2
    paths = find_unique_images(args.images)
    if not paths:
        print(f"No supported image files found in {args.images}")
        return 2

    masks_dir = args.out / "masks"
    overlays_dir = args.out / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    window = "Skein — review automatic yarn-body proposals"
    accepted = rejected = no_proposal = 0
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for i, path in enumerate(paths, 1):
        existing_manual = args.manual_masks / f"{path.stem}_mask.png"
        if existing_manual.exists():
            print(f"Already has manual mask; skipping: {path.name}")
            continue
        mask_path = masks_dir / f"{path.stem}_mask.png"
        overlay_path = overlays_dir / f"{path.stem}_overlay.jpg"
        if mask_path.exists():
            print(f"Already reviewed; skipping: {path.name}")
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Unreadable; skipping: {path.name}")
            continue
        mask, stats = proposal_mask(image)
        if mask is None:
            no_proposal += 1
            view = image.copy()
            msg = str(stats.get("warning", "no proposal"))
            cv2.putText(view, "NO RELIABLE AUTO-PROPOSAL", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(view, msg[:100], (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            detail = f"{i}/{len(paths)} | {msg} | N=skip, Q=quit (use manual annotator)"
            shown = add_banner(view, path.name, detail, args.display_width)
            cv2.imshow(window, shown)
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), 27):
                    cv2.destroyAllWindows()
                    print(f"Accepted={accepted}, rejected={rejected}, no_proposal={no_proposal}")
                    return 0
                if key == ord('n'):
                    rejected += 1
                    break
            continue

        overlay = make_overlay(image, mask)
        warnings = str(stats.get("warning", ""))
        detail = (
            f"{i}/{len(paths)} | fg={stats['foreground_pct']}% | span={stats['xspan_pct']}% | "
            f"contrast={stats['contrast']} | {warnings}"
        )
        show_overlay = True
        while True:
            frame = overlay if show_overlay else image
            shown = add_banner(frame.copy(), path.name, detail, args.display_width)
            cv2.imshow(window, shown)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                cv2.destroyAllWindows()
                print(f"Accepted={accepted}, rejected={rejected}, no_proposal={no_proposal}")
                return 0
            if key == ord('o'):
                show_overlay = not show_overlay
            elif key == ord('n'):
                rejected += 1
                print(f"Rejected: {path.name}")
                break
            elif key == ord('y'):
                if not cv2.imwrite(str(mask_path), mask):
                    print(f"ERROR: could not save {mask_path}")
                    continue
                cv2.imwrite(str(overlay_path), overlay)
                accepted += 1
                print(f"Approved: {path.name}\n  mask: {mask_path}\n  overlay: {overlay_path}")
                break

    cv2.destroyAllWindows()
    print("Batch review finished.")
    print(f"Approved={accepted}, rejected={rejected}, no_proposal={no_proposal}")
    print("Review every saved overlay. These are human-approved heuristic labels, not ground-truth measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
