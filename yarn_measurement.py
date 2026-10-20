"""Measurement core for Skein's yarn-width comparison prototype.

Method under development: segment one isolated strand, align its principal axis
horizontally, and measure the longest foreground run across perpendicular image
columns. This follows the general body-width idea in published yarn-image work
more closely than treating every skeleton distance-transform pixel as an
independent diameter sample.

IMPORTANT: this reports 2-D apparent yarn-body width under a controlled imaging
and preparation protocol. It is not yet a validated measurement of true 3-D
yarn diameter. Image-column samples are spatially correlated and MUST NOT be
used as independent repeatability observations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.filters import threshold_sauvola


@dataclass(frozen=True)
class MeasurementConfig:
    pixels_per_mm: float
    min_width_mm: float
    max_width_mm: float
    light_yarn_on_dark: bool = True
    correct_illumination: bool = False
    median_filter: bool = False
    segmentation_method: str = "otsu"  # otsu | sauvola | unetplusplus

    def validate(self) -> None:
        if not np.isfinite(self.pixels_per_mm) or self.pixels_per_mm <= 0:
            raise ValueError("pixels_per_mm must be a finite positive number")
        if not np.isfinite(self.min_width_mm) or self.min_width_mm <= 0:
            raise ValueError("min_width_mm must be a finite positive number")
        if not np.isfinite(self.max_width_mm) or self.max_width_mm <= self.min_width_mm:
            raise ValueError("max_width_mm must be greater than min_width_mm")
        if self.segmentation_method not in {"otsu", "sauvola", "unetplusplus", "deeplabv3plus"}:
            raise ValueError("segmentation_method must be otsu, sauvola, unetplusplus, or deeplabv3plus")


def correct_illumination(gray: np.ndarray) -> np.ndarray:
    """Apply conservative background-field normalization to grayscale data."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a uint8 grayscale image")
    h, w = gray.shape
    k = max(31, (min(h, w) // 3) | 1)
    if k % 2 == 0:
        k += 1
    background = cv2.GaussianBlur(gray, (k, k), 0)
    background_f = np.maximum(background.astype(np.float32), 1.0)
    corrected = gray.astype(np.float32) / background_f * float(np.median(background_f))
    lo, hi = np.percentile(corrected, [1, 99])
    if hi <= lo:
        return gray.copy()
    normalized = np.clip((corrected - lo) * (255.0 / (hi - lo)), 0, 255)
    return normalized.astype(np.uint8)


def segment_yarn(
    rgb: np.ndarray,
    config: MeasurementConfig,
    model: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Segment the primary yarn body using a selectable method.

    ``otsu`` is the transparent baseline. ``sauvola`` is a local-threshold
    alternative for uneven illumination. ``unetplusplus`` and ``deeplabv3plus``
    require a custom-trained model checkpoint, because generic pretrained image
    models are not validated for precise yarn-core boundaries.

    AI probabilities are thresholded and cleaned just like the classical masks;
    the user must still inspect overlays. This does not promise 17-micrometer
    measurement accuracy.
    """
    config.validate()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Image must be an 8-bit RGB image")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.medianBlur(gray, 3) if config.median_filter else gray
    processed = correct_illumination(denoised) if config.correct_illumination else denoised
    warnings: list[str] = []

    if config.segmentation_method in {"unetplusplus", "deeplabv3plus"}:
        if model is None:
            raise ValueError(
                "The selected AI segmentation model has no trained checkpoint. "
                "Create and validate a yarn-core mask dataset, then train the model first."
            )
        from yarn_ml import predict_mask_tiled
        model_mask, ai_diagnostics = predict_mask_tiled(rgb, model)
        mask = (model_mask > 0).astype(np.uint8) * 255
        diagnostics_model = ai_diagnostics
    else:
        threshold_mode = cv2.THRESH_BINARY if config.light_yarn_on_dark else cv2.THRESH_BINARY_INV
        if config.segmentation_method == "otsu":
            _, mask = cv2.threshold(processed, 0, 255, threshold_mode + cv2.THRESH_OTSU)
        elif config.segmentation_method == "sauvola":
            # Odd window sized relative to the frame, but bounded to avoid
            # excessive local threshold variation on small test photographs.
            h, w = processed.shape
            win = max(15, int(round(min(h, w) * 0.03)))
            if win % 2 == 0:
                win += 1
            win = min(win, min(h, w) - (1 - min(h, w) % 2))
            if win < 3:
                raise ValueError("Image is too small for adaptive thresholding")
            local_thresh = threshold_sauvola(processed, window_size=win, k=0.18)
            if config.light_yarn_on_dark:
                mask = (processed > local_thresh).astype(np.uint8) * 255
            else:
                mask = (processed < local_thresh).astype(np.uint8) * 255
        else:
            raise ValueError(f"Unknown segmentation method: {config.segmentation_method}")
        diagnostics_model = {"segmentation_backend": config.segmentation_method}

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise ValueError("No foreground yarn detected. Check contrast and foreground polarity.")
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    largest_label = int(order[0] + 1)
    largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
    other_large = [int(a) for a in areas if a >= max(20, largest_area * 0.05)]
    if len(other_large) > 1:
        warnings.append("Multiple substantial foreground components detected; inspect the mask for loose fibers, debris, or fragmented yarn.")
    clean = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    ys, xs = np.where(clean > 0)
    h, w = clean.shape
    touches_left_right = bool((xs.min() == 0) or (xs.max() == w - 1))
    touches_top_bottom = bool((ys.min() == 0) or (ys.max() == h - 1))
    touches_border = touches_left_right or touches_top_bottom

    # For nearly horizontal/vertical strands, distinguish cropped strand ends
    # from cropped width-defining sides. Stay conservative for diagonal strands.
    points = np.column_stack((xs, ys)).astype(np.float32)
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    axis_angle_deg = abs(float(np.degrees(np.arctan2(float(line[1]), float(line[0])))))
    axis_angle_deg = min(axis_angle_deg, 180.0 - axis_angle_deg)
    if axis_angle_deg <= 20.0:
        # A horizontal strand's width edges are top/bottom, not left/right ends.
        touches_width_boundary_border = touches_top_bottom
    elif axis_angle_deg >= 70.0:
        # A vertical strand's width edges are left/right, not top/bottom ends.
        touches_width_boundary_border = touches_left_right
    else:
        # Oblique contact is ambiguous: treat any frame contact as risky.
        touches_width_boundary_border = touches_border

    if touches_width_boundary_border:
        warnings.append("Detected component touches an image border along its width direction; the yarn boundary may be clipped.")
    foreground_fraction = float(np.count_nonzero(clean) / clean.size)
    if foreground_fraction > 0.5:
        warnings.append("Foreground occupies over half the image; segmentation polarity/background may be wrong.")
    if foreground_fraction < 0.0001:
        warnings.append("Very little foreground detected; segmentation may be incomplete.")
    diagnostics = {
        "component_count_including_background": int(count),
        "selected_component_area_px": largest_area,
        "foreground_fraction": foreground_fraction,
        "touches_image_border": touches_border,
        "touches_width_boundary_border": bool(touches_width_boundary_border),
        "estimated_axis_angle_deg": float(axis_angle_deg),
        "illumination_correction_enabled": config.correct_illumination,
        "median_filter_enabled": config.median_filter,
        **diagnostics_model,
    }
    if config.segmentation_method != "otsu":
        warnings.append(
            f"Segmentation backend '{config.segmentation_method}' is experimental for yarn-core boundaries. "
            "Review the overlay and benchmark against expert-annotated real images."
        )
    return clean, processed, warnings, diagnostics


def _longest_foreground_run(column: np.ndarray) -> int:
    """Return length of longest contiguous foreground run in a 1-D binary scan."""
    values = np.asarray(column, dtype=np.uint8) > 0
    if not values.any():
        return 0
    edges = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if starts.size == 0:
        return 0
    return int(np.max(ends - starts))


def _longest_foreground_span(column: np.ndarray) -> tuple[int, int] | None:
    """Return (start, stop) pixel-center indices for the longest foreground run."""
    values = np.asarray(column, dtype=np.uint8) > 0
    if not values.any():
        return None
    edges = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if starts.size == 0:
        return None
    idx = int(np.argmax(ends - starts))
    return int(starts[idx]), int(ends[idx])


def _interpolate_crossing(x0: float, v0: float, x1: float, v1: float, threshold: float) -> float:
    """Linearly interpolate a gray-level threshold crossing between pixel centers."""
    delta = float(v1 - v0)
    if abs(delta) < 1e-6:
        return (x0 + x1) / 2.0
    fraction = float(np.clip((threshold - v0) / delta, 0.0, 1.0))
    return x0 + fraction * (x1 - x0)


def _rotate_to_horizontal(binary: np.ndarray, gray: np.ndarray | None = None) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Rotate a binary mask so the robust least-squares principal axis is horizontal.

    The canvas is expanded to avoid clipping the rotated strand. Nearest-neighbor
    interpolation avoids inventing gray intermediate mask pixels.
    """
    yy, xx = np.nonzero(binary)
    if xx.size < 50:
        raise ValueError("Too little foreground detected for a stable axis estimate")
    points = np.column_stack((xx, yy)).astype(np.float32)
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    vx, vy = float(line[0]), float(line[1])
    angle = float(np.degrees(np.arctan2(vy, vx)))
    h, w = binary.shape
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos, abs_sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = max(1, int(np.ceil(h * abs_sin + w * abs_cos)))
    new_h = max(1, int(np.ceil(h * abs_cos + w * abs_sin)))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    rotated = cv2.warpAffine(binary.astype(np.uint8) * 255, matrix, (new_w, new_h),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rotated_gray = None
    if gray is not None:
        if gray.ndim != 2 or gray.shape != binary.shape or gray.dtype != np.uint8:
            raise ValueError("gray must be a uint8 grayscale image matching the mask")
        border_value = int(np.median(gray))
        rotated_gray = cv2.warpAffine(gray, matrix, (new_w, new_h),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=border_value)
    return rotated > 0, angle, rotated_gray


def estimate_widths(mask: np.ndarray, config: MeasurementConfig, gray: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Estimate apparent core widths using perpendicular cross-sections.

    The binary mask is aligned to its long axis. At each sampled longitudinal
    station, we select the longest contiguous foreground run across the strand,
    which helps exclude disconnected hairs and isolated debris. A median and
    quantile-trimmed distribution are then reported. This is a candidate method,
    not a validated microscopy/metrology algorithm.

    If grayscale is provided, midpoint-intensity interpolation estimates fractional-pixel
    edge positions. This is a candidate subpixel technique, not proof of optical accuracy.

    Returns width samples in millimeters, the original skeleton for visualization,
    and warnings. Width samples along one strand are correlated observations.
    """
    config.validate()
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D binary image")
    binary = mask > 0
    if int(binary.sum()) < 50:
        raise ValueError("Too little foreground detected for width estimation")
    skeleton = skeletonize(binary)
    rotated, angle_deg, rotated_gray = _rotate_to_horizontal(binary, gray)
    occupied_x = np.flatnonzero(rotated.any(axis=0))
    if occupied_x.size < 10:
        raise ValueError("Detected yarn is too short or incomplete for cross-sectional measurement")
    left, right = int(occupied_x[0]), int(occupied_x[-1])
    span = right - left + 1
    start = left + int(round(span * 0.10))
    stop = right - int(round(span * 0.10))
    if stop - start < 5:
        raise ValueError("Detected yarn is too short after excluding the ends")
    # Sample at ~5 micrometer steps when possible. Denser samples do not create
    # independent physical observations; they only describe width variation.
    step = max(1, int(round(config.pixels_per_mm * 0.005)))
    min_px = config.min_width_mm * config.pixels_per_mm
    max_px = config.max_width_mm * config.pixels_per_mm
    width_px: list[float] = []
    candidate_count = 0
    subpixel_used = 0
    for x in range(start, stop + 1, step):
        span_edges = _longest_foreground_span(rotated[:, x])
        if span_edges is None:
            continue
        run_start, run_stop = span_edges  # stop is exclusive
        run_len = run_stop - run_start
        candidate_count += 1
        width_estimate_px = float(run_len)
        if rotated_gray is not None and run_start >= 1 and run_stop < rotated_gray.shape[0] and run_len >= 3:
            profile = rotated_gray[:, x].astype(np.float32)
            inner_start = run_start + max(0, int(run_len * 0.25))
            inner_stop = max(inner_start + 1, run_stop - int(run_len * 0.25))
            inner_level = float(np.median(profile[inner_start:inner_stop]))
            left_bg = float(np.median(profile[max(0, run_start - 4):run_start]))
            right_bg = float(np.median(profile[run_stop:min(profile.size, run_stop + 4)]))
            if np.isfinite(inner_level + left_bg + right_bg):
                left_threshold = (inner_level + left_bg) / 2.0
                right_threshold = (inner_level + right_bg) / 2.0
                left_edge = _interpolate_crossing(run_start - 1, profile[run_start - 1], run_start, profile[run_start], left_threshold)
                right_edge = _interpolate_crossing(run_stop - 1, profile[run_stop - 1], run_stop, profile[run_stop], right_threshold)
                candidate_width = right_edge - left_edge
                expected = (inner_level > left_bg and inner_level > right_bg) if config.light_yarn_on_dark else (inner_level < left_bg and inner_level < right_bg)
                if expected and np.isfinite(candidate_width) and 0.25 <= candidate_width <= run_len + 2.0:
                    width_estimate_px = float(candidate_width)
                    subpixel_used += 1
        if min_px <= width_estimate_px <= max_px:
            width_px.append(width_estimate_px)
    if len(width_px) < 5:
        raise ValueError("Too few plausible cross-sections; check calibration, width limits, focus, and segmentation")
    widths_mm = np.asarray(width_px, dtype=float) / config.pixels_per_mm
    q_low, q_high = np.percentile(widths_mm, [2, 98])
    trimmed = widths_mm[(widths_mm >= q_low) & (widths_mm <= q_high)]
    if trimmed.size < 5:
        trimmed = widths_mm
    warnings: list[str] = []
    coverage = len(width_px) / max(candidate_count, 1)
    if coverage < 0.8:
        warnings.append("Many cross-sections were rejected by the configured width limits; inspect calibration and segmentation.")
    if len(trimmed) < 20:
        warnings.append("Fewer than 20 cross-sections were retained; the estimate may be unstable.")
    if float(np.std(trimmed)) / max(float(np.mean(trimmed)), 1e-12) > 0.25:
        warnings.append("Local cross-sectional widths vary substantially; inspect yarn straightness, hairiness, focus, and segmentation.")
    if abs(angle_deg) > 20:
        warnings.append("The yarn is strongly diagonal in the image; keep the strand straight and closer to horizontal for repeatable capture.")
    if rotated_gray is None:
        warnings.append("No grayscale image was supplied, so widths use integer-pixel mask boundaries; fractional-pixel edge interpolation was not applied.")
    elif subpixel_used < 0.8 * max(candidate_count, 1):
        warnings.append("Subpixel edge interpolation was unavailable for many sections due to weak/irregular contrast; widths rely more heavily on binary mask boundaries.")
    return trimmed.astype(float), skeleton, warnings


def summarize_widths(widths_mm: np.ndarray) -> dict[str, float | int]:
    if widths_mm.size == 0:
        raise ValueError("Cannot summarize an empty set of widths")
    return {
        "mean_mm": float(np.mean(widths_mm)),
        "median_mm": float(np.median(widths_mm)),
        "std_mm": float(np.std(widths_mm, ddof=1)) if widths_mm.size > 1 else 0.0,
        "p05_mm": float(np.percentile(widths_mm, 5)),
        "p95_mm": float(np.percentile(widths_mm, 95)),
        "n_local_samples": int(widths_mm.size),
    }


def compare_summaries(a: dict[str, Any], b: dict[str, Any], tolerance_mm: float) -> dict[str, Any]:
    """Preliminary threshold screen; it is not a formal equivalence test."""
    if not np.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError("tolerance_mm must be finite and non-negative")
    difference = abs(float(a["median_mm"]) - float(b["median_mm"]))
    return {
        "absolute_median_difference_mm": difference,
        "tolerance_mm": float(tolerance_mm),
        "screening_result": "SAME_WITHIN_TOLERANCE" if difference <= tolerance_mm else "DIFFERENT",
        "warning": "Preliminary screen only; repeatability, reference bias, and measurement uncertainty are not included in this decision.",
    }
