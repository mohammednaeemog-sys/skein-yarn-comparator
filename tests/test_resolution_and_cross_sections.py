"""Synthetic resolution/estimator tests; not evidence of real camera accuracy."""
import cv2
import numpy as np
import pytest

from yarn_measurement import MeasurementConfig, estimate_widths, summarize_widths
from yarn_calibration import MARKER_ID, detect_calibration


def _straight_mask(thickness_px: int, angle_deg: float = 0.0) -> np.ndarray:
    canvas = np.zeros((220, 800), dtype=np.uint8)
    y1 = 110 - thickness_px // 2
    y2 = y1 + thickness_px - 1
    cv2.rectangle(canvas, (70, y1), (730, y2), 255, -1)
    if angle_deg:
        matrix = cv2.getRotationMatrix2D((400, 110), angle_deg, 1.0)
        canvas = cv2.warpAffine(canvas, matrix, (800, 220), flags=cv2.INTER_NEAREST)
    return canvas


def test_transverse_width_estimator_recovers_horizontal_width():
    scale = 300.0
    config = MeasurementConfig(pixels_per_mm=scale, min_width_mm=0.05, max_width_mm=1.0)
    mask = _straight_mask(60)
    widths, skeleton, warnings = estimate_widths(mask, config)
    assert widths.size >= 20
    assert np.median(widths) == pytest.approx(60 / scale, abs=0.0035)
    assert skeleton.any()


def test_017mm_synthetic_width_difference_is_detected_in_raster_domain():
    # 5 pixels / 300 px per mm = 0.01667 mm. This checks algorithm arithmetic
    # on ideal masks only; it says nothing about phone optics or segmentation.
    scale = 300.0
    config = MeasurementConfig(pixels_per_mm=scale, min_width_mm=0.05, max_width_mm=1.0)
    a, _, _ = estimate_widths(_straight_mask(55), config)
    b, _, _ = estimate_widths(_straight_mask(60), config)
    measured_difference = abs(float(np.median(a)) - float(np.median(b)))
    assert measured_difference == pytest.approx(5 / scale, abs=0.0035)
    assert abs(measured_difference - 0.017) < 0.002


def test_transverse_width_estimator_tolerates_mild_diagonal_strand():
    scale = 300.0
    config = MeasurementConfig(pixels_per_mm=scale, min_width_mm=0.05, max_width_mm=1.0)
    horizontal, _, _ = estimate_widths(_straight_mask(60, 0), config)
    diagonal, _, _ = estimate_widths(_straight_mask(60, 12), config)
    assert abs(float(np.median(horizontal)) - float(np.median(diagonal))) <= 0.008


def test_calibration_warning_enforces_provisional_five_pixel_target_gate():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        pytest.skip("OpenCV ArUco module unavailable")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    marker = aruco.generateImageMarker(dictionary, MARKER_ID, 800, borderBits=1)
    image = np.full((1200, 1600, 3), 255, dtype=np.uint8)
    image[200:1000, 400:1200] = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
    profile = detect_calibration(image)
    assert profile["target_difference_pixels"] < 5.0
    assert any("5-pixel screening target" in warning for warning in profile["warnings"])
