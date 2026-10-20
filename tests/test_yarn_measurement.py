import numpy as np
import pytest
import cv2
from yarn_measurement import MeasurementConfig, estimate_widths, summarize_widths, compare_summaries


def test_config_rejects_invalid_calibration():
    cfg = MeasurementConfig(pixels_per_mm=0, min_width_mm=0.02, max_width_mm=5)
    with pytest.raises(ValueError, match="pixels_per_mm"):
        cfg.validate()


def test_estimator_recovers_synthetic_straight_strand_approximately():
    # Synthetic capsule with a known raster thickness. This validates code behavior,
    # not real-world optical accuracy.
    mask = np.zeros((100, 240), dtype=np.uint8)
    cv2.rectangle(mask, (50, 40), (190, 49), 255, thickness=-1)
    cv2.circle(mask, (50, 44), 4, 255, thickness=-1)
    cv2.circle(mask, (190, 44), 4, 255, thickness=-1)
    cfg = MeasurementConfig(pixels_per_mm=100, min_width_mm=0.02, max_width_mm=0.2)
    widths, skeleton, warnings = estimate_widths(mask, cfg)
    # Width should be close to the 10-pixel body thickness, allowing rasterization effects.
    assert abs(float(np.median(widths)) - 0.10) <= 0.025
    assert skeleton.any()
    assert widths.size >= 5


def test_summary_and_comparison():
    a = summarize_widths(np.array([0.99, 1.00, 1.01]))
    b = summarize_widths(np.array([1.00, 1.01, 1.02]))
    result = compare_summaries(a, b, tolerance_mm=0.05)
    assert result["screening_result"] == "SAME_WITHIN_TOLERANCE"
    assert result["absolute_median_difference_mm"] == pytest.approx(0.01)


def test_negative_tolerance_rejected():
    with pytest.raises(ValueError, match="tolerance_mm"):
        compare_summaries({"median_mm": 1.0}, {"median_mm": 1.0}, -0.1)


def test_sauvola_segmentation_returns_a_mask_and_method_diagnostic():
    from yarn_measurement import segment_yarn
    gray = np.full((160, 640), 24, dtype=np.uint8)
    cv2.rectangle(gray, (70, 72), (570, 87), 220, -1)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cfg = MeasurementConfig(
        pixels_per_mm=300, min_width_mm=0.01, max_width_mm=1.0,
        light_yarn_on_dark=True, segmentation_method="sauvola"
    )
    mask, processed, warnings, diagnostics = segment_yarn(rgb, cfg)
    assert mask.shape == gray.shape
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) > 0
    assert diagnostics["segmentation_backend"] == "sauvola"


def test_ai_segmentation_never_silently_falls_back_without_a_trained_model():
    from yarn_measurement import segment_yarn
    gray = np.full((160, 640), 30, dtype=np.uint8)
    cv2.rectangle(gray, (70, 72), (570, 87), 220, -1)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cfg = MeasurementConfig(
        pixels_per_mm=300, min_width_mm=0.01, max_width_mm=1.0,
        segmentation_method="unetplusplus"
    )
    with pytest.raises(ValueError, match="no trained checkpoint"):
        segment_yarn(rgb, cfg, model=None)


def test_subpixel_estimator_accepts_matching_grayscale_image():
    from yarn_measurement import estimate_widths
    # Anti-aliased synthetic strand. This checks interpolation mechanics only,
    # not the optical performance of a real camera.
    gray = np.full((140, 480), 20, dtype=np.uint8)
    cv2.rectangle(gray, (55, 57), (425, 76), 220, -1)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.8)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cfg = MeasurementConfig(pixels_per_mm=200, min_width_mm=0.02, max_width_mm=0.5)
    widths, skeleton, warnings = estimate_widths(mask, cfg, gray=gray)
    assert widths.size >= 20
    assert np.median(widths) == pytest.approx(0.10, abs=0.02)
    # At least some widths should carry fractional-pixel information from the
    # gray-level crossings rather than only integer mask-run lengths.
    assert np.any(np.abs(widths * cfg.pixels_per_mm - np.round(widths * cfg.pixels_per_mm)) > 1e-3)
    assert skeleton.any()
    assert not any("No grayscale image was supplied" in w for w in warnings)
