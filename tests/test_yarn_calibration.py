import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from yarn_calibration import (
    MARKER_ID,
    MARKER_SIDE_MM,
    calibration_target_svg,
    detect_calibration,
    load_profile,
    save_profile,
)


def _synthetic_calibration_image(marker_pixels=800, canvas=(1600, 1200)):
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        pytest.skip("OpenCV ArUco module is unavailable")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(dictionary, MARKER_ID, marker_pixels, borderBits=1)
    else:
        marker = np.zeros((marker_pixels, marker_pixels), dtype=np.uint8)
        aruco.drawMarker(dictionary, MARKER_ID, marker_pixels, marker, borderBits=1)
    width, height = canvas
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x = (width - marker_pixels) // 2
    y = (height - marker_pixels) // 2
    image[y:y + marker_pixels, x:x + marker_pixels] = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
    return image


def test_auto_calibration_detects_marker_and_estimates_scale():
    image = _synthetic_calibration_image(marker_pixels=800)
    profile = detect_calibration(image, marker_side_mm=10.0)
    assert profile["pixels_per_mm"] == pytest.approx(80.0, abs=0.25)
    assert profile["image_width_px"] == 1600
    assert profile["image_height_px"] == 1200
    assert profile["marker_id"] == MARKER_ID
    assert profile["target_difference_pixels"] == pytest.approx(1.36, abs=0.1)


def test_auto_calibration_rejects_photo_without_marker():
    blank = np.full((640, 800, 3), 220, dtype=np.uint8)
    with pytest.raises(ValueError, match="No calibration marker"):
        detect_calibration(blank)


def test_svg_has_physical_size_and_print_instructions():
    svg = calibration_target_svg().decode("utf-8")
    assert 'width="16mm"' in svg
    assert 'height="16mm"' in svg
    assert 'width="10mm"' in svg
    assert "Actual Size" in svg


def test_profile_is_saved_and_loaded(tmp_path: Path):
    source = _synthetic_calibration_image()
    profile = detect_calibration(source)
    path = tmp_path / "calibration_profile.json"
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded is not None
    assert loaded["pixels_per_mm"] == pytest.approx(profile["pixels_per_mm"])
    assert loaded["image_width_px"] == 1600
    raw = json.loads(path.read_text())
    assert raw["profile_version"] == 1
