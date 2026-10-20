"""Automatic one-time image-scale calibration for Skein.

Uses a printable ArUco marker with a declared physical side length. The user
must print the SVG at actual size and capture it using the same locked camera
setup and sample plane as the yarn. This supplies image scale; it does not by
itself validate the system's ability to resolve 0.017 mm differences.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MARKER_SIDE_MM = 10.0
MARKER_ID = 0
PROFILE_VERSION = 1


def _require_aruco():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError(
            "Automatic calibration needs OpenCV's ArUco module. Install "
            "opencv-contrib-python-headless (not opencv-python-headless), "
            "then restart the app."
        )
    return aruco


def _make_marker(marker_id: int = MARKER_ID, pixels: int = 1200) -> np.ndarray:
    aruco = _require_aruco()
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    if hasattr(aruco, "generateImageMarker"):
        return aruco.generateImageMarker(dictionary, marker_id, pixels, borderBits=1)
    marker = np.zeros((pixels, pixels), dtype=np.uint8)
    aruco.drawMarker(dictionary, marker_id, pixels, marker, borderBits=1)
    return marker


def calibration_target_svg(marker_side_mm: float = MARKER_SIDE_MM) -> bytes:
    """Create a downloadable, physically-sized SVG calibration target.

    The marker's outer square is exactly ``marker_side_mm`` in the SVG's
    physical units when printed at 100% / Actual Size. The target has a white
    quiet border for easier detection.
    """
    if not np.isfinite(marker_side_mm) or marker_side_mm <= 0:
        raise ValueError("marker_side_mm must be a positive finite number")
    marker = _make_marker()
    ok, encoded = cv2.imencode(".png", marker)
    if not ok:
        raise RuntimeError("Could not encode the calibration marker")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    canvas = marker_side_mm + 6.0
    offset = 3.0
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{canvas:g}mm" height="{canvas:g}mm" viewBox="0 0 {canvas:g} {canvas:g}">
  <title>Skein calibration target — marker outer side {marker_side_mm:g} mm</title>
  <desc>Print at 100 percent / Actual Size. Do not use Fit to Page. Place flat in the same plane as the yarn.</desc>
  <rect x="0" y="0" width="{canvas:g}" height="{canvas:g}" fill="#ffffff"/>
  <image x="{offset:g}" y="{offset:g}" width="{marker_side_mm:g}mm" height="{marker_side_mm:g}mm" preserveAspectRatio="none" href="data:image/png;base64,{b64}"/>
</svg>'''
    return svg.encode("utf-8")


def detect_calibration(rgb: np.ndarray, marker_side_mm: float = MARKER_SIDE_MM,
                       marker_id: int = MARKER_ID) -> dict[str, Any]:
    """Detect the known-size marker and estimate pixels/mm from its four edges.

    The returned image width/height are part of the profile so the app can
    flag later photos with a different pixel resolution. Large perspective,
    blur, low resolution, or a wrong physical print size can invalidate scale.
    """
    if not np.isfinite(marker_side_mm) or marker_side_mm <= 0:
        raise ValueError("marker_side_mm must be a positive finite number")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Calibration photo must be an 8-bit RGB image")
    aruco = _require_aruco()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None or len(corners) == 0:
        raise ValueError(
            "No calibration marker was detected. Print the Skein target at Actual Size, "
            "keep it flat and fully visible, and retake a sharp photo with even lighting."
        )
    ids_flat = ids.flatten()
    matches = np.flatnonzero(ids_flat == int(marker_id))
    if matches.size == 0:
        found = ", ".join(str(int(v)) for v in ids_flat)
        raise ValueError(
            f"A marker was found, but not Skein's calibration marker (ID {marker_id}). "
            f"Detected marker ID(s): {found}. Use the target downloaded from this app."
        )
    points = np.asarray(corners[int(matches[0])], dtype=np.float64).reshape(4, 2)
    lengths = np.array([np.linalg.norm(points[(i + 1) % 4] - points[i]) for i in range(4)], dtype=float)
    mean_edge = float(np.mean(lengths))
    if not np.isfinite(mean_edge) or mean_edge < 150:
        raise ValueError(
            f"The detected marker is only {mean_edge:.1f} pixels across. Retake a closer, sharper photo; "
            "the marker should be large and clearly visible in the frame."
        )
    edge_cv = float(np.std(lengths) / mean_edge)
    px_per_mm = mean_edge / float(marker_side_mm)
    h, w = gray.shape
    warnings: list[str] = []
    if edge_cv > 0.06:
        warnings.append(
            "The marker edges differ in length noticeably. Make the target parallel to the camera sensor and retake the calibration photo."
        )
    target_pixels = px_per_mm * 0.017
    if target_pixels < 5.0:
        warnings.append(
            f"At this scale, 0.017 mm spans about {target_pixels:.2f} pixels, below Skein’s provisional 5-pixel screening target. The app will not issue a decision from this profile. Use sharper macro optics or a closer fixed setup, then recalibrate. Five pixels is an engineering screening gate, not proof of accuracy."
        )
    return {
        "profile_version": PROFILE_VERSION,
        "pixels_per_mm": float(px_per_mm),
        "reference_side_mm": float(marker_side_mm),
        "marker_id": int(marker_id),
        "marker_edge_pixels": float(mean_edge),
        "marker_edge_cv": edge_cv,
        "image_width_px": int(w),
        "image_height_px": int(h),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_difference_pixels": float(target_pixels),
        "warnings": warnings,
    }


def save_profile(profile: dict[str, Any], path: str | Path) -> None:
    """Persist a local profile atomically; caller is responsible for validation."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    temporary.replace(destination)


def load_profile(path: str | Path) -> dict[str, Any] | None:
    """Read and sanity-check a previously saved calibration profile."""
    source = Path(path)
    if not source.is_file():
        return None
    try:
        profile = json.loads(source.read_text(encoding="utf-8"))
        required = ("pixels_per_mm", "image_width_px", "image_height_px", "reference_side_mm", "marker_id")
        if not all(key in profile for key in required):
            return None
        scale = float(profile["pixels_per_mm"])
        width = int(profile["image_width_px"])
        height = int(profile["image_height_px"])
        if not np.isfinite(scale) or scale <= 0 or width <= 0 or height <= 0:
            return None
        profile["pixels_per_mm"] = scale
        profile["image_width_px"] = width
        profile["image_height_px"] = height
        profile.setdefault("warnings", [])
        return profile
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def clear_profile(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        # Caller can still clear its in-memory profile; persistent file can be removed manually.
        pass
