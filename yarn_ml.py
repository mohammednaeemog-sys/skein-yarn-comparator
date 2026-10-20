"""Optional supervised yarn-segmentation model support.

This module intentionally imports PyTorch and segmentation-models-pytorch lazily.
The default application can still run without the optional ML dependencies or a
trained checkpoint. A custom model must be trained and evaluated on yarn photos
with reviewed masks; generic foundation-model weights are not treated as a
precision yarn-boundary solution by default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILENAMES = {
    "unetplusplus": "yarn_unetplusplus_resnet34_best.pth",
    "deeplabv3plus": "yarn_deeplabv3plus_resnet34_best.pth",
}


def checkpoint_path(architecture: str) -> Path:
    if architecture not in MODEL_FILENAMES:
        raise ValueError(f"Unsupported model architecture: {architecture}")
    return DEFAULT_MODEL_DIR / MODEL_FILENAMES[architecture]


def available_models() -> list[str]:
    """List models that actually have local trained checkpoints."""
    return [name for name in MODEL_FILENAMES if checkpoint_path(name).is_file()]


def _build_model(architecture: str):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "AI model support is optional. Install requirements-ml.txt in this project environment first."
        ) from exc

    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None
        )
    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None
        )
    raise ValueError(f"Unsupported model architecture: {architecture}")


def load_trained_model(architecture: str, path: str | Path | None = None):
    """Load one locally trained checkpoint. No automatic fallback is performed."""
    import torch

    source = Path(path) if path else checkpoint_path(architecture)
    if not source.is_file():
        raise FileNotFoundError(
            f"No trained {architecture} checkpoint exists at {source}. Annotate real yarn photos and train a model first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(source, map_location=device, weights_only=False)
    model_architecture = payload.get("architecture", architecture) if isinstance(payload, dict) else architecture
    if model_architecture != architecture:
        raise ValueError(f"Checkpoint architecture is {model_architecture}, but {architecture} was selected.")
    model = _build_model(architecture)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    model._skein_architecture = architecture
    model._skein_device = str(device)
    model._skein_checkpoint = str(source)
    model._skein_validation_dice = payload.get("validation_dice") if isinstance(payload, dict) else None
    return model


def predict_mask_tiled(
    rgb: np.ndarray,
    model: Any,
    tile_size: int = 512,
    overlap: int = 96,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict yarn-core probability mask on a native-resolution RGB image.

    Overlapping inference tiles avoid shrinking a high-resolution image to a
    small network input. Pixel probabilities are averaged where tiles overlap.
    Return is a 0/255 uint8 mask plus diagnostics. The chosen 0.5 threshold is a
    starting point and should be tuned on a validation set for the trained model.
    """
    import torch

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Inference expects an RGB uint8 image")
    if tile_size < 64 or overlap < 0 or overlap >= tile_size // 2:
        raise ValueError("Require tile_size >= 64 and 0 <= overlap < tile_size/2")

    device = next(model.parameters()).device
    height, width = rgb.shape[:2]
    stride = tile_size - overlap
    y_positions = list(range(0, max(1, height - tile_size + 1), stride))
    x_positions = list(range(0, max(1, width - tile_size + 1), stride))
    if not y_positions or y_positions[-1] != max(0, height - tile_size):
        y_positions.append(max(0, height - tile_size))
    if not x_positions or x_positions[-1] != max(0, width - tile_size):
        x_positions.append(max(0, width - tile_size))

    image = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    model_threshold = 0.5

    model.eval()
    with torch.inference_mode():
        for y in sorted(set(y_positions)):
            for x in sorted(set(x_positions)):
                patch = rgb[y:min(y + tile_size, height), x:min(x + tile_size, width)]
                ph, pw = patch.shape[:2]
                if ph < tile_size or pw < tile_size:
                    # Reflect-pad instead of shrinking the object; crop predictions back to the source area.
                    pad_h, pad_w = tile_size - ph, tile_size - pw
                    pad_mode = "reflect" if ph > 1 and pw > 1 else "edge"
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)
                arr = patch.astype(np.float32) / 255.0
                arr = (arr - mean) / std
                tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                output = model(tensor)
                if isinstance(output, dict):
                    output = output.get("out", next(iter(output.values())))
                probs = torch.sigmoid(output)[0, 0].detach().cpu().numpy()[:ph, :pw]
                # Smooth cosine window reduces seams while preserving native pixels.
                wy = np.hanning(max(ph, 3)).astype(np.float32)[:ph]
                wx = np.hanning(max(pw, 3)).astype(np.float32)[:pw]
                weights = np.maximum(np.outer(wy, wx), 0.05)
                probability_sum[y:y + ph, x:x + pw] += probs * weights
                weight_sum[y:y + ph, x:x + pw] += weights

    probabilities = probability_sum / np.maximum(weight_sum, 1e-6)
    binary = (probabilities >= model_threshold).astype(np.uint8) * 255
    diagnostics = {
        "segmentation_backend": getattr(model, "_skein_architecture", "trained neural network"),
        "model_device": getattr(model, "_skein_device", str(device)),
        "model_validation_dice": getattr(model, "_skein_validation_dice", None),
        "inference_tile_size_px": int(tile_size),
        "inference_overlap_px": int(overlap),
        "probability_threshold": model_threshold,
        "foreground_probability_mean": float(probabilities[binary > 0].mean()) if np.any(binary > 0) else 0.0,
    }
    return binary, diagnostics
