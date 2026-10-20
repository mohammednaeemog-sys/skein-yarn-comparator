"""Diagnose whether this Skein environment can actually use an NVIDIA GPU."""
from __future__ import annotations
import platform
import sys


def main() -> int:
    print("Skein GPU diagnostic")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    try:
        import torch
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        print("Install a CUDA-enabled PyTorch build using https://pytorch.org/get-started/locally/")
        return 2

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'CPU-only'}")
    available = torch.cuda.is_available()
    print(f"CUDA available to PyTorch: {available}")
    if not available:
        print("GPU status: NOT AVAILABLE TO THIS PYTHON ENVIRONMENT")
        print("Check the NVIDIA driver, then install the Windows/Pip/Python/CUDA build from the official PyTorch selector.")
        return 1

    try:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        print(f"GPU: {props.name}")
        print(f"VRAM: {props.total_memory / (1024**3):.2f} GiB")
        print(f"Compute capability: {props.major}.{props.minor}")
        # Run an actual CUDA operation so this is more than a name-only check.
        a = torch.randn((1024, 1024), device="cuda")
        b = a @ a
        torch.cuda.synchronize()
        print(f"CUDA smoke test: PASS (tensor result shape {tuple(b.shape)})")
    except Exception as exc:
        print(f"CUDA smoke test: FAILED ({type(exc).__name__}: {exc})")
        return 3
    print("GPU is ready for optional model training/inference.")
    print("Note: GPU acceleration improves model computation speed; it does not itself guarantee 17 µm measurement accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
