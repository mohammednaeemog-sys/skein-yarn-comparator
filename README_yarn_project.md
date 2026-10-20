# Skein — Yarn Diameter Comparator (research prototype v4)

Skein is a local Streamlit application for comparing the **2-D apparent core-body width** of two yarn photos captured under a controlled setup. The target is to distinguish samples whose independent reference widths differ by as little as 0.017 mm (17 µm).

**Not yet validated for 17 µm performance.** Synthetic/software tests validate algorithm mechanics only. This build does not ship with a yarn-trained AI model because the project does not yet contain an expert-annotated, representative real-photo dataset. The UI exposes an AI method only when a local trained model checkpoint exists.

## What's new in v4

- More functional, less decorative interface, with short entrance transitions, hover/focus states, an upload/setup/result step tracker, and a staged analysis progress display. Animations respect `prefers-reduced-motion`.
- Added a Sauvola local-threshold segmentation candidate to compare with the transparent Otsu baseline.
- Added candidate fractional-pixel boundary interpolation when grayscale data is available. This is a computational technique, not a claim of proven subpixel optical accuracy.
- Added optional inference plumbing for yarn-specific **U-Net++ / ResNet34** and **DeepLabV3+ / ResNet34** checkpoints. No generic pretrained model is passed off as a yarn-core model.
- Added training and held-out segmentation benchmarking scripts plus a model-selection / data-annotation guide.

## Core installation (Windows / PowerShell)

Open a terminal in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -v
python -m streamlit run yarn_diameter_app.py
```

If `.venv` already exists in this folder, activate it instead of recreating it. If PowerShell blocks activation, use `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `\.venv\Scripts\python.exe -m streamlit run yarn_diameter_app.py` without activating.

## Optional AI model training on an NVIDIA GPU (recommended for your RTX 3050 8 GB)

The classical OpenCV measurement path does not need the GPU. The NVIDIA GPU can accelerate training and inference for the optional yarn-specific segmentation models. An RTX 3050 with 8 GB VRAM is a reasonable starting point for 512-pixel patches with mixed precision; begin with batch size 2 and increase only if memory usage allows. Your current dataset still needs reviewed real yarn masks before training can teach a model what counts as the yarn body.

### 1. Install the CUDA-enabled PyTorch build

Keep using this project's activated `.venv`. Open the official PyTorch installer selector at https://pytorch.org/get-started/locally/ and select **Windows → Pip → Python → a CUDA build** (not CPU). Run the exact command it generates in the activated environment. Use the selector rather than copying a CUDA command from an old tutorial, because compatible builds change. A supported NVIDIA driver is required; in most cases you do not need to separately install the full CUDA Toolkit just to use the official PyTorch wheel.

### 2. Verify GPU access before proceeding

```powershell
python check_gpu.py
```

The diagnostic should print `CUDA available to PyTorch: True`, the GPU name, VRAM, and `CUDA smoke test: PASS`. If it says `False`, don't start training yet; fix the driver/PyTorch installation first.

### 3. Install the model library without replacing PyTorch

```powershell
python -m pip install -r requirements-ml.txt
```

`requirements-ml.txt` deliberately does not list `torch` or `torchvision`—those must come from the official selector so pip does not inadvertently replace the CUDA-enabled installation with an unintended build.

### 4. Train candidates conservatively on 8 GB VRAM

After arranging expert-reviewed real images and masks as explained in `ML_MODEL_SELECTION.md`, start with batch size 2 and 512-pixel patches. Training uses CUDA automatically when `torch.cuda.is_available()` is true and uses mixed precision on CUDA.

```powershell
python train_yarn_segmentation.py --architecture unetplusplus --epochs 40 --patch-size 512 --batch-size 2
python train_yarn_segmentation.py --architecture deeplabv3plus --epochs 40 --patch-size 512 --batch-size 2
python benchmark_segmentation.py --data-dir data/yarn_masks --output reports/segmentation_benchmark.csv
```

If CUDA reports an out-of-memory error, reduce the batch size to 1 first, then reduce patch size to 384 (which is divisible by 32) if needed. Do not compare model quality using different validation/test specimens. GPU acceleration improves training speed, not the optical information in the images or the accuracy guarantee. No trained yarn model is bundled with this project.

Organize reviewed real photos and binary core masks as explained in `ML_MODEL_SELECTION.md`. Then train candidate models:

```powershell
python train_yarn_segmentation.py --architecture unetplusplus --epochs 40 --batch-size 4
python train_yarn_segmentation.py --architecture deeplabv3plus --epochs 40 --batch-size 4
python benchmark_segmentation.py --data-dir data/yarn_masks --output reports/segmentation_benchmark.csv
python evaluate_diameter_pairs.py --csv filled_validation.csv --images-dir path/to/original_images --method unetplusplus --output reports/diameter_validation.csv
```

Default model filenames are:
- `models/yarn_unetplusplus_resnet34_best.pth`
- `models/yarn_deeplabv3plus_resnet34_best.pth`

A checkpoint appears in the app's advanced segmentation menu only after it exists. Use `evaluate_diameter_pairs.py` only after filling the CSV with independently measured widths and the correct scale for each capture. Choose the model with held-out boundary metrics **and** lower physical diameter error, not the model with the most advanced name. Keep the `test` specimens untouched until final model selection.

## Calibration and capture

Skein's printed target is a scale reference, not an accuracy certificate. Print the target at Actual Size, verify its physical dimensions, then capture it at the exact yarn plane using the same lens, zoom, focus, resolution, orientation and distance. Same camera height alone is insufficient. Keep original image files; do not resize screenshots before uploading.

At 100 px/mm, 0.017 mm corresponds to only 1.7 pixels. The provisional five-pixel difference gate requires about 294 px/mm, but that is only a sampling screen—not proof the phone optics can resolve it. If the optical setup fails the feasibility experiment, test macro optics or a low-cost digital microscope before spending time on ML.

## Validation requirements

1. Define whether the measurand is the yarn core/body excluding loose hairs or the full outer envelope.
2. Obtain independent physical reference widths and record reference method/uncertainty.
3. Capture multiple repositioned images per independent specimen.
4. Report bias, absolute diameter error, repeatability, boundary quality, false-SAME rate for independently verified pairs differing by at least 17 µm, and inconclusive rate.
5. Split ML data by physical specimen/source/session, never by neighboring crop.
6. Do not count cross-sections in one image as independent experiments.

See `RESEARCH_AND_VALIDATION_PLAN.md` and `ML_MODEL_SELECTION.md` for sources, recommended experiments, mask convention, training protocol, and candidate model comparison.
