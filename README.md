# Skein — Yarn Diameter Comparator

Skein is a Streamlit-based research prototype for comparing apparent yarn widths from two images.

## Features

- Midnight Neon interface and animated visuals
- Image-based yarn segmentation and width estimation
- Camera calibration workflow
- Detected-boundary overlays and quality warnings
- Measurement comparison and report downloads
- Otsu baseline and optional experimental segmentation methods

## Run locally

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the application:

```powershell
python -m streamlit run yarn_diameter_app_neon.py
```

## Tests

```powershell
python -m py_compile yarn_diameter_app_neon.py yarn_measurement.py yarn_calibration.py yarn_ml.py
python -m pytest -v
```

## Measurement limitations

Skein is a research prototype, not a validated metrology instrument. Its ability to reliably detect a 0.017 mm (17 micrometre) diameter difference has not been established. Physical calibration and independent reference measurements on real yarn samples are required.

Uncalibrated pixel measurements must not be interpreted as physical dimensions. Results should remain inconclusive when calibration or image-quality checks fail.

## Local files

Datasets, local photographs, calibration profiles, generated reports, virtual environments, and model checkpoints are not intended to be published in this repository.
