# Current Skein v4 GPU-ready status

- [x] Local Streamlit application with two-photo upload and calibration workflow
- [x] Functional UI refresh: restrained transition/hover animations, responsive workflow tracker, staged progress status, reduced-motion support
- [x] Persistent local camera calibration profile and image-dimension checks
- [x] Otsu and Sauvola segmentation candidate methods
- [x] Fractional-pixel intensity-interpolated edge estimate candidate when grayscale is provided
- [x] Optional U-Net++/ResNet34 and DeepLabV3+/ResNet34 inference plumbing; model selection appears only for local trained checkpoints
- [x] GPU-capable training script and held-out segmentation benchmark utility
- [x] Added RTX 3050-friendly GPU diagnostic with an actual CUDA smoke test
- [x] Corrected optional ML requirements so they do not reinstall/override the selected PyTorch CUDA build
- [x] Documented 8 GB VRAM starting settings (512 px patches, batch size 2, mixed precision)
- [x] Documented expert mask convention and specimen-level data splitting strategy
- [x] Synthetic/software tests passed: 15 tests
- [x] Python syntax compilation passed for app, core, calibration, ML inference, training, and benchmark scripts
- [ ] Streamlit v4 app launched and visually checked on the user's Windows PC
- [ ] Representative real-yarn core masks annotated and dataset audit completed
- [ ] Candidate models trained and compared on held-out specimens
- [ ] Actual phone/lens sampling and optical blur feasibility established
- [ ] Independent reference widths collected with uncertainty recorded
- [ ] Bias, repeatability, pairwise-difference errors and false-SAME rate reported
- [ ] Pre-specified acceptance criteria passed on untouched specimens

**Important:** there is no trained AI checkpoint bundled. A model architecture name is not evidence of model performance. Do not claim reliable detection of 0.017 mm differences until the complete physical imaging and analysis chain passes independent validation.
