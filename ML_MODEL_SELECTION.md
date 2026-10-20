# Skein ML model plan: choose by held-out yarn results, not by model reputation

## Recommendation

Do **not** ship a generic vision model as if it were a yarn metrology model. There is no representative, expert-annotated dataset in the current project, so there is not yet a supportable claim that any AI model is the most accurate. Keep the transparent Otsu baseline and Sauvola local-threshold option available; train and benchmark yarn-specific models against them.

### Models to benchmark

1. **Classical CV baseline:** Otsu threshold + connected-component checks. This is simple and interpretable.
2. **Classical local threshold:** Sauvola, for cases where background illumination varies; it can also over-segment texture, so it must be benchmarked.
3. **U-Net++ with ResNet34 encoder:** primary supervised deep-learning candidate for a binary yarn-core mask. The nested decoder / skip-connection design is documented in Segmentation Models PyTorch. Fine-tune on yarn-specific masks; do not use ImageNet encoder weights as if they were yarn segmentation weights.
4. **DeepLabV3+ with ResNet34 encoder:** competing supervised model. Evaluate with the same split, augmentation, loss, threshold tuning and independent held-out test set.
5. **SAM 2 or another foundation segmentation model:** potentially useful to **assist annotation** or suggest an initial mask. It is a promptable generic segmentation model, not a ready-made measurement system for the precise outer/core boundary of a yarn bundle. Its masks should not be used for final diameter estimates unless yarn-specific boundary performance wins a measured comparison.

Sources:
- Yarn diameter/fibre image-processing study: Abdelkader (2022), *Materials*, 15(4), 1299, https://doi.org/10.3390/ma15041299. It demonstrates automated image-processing approaches on microscopic yarn/fibre imagery, but it does not establish the phone-camera 0.017 mm requirement in this project.
- Image-based yarn characterization benchmarked against a commercial apparatus: Gonçalves et al. (2015), *Measurement*, https://doi.org/10.1016/j.measurement.2015.02.010.
- Low-cost yarn parameterisation using a USB camera and standard-wire calibration, compared with a commercial yarn tester: https://doi.org/10.1016/j.measurement.2014.09.028.
- U-Net++ architecture documentation: https://segmentation-models-pytorch.readthedocs.io/en/latest/models.html
- DeepLabV3 in Torchvision: https://docs.pytorch.org/vision/main/models/deeplabv3.html
- SAM 2 official description of promptable segmentation: https://ai.meta.com/research/sam2/
- Measurement uncertainty (JCGM/GUM): https://www.iso.org/sites/JCGM/GUM-introduction.htm

## Why a yarn-specific model needs labels

A trained model needs images and corresponding ground-truth masks with a consistent operational definition of yarn width. Agree on whether the target is:

- **Core/body boundary** excluding long flyaway fibres, or
- **Full visible envelope** including hairiness.

Do not mix these two target definitions in one training set. The current app aims to estimate apparent body width; it does not infer the unloaded 3-D physical cross-section of a compressible yarn from a single 2-D projection.

## Dataset layout

Prepare real full-resolution photographs and binary masks, with matching base filenames:

```text
data/yarn_masks/
  images/
    train/   # many independent yarn samples / sessions
    val/     # independent specimens/sessions used to select model/checkpoint
    test/    # untouched specimens used once for final comparison
  masks/
    train/
    val/
    test/
```

Mask convention: white (>=128) is yarn core; black is background. Annotate the full image in a tool such as CVAT or labelme, export binary masks as PNG. Ensure the file dimensions exactly match the source image.

**Split by physical specimen / source / capture session before making crops.** Never random-split neighboring crops or adjacent sections from the same photographed strand into train and test. That would leak almost-identical pixels into both sets.

Start with a pilot of varied yarn types, thicknesses, backgrounds, hairiness, focus states, and several independent recaptures. A tiny homogeneous collection can make validation look good without generalizing.

## Train and compare candidates

Optional packages are separate from the default app requirements:

```powershell
python -m pip install -r requirements-ml.txt
python train_yarn_segmentation.py --architecture unetplusplus --epochs 40 --batch-size 4
python train_yarn_segmentation.py --architecture deeplabv3plus --epochs 40 --batch-size 4
python benchmark_segmentation.py --data-dir data/yarn_masks --output reports/segmentation_benchmark.csv
```

ImageNet backbone weights are used only as initialization; the segmentation head and model must be fine-tuned on yarn masks. If offline, use `--encoder-weights none` but expect training to be more data-hungry. Review the labels before training. No checkpoint ships with this repo; the model selection option appears in the app only after a corresponding local checkpoint exists.

Select the model using held-out mask metrics (Dice, IoU, one-pixel-tolerance boundary F1), failure rate, and downstream diameter error, not only the mask overlap score. For a thin object, a high Dice score can coexist with a boundary error that matters to metrology. The best segmentation model is the one with lower validated **diameter error** on independent specimens, provided it remains repeatable and reliable across the intended capture conditions.

## Measure the thing you actually care about

Segmentation is an intermediate stage. For each independently referenced yarn, collect:

- Reference apparent width in a defined imaging plane and reference uncertainty.
- Skein median width in mm and its bias / absolute error.
- Three or more repositioned captures per specimen to estimate repeatability.
- Pairwise absolute-difference error and the false-SAME rate for true pairs whose independent reference difference is 17 micrometers or more.
- Proportion of cases reported INCONCLUSIVE and why.

Never count thousands of adjacent cross-sections from one image as thousands of independent experiments. Report results by independently captured specimen/pair. Set acceptance criteria before the final held-out evaluation.

## Important limitation

A CNN can improve masks only when useful image signal exists. It cannot restore physical detail lost through optics, motion blur, focus, compression, or inadequate sampling. The 0.017 mm requirement is accepted only when the complete capture + calibration + boundary + measurement pipeline demonstrates sufficiently low bias, uncertainty, and false-SAME rate on independent real samples.
