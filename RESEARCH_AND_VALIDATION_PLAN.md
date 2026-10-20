# Skein: research audit and validation plan

**Status:** prototype under development. Not validated for 17 µm diameter-difference detection.

## 1. Exact requirement and measurand

The target is to compare two separate phone photographs of yarn captured with the same fixed camera setup and determine whether their yarn-body widths are equivalent. The smallest diameter difference of interest is **0.017 mm = 17 µm**. This is a difference between samples, not a claim that the yarn itself is 17 µm in diameter and not 0.0017 mm.

Until a physical yarn-diameter reference method is selected, the software measurand is defined more narrowly as **2-D apparent core-body width of a single straight strand in a controlled image, excluding loose hairs where segmentation permits**. This must not be reported as true 3-D cross-sectional diameter without further evidence. Yarn is made from fibres and its visible envelope changes with hairiness, twist, tension, compression, angle, and focus.

## 2. Literature audit

### Abdelkader (2022), *Materials* 15(4), 1299

Open-access article: https://doi.org/10.3390/ma15041299

- The paper measures yarn-body width from microscopic images, binarizes the body, processes away hairiness, and samples image rows/columns. It compares its image-processing estimate against manual measurements in ImageJ.
- The reported yarn result is approximately **0.47 ± 0.03 mm** by the proposed method versus **0.42 ± 0.03 mm** by manual ImageJ measurement: a difference of about **0.05 mm** between methods. That is larger than this project's 0.017 mm target difference.
- The paper reports SEM images captured using a Zeiss Ultra Plus FE-SEM at 5 kV and 50× magnification—not ordinary phone-camera photos. Its result therefore supports image processing as a plausible approach, but it does **not** validate a phone camera or 17 µm pairwise-difference classification.
- The paper's explicit removal of yarn hairiness is relevant to deciding whether Skein should measure the yarn core or its outer hairy envelope.

### Sengupta, Roy & Sengupta (2015), *Measurement* 59, 96–109

Article: https://doi.org/10.1016/j.measurement.2014.09.028

- The authors describe a lower-cost yarn parameterisation system built with a USB camera, a mechanical setup for moving yarn, image processing, and calibration from a standard wire.
- They compare jute and cotton yarn measurements against a commercial Uster tester 3.
- This supports the design principle that controlled mechanics, a reference object, and comparison against an independent instrument are central to credible low-cost yarn measurement. The abstract does not establish 17 µm performance for the current phone-based design.

### Imaging resolution and sampling

- Nikon MicroscopyU, *Spatial Resolution in Digital Imaging*: https://www.microscopyu.com/tutorials/spatial-resolution-in-digital-imaging
- Molecular Expressions / Florida State University, *Digital Image Basics*: https://micro.magnet.fsu.edu/primer/digitalimaging/digitalimagebasics.html

Both explain that image quality depends on optical resolution as well as pixel sampling; additional digital pixels cannot restore detail that the optics failed to resolve. A software calibration converts pixels to physical units but does not itself prove spatial resolution.

### Metrology and decision uncertainty

- NIST, *Measurement Uncertainty*: https://www.nist.gov/itl/sed/topic-areas/measurement-uncertainty
- NIST TN 1297, Appendix D4: https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-d4
- JCGM 106:2012, *The role of measurement uncertainty in conformity assessment*: https://www.bipm.org/en/doi/10.59161/jcgm106-2012

These references require attention to reference standards, implementation, repeatability, reproducibility, and uncertainty when reporting or making conformity decisions. Thousands of adjacent width samples from one image are correlated; they are not thousands of independent repeat captures.

### Relevant standardized methods (scope note)

- ISO 137:2015, *Wool — Determination of fibre diameter — Projection microscope method*: https://www.iso.org/standard/61688.html
- ASTM D2130-22, *Diameter of Wool and Other Animal Fibers by Microprojection*: https://store.astm.org/d2130-22.html

These are methods for **fibre** diameter in defined measurement setups, not direct proof that a phone image of the outside of a yarn bundle measures its 3-D yarn diameter.

## 3. Findings from auditing the current code

1. The prior estimator skeletonized a threshold mask, measured twice the distance transform at skeleton pixels, and summarized all local samples. It did not explicitly sample cross-sections perpendicular to the yarn axis.
2. Otsu thresholding and selecting the largest connected component can fail on loose hairs, blur, low contrast, shadow, and irregular fibres. Boundary overlays must be inspected.
3. The local width samples from one image are correlated. Their sample standard deviation describes within-image variation, not a complete measurement-uncertainty budget.
4. The calibration marker can calculate pixels/mm only if its printed physical size is correct and the calibration target and yarn occupy the same plane under the same optics, zoom, focus, resolution, and orientation. Same camera height alone is insufficient.
5. The code previously used a three-pixel advisory threshold for the 17 µm difference. Version 3 introduced a **provisional five-pixel screening gate**. This is an engineering gate to reject gross undersampling, not a guarantee of measurement capability.
6. A deep-learning segmentation model is not yet justified: there is no labelled, representative set of real yarn images and expert boundary annotations. A model cannot fix optical undersampling or uncalibrated scale. Begin with a transparent computer-vision baseline; add ML only if a labeled dataset and held-out specimens show a measurable improvement.

## 4. Updated algorithm under test (v4)

Version 4 keeps the transparent Otsu baseline and adds a Sauvola local-threshold candidate. The mask's principal axis is aligned horizontally, and apparent body width is sampled using perpendicular cross-sections. When grayscale data is available, an additional midpoint-intensity interpolation estimates fractional-pixel edge crossings. This subpixel estimator is a candidate numerical technique, not proof of subpixel optical accuracy; it can be biased by blur, lighting, shadows, hairiness, threshold errors and texture.

Known limitations remain: threshold masks can include connected hairs; PCA alignment assumes one roughly straight strand; raster rotation changes boundaries; image columns are correlated; and the same binary mask can give a precise-looking number despite wrong boundaries. We are not calling this final or validated.

## 5. Hardware feasibility gate

At the earlier 100 px/mm setting, 0.017 mm is represented by only 1.7 pixels. The application should return INCONCLUSIVE at that scale under the provisional rule.

For orientation:

| Image scale | Pixels corresponding to 0.017 mm |
|---:|---:|
| 100 px/mm | 1.7 px |
| 176 px/mm | about 3 px |
| 294 px/mm | about 5 px |
| 588 px/mm | about 10 px |

Five or ten pixels across the *difference* do not establish accuracy by themselves. Optical blur, sensor sampling, demosaicing, focus, edge-definition, scale variation, and specimen variability still have to be validated. A phone with a macro optical setup or a low-cost digital microscope may be needed if the ordinary phone lens cannot meet the gate.

## 6. Experimental protocol

### A. Freeze the capture method

- Use one phone/lens and a rigid stand; disable automatic lens switching, digital zoom, and changes of resolution/orientation.
- Keep camera-to-sample distance, focus, magnification, lighting and exposure fixed. A dark matte background with controlled diffuse illumination is a starting point for light yarn.
- Use a printed marker only as a provisional scale target; print at Actual Size and physically verify its 10 mm side. More defensible tests should use a known-size reference in the yarn plane and record its uncertainty.
- Use a consistent strand-preparation method. State whether the target is core-body width excluding loose hairiness, and control tension and compression.

### B. Pilot captures

For multiple independently sourced yarn strands, capture at least three repositioned images per strand under the frozen setup. Include pairs whose independently measured reference differences are near 0, below 17 µm, around 17 µm, and above 17 µm. The reference measurements must not come from Skein itself. A calibrated digital microscope with a stage scale or an appropriate laboratory method is preferable; a caliper may compress yarn and may measure a different physical quantity.

### C. Report these metrics

- Scale calibration repeatability and any scale drift after repositioning.
- Boundary segmentation quality and failure rate.
- Bias and absolute error against the independent reference.
- Between-capture repeatability (not just variation across pixels in one image).
- Bland–Altman agreement plot or equivalent error analysis.
- Classification sensitivity for pairs differing by at least 17 µm, false-SAME rate, false-DIFFERENT rate, and INCONCLUSIVE rate.
- Results split by **specimen/pair**, never by random adjacent image columns, to avoid leakage.

### D. Provisional acceptance goals (to be agreed before testing)

A conservative starting goal is that the 95% uncertainty bound for the *difference between two measurements* should be no larger than about **0.0085 mm** (half of the 0.017 mm target difference), with an acceptably low false-SAME rate on independently verified pairs at or above 17 µm. This is a proposed project criterion—not a claim from a paper or a universal standard. The final threshold and sample size should be set with the supervisor/user and reference instrument before testing.

## 7. ML plan implemented, model choice still pending

Version 4 includes optional training/inference code for U-Net++/ResNet34 and DeepLabV3+/ResNet34. The app exposes an AI option only when a local checkpoint exists. It does not ship with a checkpoint: no representative expert-annotated dataset was available, so claiming a particular model is best would be unsupported.

Train both candidates on the same expert-reviewed yarn-core masks and physical-specimen-level splits. Compare against Otsu and Sauvola baselines using held-out Dice, IoU, one-pixel boundary F1, segmentation failure rate and, most importantly, absolute width error against independent reference measurements. A general-purpose promptable segmenter such as SAM 2 can help draft masks, but it should not be the final measurement backend without equivalent held-out validation. Follow `ML_MODEL_SELECTION.md` for the labeling and training protocol.

## 8. Immediate next actions

1. Install this version and confirm that a calibration which maps the target difference to fewer than five pixels is blocked as INCONCLUSIVE.
2. Capture an actual calibration-target image using the exact planned phone setup.
3. Capture real yarn samples, preserve original full-resolution files, and examine the boundary overlays.
4. Obtain an independent reference method for a subset of the exact same samples.
5. Decide whether the phone setup passes the feasibility gate. If it does not, test macro optics or a digital microscope before investing in ML.

## References

1. Abdelkader, M. (2022). MATLAB Algorithms for Diameter Measurements of Textile Yarns and Fibers through Image Processing Techniques. *Materials*, 15(4), 1299. https://doi.org/10.3390/ma15041299
2. Sengupta, A., Roy, S., & Sengupta, S. (2015). Development of a low cost yarn parameterisation unit by image processing. *Measurement*, 59, 96–109. https://doi.org/10.1016/j.measurement.2014.09.028
3. National Institute of Standards and Technology. Measurement Uncertainty. https://www.nist.gov/itl/sed/topic-areas/measurement-uncertainty
4. NIST TN 1297, Appendix D4. https://www.nist.gov/pml/nist-technical-note-1297/nist-tn-1297-appendix-d4
5. Joint Committee for Guides in Metrology. JCGM 106:2012. https://www.bipm.org/en/doi/10.59161/jcgm106-2012
6. Nikon MicroscopyU. Spatial Resolution in Digital Imaging. https://www.microscopyu.com/tutorials/spatial-resolution-in-digital-imaging
7. Molecular Expressions, Florida State University. Digital Image Basics. https://micro.magnet.fsu.edu/primer/digitalimaging/digitalimagebasics.html
8. ISO 137:2015. Wool — Determination of fibre diameter — Projection microscope method. https://www.iso.org/standard/61688.html
9. ASTM D2130-22. Diameter of Wool and Other Animal Fibers by Microprojection. https://store.astm.org/d2130-22.html

## 9. v4 implementation status

- The app now includes a restrained interaction layer (short entrance/hover transitions, workflow steps, staged processing feedback, reduced-motion support) and offers Otsu/Sauvola segmentation options.
- A grayscale midpoint-intensity edge interpolation candidate is enabled for cross-sections, but it still requires physical validation.
- U-Net++ and DeepLabV3+ training/inference scripts are included, but no trained weights are bundled.
- Fifteen automated tests pass in the development container; they cover software and synthetic inputs only.

This updates software readiness, not metrology validation. See `PROJECT_STATUS.md` for the outstanding physical validation gates and `ML_MODEL_SELECTION.md` for the model benchmark workflow.
