"""Skein — Yarn Diameter Comparator (research prototype with interactive workflow.

Run: streamlit run yarn_diameter_app.py

This is a screening prototype, not a validated metrology instrument. It can
estimate apparent widths from controlled photographs but cannot guarantee that
a 0.017 mm diameter difference is detectable until validated against reliable
physical reference measurements on independent samples.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from yarn_calibration import (
    MARKER_SIDE_MM,
    calibration_target_svg,
    detect_calibration,
    load_profile,
    save_profile,
    clear_profile,
)
from yarn_ml import available_models, checkpoint_path, load_trained_model

from yarn_measurement import (
    MeasurementConfig,
    segment_yarn as core_segment_yarn,
    estimate_widths as core_estimate_widths,
    summarize_widths,
    compare_summaries,
)

st.set_page_config(
    page_title="Skein | Yarn Diameter Comparator",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# A saved profile belongs to this local app folder and this fixed camera setup.
# It saves users from entering the image scale for every pair of yarn images.
CALIBRATION_PROFILE_PATH = Path(__file__).resolve().parent / "calibration_profile.json"
MODEL_DIR = Path(__file__).resolve().parent / "models"

@st.cache_resource(show_spinner=False)
def cached_segmentation_model(architecture: str, checkpoint: str, mtime_ns: int):
    """Load one trained model once per Streamlit process and checkpoint version."""
    return load_trained_model(architecture, checkpoint)

if "calibration_profile" not in st.session_state:
    st.session_state["calibration_profile"] = load_profile(CALIBRATION_PROFILE_PATH)


def load_rgb(uploaded):
    """Load an uploaded photo as RGB while respecting EXIF orientation."""
    with Image.open(BytesIO(uploaded.getvalue())) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        return np.array(normalized)

# ---------------------------------------------------------------------------
# Visual system — warm ivory, forest green, sage, and plum.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    :root {
      --ivory: #F7F5EC; --paper: #FFFFFF; --ink: #22352C;
      --forest: #233D31; --forest-2: #355544; --sage: #8F9F7B;
      --sage-light: #E9EDDF; --cream: #F3EBD1; --line: #E3E4D9;
      --muted: #70796F; --plum: #432331; --rose: #C78375;
      --rose-light: #F5E7E1;
    }
    html, body, [class*="css"], .stApp { font-family: 'DM Sans', ui-sans-serif, system-ui, sans-serif; }
    .stApp { background: var(--ivory); color: var(--ink); }
    .block-container { max-width: 1380px; padding-top: 1.35rem; padding-bottom: 3rem; }
    header[data-testid="stHeader"] { background: rgba(247,245,236,.92); }
    #MainMenu, footer { visibility: hidden; }
    h1, h2, h3, p { color: var(--ink); }
    .brand-row { display:flex; align-items:center; justify-content:space-between; margin: .15rem 0 1.15rem; gap:1rem; }
    .brand { display:flex; gap:.75rem; align-items:center; }
    .brand-icon { width:42px; height:42px; border-radius:14px; background:var(--sage-light); display:flex; align-items:center; justify-content:center; font-size:24px; }
    .brand-name { letter-spacing:.14em; font-size:1.1rem; font-weight:800; line-height:1.2; color:var(--forest); }
    .brand-caption { letter-spacing:.11em; text-transform:uppercase; font-size:.58rem; font-weight:700; color:var(--muted); margin-top:.22rem; }
    .beta-pill { border-radius:999px; background:var(--cream); color:var(--plum); padding:.55rem .9rem; font-size:.68rem; font-weight:800; letter-spacing:.06em; }
    .hero { background:var(--forest); border-radius:24px; color:white; padding:2rem 2.1rem; display:flex; justify-content:space-between; align-items:center; gap:2rem; min-height:205px; overflow:hidden; margin-bottom:1.35rem; }
    .hero-copy { max-width:760px; }
    .eyebrow { display:inline-block; border-radius:999px; padding:.38rem .72rem; background:var(--forest-2); color:var(--cream); font-size:.65rem; font-weight:800; letter-spacing:.1em; margin-bottom:.7rem; }
    .hero h1 { color:white; font-size:clamp(2.2rem,3.7vw,3.25rem); line-height:1.04; letter-spacing:-.045em; margin:0 0 .75rem; }
    .hero p { color:#D8E2D7; max-width:690px; line-height:1.6; font-size:.97rem; margin:0; }
    .hero-art { min-width:340px; width:340px; height:155px; background:var(--forest-2); border-radius:20px; display:flex; align-items:center; justify-content:space-evenly; position:relative; }
    .yarn-orbit { width:118px; height:118px; border-radius:50%; background:var(--sage); display:flex; align-items:center; justify-content:center; }
    .yarn-orbit-mid { width:78px; height:78px; border-radius:50%; background:var(--cream); display:flex; align-items:center; justify-content:center; }
    .yarn-orbit-core { width:42px; height:42px; border-radius:50%; background:var(--rose); display:flex; align-items:center; justify-content:center; }
    .yarn-orbit-core:after { content:''; width:14px; height:14px; border-radius:50%; background:var(--plum); }
    .target-stack { display:flex; flex-direction:column; gap:.55rem; }
    .target-chip { padding:.8rem 1rem; background:var(--plum); border-radius:14px; min-width:155px; }
    .target-value { font-size:1.7rem; font-weight:800; line-height:1.1; color:white; }
    .target-label { color:#EEDFD4; font-size:.56rem; font-weight:800; letter-spacing:.04em; margin-top:.35rem; }
    .target-note { background:#E1E8D8; color:var(--forest); border-radius:11px; padding:.62rem .72rem; font-size:.69rem; font-weight:700; }
    .section-head { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin:1.45rem 0 .7rem; }
    .section-title { font-size:1.4rem; font-weight:800; letter-spacing:-.025em; margin:0; color:var(--ink); }
    .section-sub { color:var(--muted); font-size:.84rem; margin:.2rem 0 0; }
    .step-pill { background:var(--sage-light); color:var(--forest); border-radius:999px; padding:.62rem .95rem; font-size:.7rem; font-weight:800; white-space:nowrap; }
    div[data-testid="stFileUploader"] section { border:1.5px dashed #9CAB8A; border-radius:15px; background:#F3F5EC; padding:.8rem .85rem; }
    div[data-testid="stFileUploader"] section > div { gap:.35rem; }
    div[data-testid="stFileUploader"] small { color:var(--muted); }
    .stButton > button[kind="primary"] { background:var(--forest); color:#fff; border:1px solid var(--forest); border-radius:12px; padding:.7rem 1.05rem; font-weight:700; min-height:2.7rem; }
    .stButton > button[kind="primary"]:hover { background:var(--forest-2); border-color:var(--forest-2); }
    .stButton > button { border-radius:12px; border-color:#C8D0BE; color:var(--forest); font-weight:700; }
    [data-testid="stMetric"] { border:1px solid var(--line); background:white; padding:1.1rem 1.2rem; border-radius:16px; }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    [data-testid="stMetricValue"] { color:var(--forest); font-weight:800; }
    div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:15px; background:#fff; }
    .info-card { height:100%; border:1px solid var(--line); border-radius:18px; background:white; padding:1.1rem 1.2rem; margin-bottom:.25rem; }
    .card-kicker { font-size:.64rem; font-weight:800; letter-spacing:.1em; color:#5F7758; text-transform:uppercase; }
    .card-title { font-size:1.03rem; font-weight:800; margin:.25rem 0 .45rem; color:var(--ink); }
    .card-copy { color:var(--muted); font-size:.79rem; line-height:1.5; }
    .warning-note { border-radius:14px; background:#F3EBD1; color:var(--plum); padding:.9rem 1rem; font-size:.8rem; line-height:1.55; margin:.5rem 0 1rem; }
    .result-box { border-radius:19px; background:var(--plum); color:white; padding:1.25rem 1.4rem; margin-top:1rem; }
    .result-title { color:white; font-size:1.35rem; font-weight:800; margin:0 0 .3rem; }
    .result-copy { color:#E4D8D8; font-size:.83rem; line-height:1.55; margin:0; }
    .status-chip { display:inline-block; border-radius:999px; padding:.48rem .82rem; font-size:.67rem; font-weight:800; letter-spacing:.02em; }
    .status-green { background:#DDE9D9; color:#284B31; }
    .status-red { background:#E8B0A3; color:#462633; }
    .status-yellow { background:#F3EBD1; color:#432331; }
    .footer-note { color:var(--muted); font-size:.69rem; display:flex; justify-content:space-between; gap:1rem; margin-top:1.25rem; padding-top:.75rem; border-top:1px solid var(--line); }
    .tiny-muted { color:var(--muted); font-size:.72rem; }
    /* Product-like motion: short, quiet transitions; no endless decorative motion. */
    @keyframes fadeRise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
    @keyframes softPulse { 0%,100% { box-shadow:0 0 0 0 rgba(95,119,88,.12); } 50% { box-shadow:0 0 0 5px rgba(95,119,88,0); } }
    @keyframes progressSweep { 0% { background-position:150% 0; } 100% { background-position:-150% 0; } }
    .brand-row { animation:fadeRise .32s ease-out both; }
    .hero { animation:fadeRise .42s .04s ease-out both; }
    .section-head { animation:fadeRise .32s ease-out both; }
    .info-card { transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
    .info-card:hover { transform:translateY(-2px); border-color:#B5C3A8; box-shadow:0 8px 22px rgba(34,53,44,.055); }
    div[data-testid="stFileUploader"] section { transition:border-color .18s ease, background .18s ease, box-shadow .18s ease; }
    div[data-testid="stFileUploader"] section:hover { border-color:var(--forest); background:#F0F4E9; box-shadow:0 0 0 3px rgba(143,159,123,.12); }
    .stButton > button { transition:transform .14s ease, box-shadow .14s ease, background .14s ease; }
    .stButton > button:hover { transform:translateY(-1px); box-shadow:0 5px 12px rgba(35,61,49,.10); }
    [data-testid="stMetric"] { transition:transform .18s ease, border-color .18s ease; }
    [data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:#B5C3A8; }
    .result-box { animation:fadeRise .38s ease-out both; }
    .workflow-track { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem; margin:.25rem 0 1.1rem; }
    .workflow-step { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.7rem .85rem; display:flex; gap:.65rem; align-items:center; min-height:52px; transition:background .18s ease,border-color .18s ease; }
    .workflow-step .step-no { width:25px; height:25px; border-radius:8px; display:flex; align-items:center; justify-content:center; background:#F0F1EB; color:var(--muted); font-size:.72rem; font-weight:800; flex:0 0 auto; }
    .workflow-step strong { display:block; color:var(--ink); font-size:.77rem; }
    .workflow-step small { display:block; color:var(--muted); font-size:.66rem; margin-top:.12rem; }
    .workflow-step.done { border-color:#B8C7AD; background:#F4F7F0; }
    .workflow-step.done .step-no { background:var(--forest); color:white; }
    .workflow-step.active { border-color:#C4A96D; background:#FCF8ED; }
    .workflow-step.active .step-no { background:var(--cream); color:var(--plum); animation:softPulse 2s ease-in-out 2; }
    .hero { background:linear-gradient(112deg,#fff 0%,#fff 67%,#EEF1E7 100%); color:var(--ink); border:1px solid var(--line); min-height:158px; padding:1.6rem 1.8rem; }
    .hero h1 { color:var(--forest); font-size:clamp(1.9rem,3vw,2.65rem); letter-spacing:-.04em; margin:.05rem 0 .65rem; }
    .hero p { color:var(--muted); max-width:690px; line-height:1.55; font-size:.92rem; }
    .hero .eyebrow { background:#EAF0E4; color:var(--forest); margin-bottom:.55rem; }
    .hero-ops { min-width:250px; max-width:305px; border:1px solid var(--line); border-radius:15px; background:rgba(255,255,255,.84); padding:1rem 1.05rem; }
    .hero-ops-label { font-size:.62rem; letter-spacing:.09em; color:var(--muted); font-weight:800; text-transform:uppercase; }
    .hero-ops-value { color:var(--forest); font-size:1.7rem; font-weight:800; letter-spacing:-.035em; margin:.18rem 0 .12rem; }
    .hero-ops-copy { color:var(--muted); font-size:.7rem; line-height:1.45; }
    .local-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#6D8B61; margin-right:.38rem; vertical-align:middle; }
    [data-testid="stProgress"] > div > div > div { background:linear-gradient(90deg,#355544,#8F9F7B,#355544); background-size:200% 100%; animation:progressSweep 1.5s linear infinite; }
    @media (max-width: 850px) { .hero { padding:1.2rem; } .hero-ops { display:none; } .footer-note { flex-direction:column; } .workflow-track { grid-template-columns:1fr; } }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; scroll-behavior:auto !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brand-row">
      <div class="brand"><div class="brand-icon">🧶</div><div><div class="brand-name">SKEIN</div><div class="brand-caption">Yarn measurement studio</div></div></div>
      <div class="beta-pill">BETA · RESEARCH PROTOTYPE</div>
    </div>
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">MEASUREMENT WORKSPACE · RESEARCH BUILD</div>
        <h1>Yarn diameter comparison</h1>
        <p>Compare two yarn samples using the same camera setup. Review the detected boundaries, scale check, and measurement quality before accepting a result.</p>
      </div>
      <div class="hero-ops">
        <div class="hero-ops-label">Sensitivity target</div>
        <div class="hero-ops-value">17 μm</div>
        <div class="hero-ops-copy">Smallest difference to validate—not a guaranteed accuracy figure.</div>
        <div class="hero-ops-copy" style="margin-top:.65rem"><span class="local-dot"></span>Images are processed locally in this app.</div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-head"><div><div class="section-title">01 · Add your two yarn photos</div><div class="section-sub">One straight strand per photo, with the whole yarn visible.</div></div><div class="step-pill">STEP 1 OF 3</div></div>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns(2, gap="large")
file_types = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
with col_a:
    st.markdown('<div class="info-card"><div class="card-kicker">Sample A</div><div class="card-title">First yarn</div><div class="card-copy">Choose a clear photo of the first yarn strand. Avoid crossed fibers and cropped edges.</div></div>', unsafe_allow_html=True)
    file_a = st.file_uploader("Choose photo for Sample A", type=file_types, key="sample_a", help="PNG, JPG, BMP or TIFF. Use the same camera setup for both samples.")
with col_b:
    st.markdown('<div class="info-card"><div class="card-kicker">Sample B</div><div class="card-title">Second yarn</div><div class="card-copy">Photograph the second strand at the same camera height, focus, lighting and magnification.</div></div>', unsafe_allow_html=True)
    file_b = st.file_uploader("Choose photo for Sample B", type=file_types, key="sample_b", help="PNG, JPG, BMP or TIFF. Do not change zoom, distance or image resolution between captures.")

if file_a is not None or file_b is not None:
    p1, p2 = st.columns(2, gap="large")
    if file_a is not None:
        with p1:
            st.image(Image.open(BytesIO(file_a.getvalue())), caption="Sample A · uploaded photo", use_container_width=True)
    if file_b is not None:
        with p2:
            st.image(Image.open(BytesIO(file_b.getvalue())), caption="Sample B · uploaded photo", use_container_width=True)

photo_pair_ready = file_a is not None and file_b is not None
profile_for_steps = st.session_state.get("calibration_profile")
steps_html = f"""
<div class="workflow-track">
  <div class="workflow-step {'done' if photo_pair_ready else 'active'}"><span class="step-no">{'✓' if photo_pair_ready else '1'}</span><div><strong>Photos</strong><small>{'Both samples added' if photo_pair_ready else 'Add Sample A and B'}</small></div></div>
  <div class="workflow-step {'done' if profile_for_steps else ('active' if photo_pair_ready else '')}"><span class="step-no">{'✓' if profile_for_steps else '2'}</span><div><strong>Camera scale</strong><small>{'Saved profile available' if profile_for_steps else 'Calibrate once'}</small></div></div>
  <div class="workflow-step {'active' if photo_pair_ready and profile_for_steps else ''}"><span class="step-no">3</span><div><strong>Review result</strong><small>{'Ready for quality checks' if photo_pair_ready and profile_for_steps else 'Needs photos and calibration'}</small></div></div>
</div>
"""
st.markdown(steps_html, unsafe_allow_html=True)

st.markdown(
    '<div class="section-head"><div><div class="section-title">02 · Measurement setup</div><div class="section-sub">A comparison is only as trustworthy as its calibration and image quality.</div></div><div class="step-pill">TARGET: 0.017 mm</div></div>',
    unsafe_allow_html=True,
)

cal_col, target_col = st.columns(2, gap="large")
with cal_col:
    st.markdown('<div class="info-card"><div class="card-kicker">One-time setup</div><div class="card-title">Let Skein measure the image scale</div><div class="card-copy">Download the calibration target, print it at actual size, photograph it at your fixed yarn plane, and upload that photo once. Skein detects the marker and saves a local camera profile. You do not need to enter either yarn’s diameter.</div></div>', unsafe_allow_html=True)
    try:
        target_svg_data = calibration_target_svg(MARKER_SIDE_MM)
        st.download_button(
            "Download 10 mm calibration target (SVG)",
            data=target_svg_data,
            file_name="skein_calibration_target_10mm.svg",
            mime="image/svg+xml",
            help="Print at 100% / Actual Size. Do not use Fit to Page. The marker's outside square is 10 mm."
        )
    except Exception as target_error:
        st.error(f"Automatic target generation is unavailable: {target_error}")

    calibration_photo = st.file_uploader(
        "First-time setup: upload a photo of the calibration target",
        type=file_types,
        key="calibration_reference_photo",
        help="Capture the printed marker flat at the same plane, camera height, lens, zoom, focus, resolution, and lighting as the yarn photos. Keep the marker fully in frame."
    )
    if st.button("Detect marker and save camera profile", type="primary", key="detect_and_save_calibration"):
        if calibration_photo is None:
            st.error("Upload a photo of the printed Skein calibration target first.")
        else:
            try:
                calibration_rgb = load_rgb(calibration_photo)
                detected_profile = detect_calibration(calibration_rgb, marker_side_mm=MARKER_SIDE_MM)
                save_profile(detected_profile, CALIBRATION_PROFILE_PATH)
                st.session_state["calibration_profile"] = detected_profile
                st.success("Calibration profile saved on this computer.")
            except Exception as calibration_error:
                st.error(str(calibration_error))

    if st.session_state.get("calibration_profile") is not None:
        if st.button("Reset saved camera profile", key="reset_camera_profile"):
            clear_profile(CALIBRATION_PROFILE_PATH)
            st.session_state["calibration_profile"] = None
            st.rerun()

    with st.expander("How the one-time calibration works"):
        st.markdown(
            "1. Download the SVG target and print it at **100% / Actual Size**. Do not select Fit to Page.\n\n"
            "2. Measure the black marker's outside square with a ruler once; it should be 10 mm. If it is not, do not use that print.\n\n"
            "3. Place the target flat at the same height/plane as the yarn and capture a sharp photo.\n\n"
            "4. Keep the same phone, lens, zoom, focus, resolution, distance, and orientation for calibration and both yarn photos. Lock focus if your phone supports it.\n\n"
            "5. Upload the calibration photo above. Skein detects the known marker size and calculates pixels/mm automatically."
        )

with target_col:
    st.markdown('<div class="info-card"><div class="card-kicker">Comparison goal</div><div class="card-title">Resolve very small differences</div><div class="card-copy">The target is to distinguish a difference as small as 0.017 mm. The app has not proven that capability yet. Calibration makes the units automatic; it does not guarantee the optical resolution.</div></div>', unsafe_allow_html=True)
    tolerance_mm = st.number_input("Maximum difference still considered ‘same’ (mm)", min_value=0.0, max_value=1.0, value=0.0080, step=0.001, format="%.4f", help="Temporary screening threshold. A difference above this threshold is intended to be flagged as different, but the system must be validated against real samples.")

calibration_profile = st.session_state.get("calibration_profile")
calibration_verified = calibration_profile is not None
px_per_mm = float(calibration_profile["pixels_per_mm"]) if calibration_profile else 100.0
if calibration_profile:
    st.success(
        f"Saved camera profile loaded: {px_per_mm:.2f} pixels/mm · "
        f"calibration photo {calibration_profile['image_width_px']} × {calibration_profile['image_height_px']} px. "
        "The scale is filled automatically for new comparisons."
    )
    target_in_pixels = px_per_mm * 0.017
    st.caption(f"At the saved scale, your 0.017 mm target corresponds to about {target_in_pixels:.2f} pixels in the image. Skein uses 5 pixels as a provisional screening floor; that is not a guarantee of detection accuracy.")
    for calibration_warning in calibration_profile.get("warnings", []):
        st.warning(calibration_warning)
else:
    st.info("No camera profile is saved yet. Complete the one-time calibration above; measurements may be previewed, but comparison status will remain INCONCLUSIVE.")

if tolerance_mm >= 0.017:
    st.warning("Your similarity threshold is at or above 0.017 mm. A measured difference of 0.017 mm could be labeled ‘same within tolerance’. Lower the threshold if that difference must be flagged.")
calibration_profile_quality_ok = bool(calibration_profile is not None and not calibration_profile.get("warnings", []))

with st.expander("Advanced image settings", expanded=False):
    trained_models = available_models()
    segmentation_labels = {
        "Otsu threshold · baseline": "otsu",
        "Sauvola local threshold · uneven lighting": "sauvola",
    }
    if "unetplusplus" in trained_models:
        segmentation_labels["Trained AI · U-Net++ / ResNet34"] = "unetplusplus"
    if "deeplabv3plus" in trained_models:
        segmentation_labels["Trained AI · DeepLabV3+ / ResNet34"] = "deeplabv3plus"
    c1, c2, c3, c4 = st.columns([1.3, 1.2, 1.25, 1.0])
    with c1:
        segmentation_label = st.selectbox(
            "Boundary detection method", list(segmentation_labels.keys()),
            help="Use classical methods until a yarn-specific model has been trained and checked on held-out real images."
        )
        segmentation_method = segmentation_labels[segmentation_label]
    with c2:
        polarity = st.selectbox("Yarn/background contrast", ["Light yarn on dark background", "Dark yarn on light background"])
    with c3:
        correct_illumination_enabled = st.checkbox("Correct uneven illumination", value=False, help="May reduce slow background gradients. Always inspect the overlay because correction can alter edges.")
        median_filter_enabled = st.checkbox("Apply 3×3 median filter", value=False, help="Optional noise reduction before segmentation. Keep off unless the overlay and width estimates improve on the same images.")
    with c4:
        min_width_mm = st.number_input("Minimum width (mm)", min_value=0.001, max_value=100.0, value=0.02, step=0.01, format="%.4f")
        max_width_mm = st.number_input("Maximum width (mm)", min_value=0.01, max_value=1000.0, value=5.0, step=0.1, format="%.3f")
    if not trained_models:
        st.info("No custom-trained yarn AI model is installed yet. That's intentional: a U-Net++ or DeepLabV3+ checkpoint must be trained on reviewed yarn-core masks and evaluated on separate specimens before it is trusted. See ML_MODEL_SELECTION.md.")

if segmentation_method in {"unetplusplus", "deeplabv3plus"}:
    model_file = checkpoint_path(segmentation_method)
    try:
        segmentation_model = cached_segmentation_model(segmentation_method, str(model_file), model_file.stat().st_mtime_ns)
        st.caption(f"AI segmentation active: {segmentation_method} · device: {getattr(segmentation_model, '_skein_device', 'unknown')} · checkpoint: {model_file.name}")
    except Exception as model_error:
        segmentation_model = None
        st.error(f"The selected AI model could not be loaded: {model_error}")
else:
    segmentation_model = None

st.markdown(
    '<div class="warning-note"><b>Before you compare:</b> keep the camera fixed; use the same zoom, focus, lighting and image resolution; place each yarn straight on a matte contrasting surface; and leave background visible on all four sides. Calibration must be verified independently.</div>',
    unsafe_allow_html=True,
)


def overlay_mask(rgb, mask, skeleton=None):
    out = rgb.copy()
    selected = mask > 0
    tint = np.zeros_like(out)
    tint[:, :, 1] = 220
    out[selected] = (0.58 * out[selected] + 0.42 * tint[selected]).astype(np.uint8)
    if skeleton is not None:
        out[skeleton] = [199, 104, 90]
    return out


def analyze(rgb, label, model=None):
    config = MeasurementConfig(
        pixels_per_mm=float(px_per_mm),
        min_width_mm=float(min_width_mm),
        max_width_mm=float(max_width_mm),
        light_yarn_on_dark=(polarity == "Light yarn on dark background"),
        correct_illumination=bool(correct_illumination_enabled),
        median_filter=bool(median_filter_enabled),
        segmentation_method=segmentation_method,
    )
    mask, processed_gray, segmentation_warnings, diagnostics = core_segment_yarn(rgb, config, model=model)
    widths, skeleton, width_warnings = core_estimate_widths(mask, config, gray=processed_gray)
    summary = summarize_widths(widths)
    return {
        "label": label, "mask": mask, "skeleton": skeleton, "widths": widths,
        "mean": summary["mean_mm"], "median": summary["median_mm"], "std": summary["std_mm"],
        "n": summary["n_local_samples"], "overlay": overlay_mask(rgb, mask, skeleton), "original": rgb,
        "warnings": segmentation_warnings + width_warnings, "diagnostics": diagnostics,
        "processed_gray": processed_gray,
    }


def fmt_mm(value):
    return f"{value:.4f} mm"


def critical_quality_issues(result):
    issues = list(result["warnings"])
    diagnostics = result["diagnostics"]
    # New diagnostics distinguish strand-end cropping from clipping of the
    # width-defining sides. Fall back conservatively for older diagnostics.
    width_edge_touched = diagnostics.get(
        "touches_width_boundary_border", diagnostics.get("touches_image_border", False)
    )
    if width_edge_touched:
        issues.append("The detected yarn width boundary touches an image border.")
    return list(dict.fromkeys(issues))


st.markdown(
    '<div class="section-head"><div><div class="section-title">03 · Review the comparison</div><div class="section-sub">Upload both photos to run the current cross-sectional-width measurement method.</div></div></div>',
    unsafe_allow_html=True,
)

if file_a is None or file_b is None:
    st.markdown(
        '<div class="result-box"><div class="result-title">Your comparison will appear here</div><p class="result-copy">Add Sample A and Sample B above. We’ll show the estimated width, boundary overlay, quality warnings, and a preliminary difference screen.</p><div style="margin-top:.85rem"><span class="status-chip status-green">SAME WITHIN TOLERANCE</span> &nbsp; <span class="status-chip status-red">DIFFERENT</span> &nbsp; <span class="status-chip status-yellow">INCONCLUSIVE</span></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Possible outcomes shown above are examples only; no comparison has been made yet.")
elif segmentation_method in {"unetplusplus", "deeplabv3plus"} and segmentation_model is None:
    st.error("The AI model is selected but could not be loaded. No fallback to classical segmentation has been made. Check the optional ML installation and the checkpoint, then try again.")
else:
    try:
        with st.status("Reviewing the two samples", expanded=False) as analysis_status:
            progress = st.progress(0, text="Reading the original-resolution photographs…")
            rgb_a = load_rgb(file_a)
            rgb_b = load_rgb(file_b)
            progress.progress(18, text="Checking image dimensions and loading calibration…")
            analysis_status.write("Sample A: detecting the yarn body and cross-sections.")
            result_a = analyze(rgb_a, "Sample A", model=segmentation_model)
            progress.progress(48, text="Sample A processed · checking Sample B…")
            analysis_status.write("Sample B: applying the same measurement method.")
            result_b = analyze(rgb_b, "Sample B", model=segmentation_model)
            progress.progress(78, text="Comparing estimates and quality flags…")
            difference = abs(result_a["median"] - result_b["median"])
            comparison = compare_summaries({"median_mm": result_a["median"]}, {"median_mm": result_b["median"]}, float(tolerance_mm))
            progress.progress(100, text="Analysis pass complete · inspect the checks below.")
            analysis_status.update(label="Analysis pass complete", state="complete", expanded=False)

        # A stored scale is only reusable when the photo pixel dimensions still match
        # the calibrated camera profile and both sample images match each other.
        profile_dimensions_match = bool(
            calibration_profile is not None
            and rgb_a.shape[1] == int(calibration_profile["image_width_px"])
            and rgb_a.shape[0] == int(calibration_profile["image_height_px"])
            and rgb_b.shape[1] == int(calibration_profile["image_width_px"])
            and rgb_b.shape[0] == int(calibration_profile["image_height_px"])
        )

        # A fallback scale is for preview calculations only. Never label it as mm.
        measurement_scale_usable = bool(
            calibration_verified
            and calibration_profile_quality_ok
            and profile_dimensions_match
            and rgb_a.shape[:2] == rgb_b.shape[:2]
        )
        def display_width(value_mm: float) -> str:
            if measurement_scale_usable:
                return fmt_mm(value_mm)
            return f"{value_mm * px_per_mm:.1f} px (uncalibrated)"

        if calibration_profile is not None and not profile_dimensions_match:
            st.warning(
                "These yarn photos do not match the pixel dimensions of the saved calibration photo. "
                "The app will show a preview, but the final decision will be INCONCLUSIVE. "
                "Retake the photos using the same resolution and orientation as calibration."
            )

        # Any visible quality warnings, an absent/incompatible profile, or different sample
        # dimensions block a confident classification.
        issues_a = critical_quality_issues(result_a)
        issues_b = critical_quality_issues(result_b)
        all_issues = [("Sample A", x) for x in issues_a] + [("Sample B", x) for x in issues_b]
        quality_ok = len(all_issues) == 0
        if (not calibration_verified or not calibration_profile_quality_ok or not profile_dimensions_match
                or rgb_a.shape[:2] != rgb_b.shape[:2] or not quality_ok):
            status = "INCONCLUSIVE"
            status_class = "status-yellow"
            decision_copy = "The calibration profile is missing, incompatible, below the provisional 5-pixel sampling guideline, or image-quality checks raised concerns. Follow the calibration and photo guidance before relying on a same/different decision."
        elif difference <= tolerance_mm:
            status = "PRELIMINARY: SAME WITHIN TOLERANCE"
            status_class = "status-green"
            decision_copy = "The estimated median difference is within the selected comparison threshold. This is a preliminary screen, not statistical proof of equivalence."
        else:
            status = "PRELIMINARY: DIFFERENT"
            status_class = "status-red"
            decision_copy = "The estimated median difference exceeds the selected comparison threshold. This is not yet a validated quality-control decision; verify performance against independent physical reference measurements."

        m1, m2, m3 = st.columns(3, gap="medium")
        m1.metric("Sample A median width", display_width(result_a["median"]))
        m2.metric("Sample B median width", display_width(result_b["median"]))
        m3.metric("Absolute estimated difference", display_width(difference))
        st.markdown(f'<div class="result-box"><div class="result-title">{status}</div><p class="result-copy">{decision_copy}</p><div style="margin-top:.75rem"><span class="status-chip {status_class}">{status}</span></div></div>', unsafe_allow_html=True)
        st.caption(f"Segmentation method: {segmentation_method}. " + comparison["warning"] + " The 0.017 mm sensitivity target has not been validated for your camera and yarn types.")

        if rgb_a.shape[:2] != rgb_b.shape[:2]:
            st.warning("The image dimensions differ. This can invalidate the shared calibration if magnification or resizing differs. Capture both images with identical image settings.")

        st.markdown('<div class="section-head"><div><div class="section-title">Detected yarn boundaries</div><div class="section-sub">Green = detected region · coral = sampled centerline. Check that the mask follows the yarn body, not the loose hairs or background.</div></div></div>', unsafe_allow_html=True)
        overlay_view = st.radio(
            "Image review", ["Detected boundaries", "Original photos"],
            horizontal=True, key="boundary_review_mode",
            help="Switch between the detected yarn mask and the untouched photo. Inspect the boundary view before trusting a measurement."
        )
        v1, v2 = st.columns(2, gap="large")
        for col, result in ((v1, result_a), (v2, result_b)):
            with col:
                st.markdown(f'<div class="info-card"><div class="card-kicker">{result["label"]}</div><div class="card-title">{overlay_view}</div><div class="card-copy">{result["n"]:,} local width samples retained · samples are correlated.</div></div>', unsafe_allow_html=True)
                st.image(result["overlay"] if overlay_view == "Detected boundaries" else result["original"], use_container_width=True)
                if result["warnings"]:
                    with st.expander(f"Quality warnings — {result['label']}", expanded=True):
                        for warning in result["warnings"]:
                            st.warning(warning)
                st.caption(f"Foreground: {result['diagnostics']['foreground_fraction']:.3%} · Connected components: {result['diagnostics']['component_count_including_background'] - 1} · Touches image border: {result['diagnostics']['touches_image_border']}")

        st.markdown('<div class="section-head"><div><div class="section-title">Measurement details</div><div class="section-sub">Cross-sections describe within-image variation; they are correlated and do not replace repeated captures or independent specimens.</div></div></div>', unsafe_allow_html=True)
        detail_table = {
            "Measurement": ["Mean width", "Median width", "Standard deviation of local estimates", "Local width samples"],
            "Sample A": [display_width(result_a["mean"]), display_width(result_a["median"]), display_width(result_a["std"]), f"{result_a['n']:,}"],
            "Sample B": [display_width(result_b["mean"]), display_width(result_b["median"]), display_width(result_b["std"]), f"{result_b['n']:,}"],
        }
        st.dataframe(detail_table, hide_index=True, use_container_width=True)

        if st.checkbox("Show width-distribution chart", value=True, key="show_width_distribution"):
            hist_fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
            plot_widths_a = result_a["widths"] if measurement_scale_usable else result_a["widths"] * px_per_mm
            plot_widths_b = result_b["widths"] if measurement_scale_usable else result_b["widths"] * px_per_mm
            ax.hist(plot_widths_a, bins=35, alpha=.75, label="Sample A", color="#8F9F7B")
            ax.hist(plot_widths_b, bins=35, alpha=.68, label="Sample B", color="#C78375")
            ax.set_xlabel("Estimated local width (mm)" if measurement_scale_usable else "Estimated local width (pixels; uncalibrated)")
            ax.set_ylabel("Count")
            ax.legend(frameon=False)
            ax.grid(True, axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(hist_fig, use_container_width=True)
            plt.close(hist_fig)

        report = [
            "Skein — Yarn Diameter Comparator report",
            (f"Calibration scale: {px_per_mm:.4f} pixels/mm (usable for these images)" if measurement_scale_usable else f"Calibration scale: NOT USABLE for physical measurements; {px_per_mm:.4f} px/mm was used only to create pixel previews"),
            f"Saved automatic calibration profile available: {calibration_verified}",
            f"Calibration reference marker side: {calibration_profile.get('reference_side_mm', MARKER_SIDE_MM) if calibration_profile else 'not set'} mm",
            f"Calibration photo dimensions: {calibration_profile.get('image_width_px', 'not set') if calibration_profile else 'not set'} × {calibration_profile.get('image_height_px', 'not set') if calibration_profile else 'not set'} px",
            f"Segmentation backend: {segmentation_method}",
            f"3×3 median filter enabled: {median_filter_enabled}",
            (f"Comparison threshold: {tolerance_mm:.4f} mm" if measurement_scale_usable else f"Comparison threshold setting: {tolerance_mm:.4f} mm (not physically interpretable until calibration is usable)"),
            "Target smallest difference: 0.017 mm (not validated)",
            (f"0.017 mm target at saved image scale: {0.017 * px_per_mm:.4f} pixels" if measurement_scale_usable else "0.017 mm target pixel equivalent: unavailable (no compatible, warning-free calibration profile)"),
            f"Calibration profile created (UTC): {calibration_profile.get('created_at_utc', 'unknown') if calibration_profile else 'not set'}",
            f"Sample A mean: {display_width(result_a['mean'])}",
            f"Sample A median: {display_width(result_a['median'])}",
            f"Sample A SD of local estimates: {display_width(result_a['std'])}",
            f"Sample A local samples: {result_a['n']}",
            f"Sample B mean: {display_width(result_b['mean'])}",
            f"Sample B median: {display_width(result_b['median'])}",
            f"Sample B SD of local estimates: {display_width(result_b['std'])}",
            f"Sample B local samples: {result_b['n']}",
            f"Absolute median difference: {display_width(difference)}",
            f"Preliminary status: {status}",
            "Quality warnings:",
            *[f"- {label}: {issue}" for label, issue in all_issues],
            "Caution: This prototype is not validated for detecting 0.017 mm differences. Validate against an independent calibrated reference on multiple independently collected yarn samples.",
        ]
        st.download_button("Download measurement report (.txt)", data="\n".join(report).encode("utf-8"), file_name="skein_yarn_comparison_report.txt", mime="text/plain")

    except Exception as exc:
        st.error(f"We couldn’t measure one or both photos reliably: {exc}")
        st.markdown('<div class="warning-note"><b>Try again:</b> use a sharp image with one straight strand, visible background on every edge, even lighting and a contrasting matte surface. Confirm yarn/background contrast and calibration settings.</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note"><span><b>SKEIN</b> · YARN MEASUREMENT STUDIO</span><span>Prototype only · Physical accuracy depends on calibration, segmentation, and repeatability validation.</span></div>',
    unsafe_allow_html=True,
)
