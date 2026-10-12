"""Iluminado - Retinal Disease Screening (Streamlit demo).

Run from the repo root:  streamlit run app/streamlit_app.py

Runs the trained models with the lightweight TensorFlow Lite runtime (ai-edge-litert)
instead of full TensorFlow, which keeps memory well inside free-hosting limits.
The .tflite files are made from the trained Keras models by scripts/convert_tflite.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models"
SAMPLE_DIR = Path(__file__).resolve().parent / "samples"

from iluminado.data import preprocess_image  # noqa: E402  (same preprocessing as training)

MAX_UPLOAD_MB = 20            # also set in .streamlit/config.toml (server.maxUploadSize)
MAX_SIDE_PX = 2048            # larger photos are shrunk first; the model only sees 384 x 384
Image.MAX_IMAGE_PIXELS = 50_000_000   # refuse absurdly large images (about 7000 x 7000)
DISEASE_NAMES = {"DR": "Diabetic retinopathy", "MH": "Media haze", "ODC": "Optic disc cupping"}

st.set_page_config(page_title="Iluminado - Retinal Disease Screening", page_icon="👁️", layout="centered")


@st.cache_resource(show_spinner="Loading models…")
def load_model(task: str):
    """Return (config, TFLite signature runner) or (None, None) if this task isn't deployed."""
    cfg_path = MODEL_DIR / f"{task}_ensemble.json"
    if not cfg_path.exists():
        return None, None
    cfg = json.loads(cfg_path.read_text())
    tflite = MODEL_DIR / cfg.get("tflite", "")
    if not cfg.get("tflite") or not tflite.exists():
        return None, None
    from ai_edge_litert.interpreter import Interpreter

    runner = Interpreter(model_path=str(tflite), num_threads=2).get_signature_runner()
    return cfg, runner


def run(runner, img: np.ndarray):
    """Probabilities (n,) and heatmaps (h, w, n) for one 0-255 image."""
    out = runner(image=img[None].astype(np.float32))
    probs = next(v for v in out.values() if v.ndim == 2)[0]
    cams = next(v for v in out.values() if v.ndim == 4)[0]
    return probs, cams


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a heatmap (red = most influential) onto the image."""
    import matplotlib

    cam = np.asarray(Image.fromarray(cam.astype(np.float32)).resize(img.shape[1::-1], Image.BILINEAR))
    cam = cam / (cam.max() + 1e-8)
    heat = matplotlib.colormaps["jet"](cam)[..., :3] * 255
    return np.clip((1 - alpha) * img + alpha * heat, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------- page
st.markdown(
    "<h1 style='font-size:clamp(1.15rem, 4.6vw, 2rem); white-space:nowrap; margin-bottom:0; padding-bottom:0;'>"
    "Iluminado - Retinal Disease Screening</h1>"
    "<p style='font-size:1rem; color:gray; margin-top:0.2rem;'>Created by Cathy Kam</p>",
    unsafe_allow_html=True,
)

st.markdown("""
##### What this dashboard is for
Iluminado reads a **retinal fundus photograph** (a picture of the back of the eye) and estimates:
- the **probability that the eye shows signs of retinal disease**, and
- whether it shows signs of three specific conditions: **diabetic retinopathy**, **media haze** and **optic disc cupping**.

A heatmap shows **which part of the retina drove the prediction**, so the result can be checked rather than taken on trust.

##### What I want to achieve
The WHO estimates that at least 1 billion people live with vision impairment that could have been prevented or has yet to be addressed, often because there are too few eye specialists. Fundus photos can be taken by trained non-specialists with portable cameras. My goal is a **low-cost first screening step** that helps decide **who should see an ophthalmologist first**, so scarce specialist time goes to the people who need it most. I also wanted it to be **honest about its limits**: every number below comes from images the model never saw during training.
""")

bin_cfg, bin_model = load_model("binary")
ml_cfg, ml_model = load_model("multilabel")

if bin_model is None and ml_model is None:
    st.error("No models found in `models/`. Run `python scripts/convert_tflite.py` after exporting the trained models.")
    st.stop()

with st.sidebar:
    st.header("About the models")
    if bin_cfg:
        st.markdown(f"**Disease risk:** EfficientNet-B0, test AUC {bin_cfg['test_auc']:.3f}. "
                    f"Threshold set for {bin_cfg['threshold_rule']}.")
    if ml_cfg:
        st.markdown(f"**Disease type (DR / MH / ODC):** EfficientNet-B0, test macro AUC {ml_cfg['test_macro_auc']:.3f}.")
    st.markdown("Data: [RFMiD](https://www.mdpi.com/2306-5729/6/2/14), three fundus cameras, 2009-2020.")
    st.header("Uploads")
    st.markdown(f"- **One image at a time**, up to **{MAX_UPLOAD_MB} MB**\n"
                "- Formats: JPG, PNG or TIFF\n"
                f"- Larger photos are shrunk to {MAX_SIDE_PX} px on the longest side before analysis\n"
                "- Images are processed in memory only and are **not saved**")

samples = sorted(SAMPLE_DIR.glob("*.jpg")) if SAMPLE_DIR.exists() else []
source = st.radio("Image", ["Try a sample image", "Upload my own"] if samples else ["Upload my own"], horizontal=True)

if source == "Try a sample image":
    label = lambda p: p.stem.rsplit("_test", 1)[0].replace("_", " ").capitalize() + f" (test image {p.stem.rsplit('_test', 1)[-1]})"
    choice = st.selectbox("Sample from the RFMiD test set (not used in training)", samples, format_func=label)
    raw = Image.open(choice).convert("RGB")
    st.caption("Sample images: RFMiD test set (Pachade et al., 2021). The label in the name is the experts' diagnosis.")
else:
    st.info(f"**Upload limit: one image, up to {MAX_UPLOAD_MB} MB** (JPG, PNG or TIFF). "
            "Please upload a retinal fundus photograph; other kinds of photos will still get a score, but it will be meaningless.")
    uploaded = st.file_uploader(f"Upload a retinal fundus photograph (max {MAX_UPLOAD_MB} MB)",
                                type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=False)
    if uploaded is None:
        st.stop()
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"That file is {uploaded.size / 1024 / 1024:.1f} MB. The limit is {MAX_UPLOAD_MB} MB; please upload a smaller image.")
        st.stop()
    try:
        raw = Image.open(uploaded)
        raw.draft("RGB", (MAX_SIDE_PX, MAX_SIDE_PX))      # fast, low-memory decoding for large JPEGs
        raw = raw.convert("RGB")
    except Image.DecompressionBombError:
        st.error("That image has too many pixels to process safely. Please upload a smaller version (under about 7000 x 7000 px).")
        st.stop()
    except (UnidentifiedImageError, OSError):
        st.error("That file couldn't be read as an image. Please upload a JPG, PNG or TIFF.")
        st.stop()

raw.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
size = (bin_cfg or ml_cfg)["img_size"]
img = preprocess_image(raw, size)
show_cam = st.toggle("Show where the model looked (heatmap)", value=True)

col1, col2 = st.columns(2)
col1.image(img.astype("uint8"), caption="Cropped and resized input", width="stretch")

if bin_model is not None:
    probs, cams = run(bin_model, img)
    p = float(probs[0])
    st.subheader("Disease risk")
    st.metric("Estimated probability of retinal disease", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    if p >= bin_cfg["threshold"]:
        st.error(f"**Above the screening threshold ({bin_cfg['threshold']:.0%}).** In a screening setting this image would be flagged for review by an ophthalmologist.")
    else:
        st.success(f"**Below the screening threshold ({bin_cfg['threshold']:.0%}).** No disease pattern detected by the model. This does not rule out eye disease.")
    if show_cam:
        col2.image(overlay(img, cams[..., 0]), caption="Heatmap: red = most influential for the disease-risk score", width="stretch")

if ml_model is not None:
    probs, _ = run(ml_model, img)
    st.subheader("Specific conditions")
    st.caption("Each condition is assessed independently; an eye can have more than one.")
    for j, (label, pj) in enumerate(zip(ml_cfg["labels"], probs)):
        flag = "⚑ above threshold" if pj >= ml_cfg["thresholds"][j] else ""
        st.markdown(f"**{DISEASE_NAMES.get(label, label)} ({label})**: {pj:.0%} {flag}")
        st.progress(float(min(max(pj, 0.0), 1.0)))
    st.caption("The model only checks these three conditions; RFMiD labels 45, so other diseases are not assessed here.")

st.divider()
st.caption("Research prototype trained on 3,200 public images (RFMiD); not clinically validated or a medical device.")
