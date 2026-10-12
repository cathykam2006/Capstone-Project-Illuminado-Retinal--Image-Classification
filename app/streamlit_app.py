"""Iluminado: retinal fundus screening demo (Streamlit).

Run from the repo root:  streamlit run app/streamlit_app.py

Expects trained models plus the ensemble JSON files written by notebooks 03 and
04 in the models/ folder. Any model listed in a JSON but missing from models/
is skipped and the remaining ensemble weights are renormalised, so you can
deploy a subset (for example, leave out a model too large for GitHub).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Keep TensorFlow lean on small cloud servers (fewer threads = less memory, no thread oversubscription)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models"
SAMPLE_DIR = Path(__file__).resolve().parent / "samples"

from iluminado.data import preprocess_image  # noqa: E402  (same preprocessing as training)

DISEASE_NAMES = {"DR": "Diabetic retinopathy", "MH": "Media haze", "ODC": "Optic disc cupping"}

st.set_page_config(page_title="Iluminado - Retinal Disease Screening", page_icon="👁️", layout="centered")


@st.cache_resource(show_spinner="Loading models…")
def load_ensemble(task: str):
    """Return (config, {name: model}) or (None, {}) if this task isn't deployed."""
    cfg_path = MODEL_DIR / f"{task}_ensemble.json"
    if not cfg_path.exists():
        return None, {}
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass   # already initialised by the other task's load
    from iluminado.models import load

    cfg = json.loads(cfg_path.read_text())
    models = {name: load(str(MODEL_DIR / fname))
              for name, fname in cfg["models"].items() if (MODEL_DIR / fname).exists()}
    return cfg, models


def ensemble_predict(cfg, models, img: np.ndarray) -> np.ndarray:
    w = np.array([cfg["weights"][n] for n in models])
    # Calling the model directly is lighter than model.predict() for a single image
    preds = np.stack([np.asarray(m(img[None], training=False))[0] for m in models.values()])
    return (w / w.sum()) @ preds


def best_model(cfg, models):
    name = max(models, key=lambda n: cfg["weights"][n])
    return name, models[name]


# ------------------------------------------------------------------- page
st.title("Iluminado - Retinal Disease Screening")
st.markdown("<p style='font-size:1.05rem; color:gray; margin-top:-0.8rem;'>Created by Cathy Kam</p>", unsafe_allow_html=True)

st.markdown("""
#### What this dashboard is for
Iluminado reads a **retinal fundus photograph** (a picture of the back of the eye) and estimates:
- the **probability that the eye shows signs of retinal disease**, and
- whether it shows signs of three specific conditions: **diabetic retinopathy**, **media haze** and **optic disc cupping**.

A heatmap shows **which part of the retina drove the prediction**, so the result can be checked rather than taken on trust.

#### What I want to achieve
The WHO estimates that at least 1 billion people live with vision impairment that could have been prevented or has yet to be addressed, often because there are too few eye specialists. Fundus photos can be taken by trained non-specialists with portable cameras. My goal is a **low-cost first screening step** that helps decide **who should see an ophthalmologist first**, so scarce specialist time goes to the people who need it most. I also wanted it to be **honest about its limits**: every number below comes from images the model never saw during training.
""")

bin_cfg, bin_models = load_ensemble("binary")
ml_cfg, ml_models = load_ensemble("multilabel")

if not bin_models and not ml_models:
    st.error(
        "No trained models found in `models/`. Train them with notebooks 03 and 04, then copy the "
        "`.keras` files and `*_ensemble.json` files into `models/`."
    )
    st.stop()

with st.sidebar:
    st.header("About the models")
    if bin_cfg:
        st.markdown(
            f"**Disease risk:** {len(bin_models)} model(s): {', '.join(bin_models)}. "
            f"Test AUC {bin_cfg['test_auc']:.3f}. Threshold set for {bin_cfg['threshold_rule']}."
        )
    if ml_cfg:
        st.markdown(
            f"**Disease type (DR / MH / ODC):** {len(ml_models)} model(s): {', '.join(ml_models)}. "
            f"Test macro AUC {ml_cfg['test_macro_auc']:.3f}."
        )
    st.markdown("Data: [RFMiD](https://www.mdpi.com/2306-5729/6/2/14), three fundus cameras, 2009-2020.")
    st.header("Privacy")
    st.markdown("Uploaded images are processed in memory only and are **not saved**.")

samples = sorted(SAMPLE_DIR.glob("*.jpg")) if SAMPLE_DIR.exists() else []
source = st.radio("Image", ["Try a sample image", "Upload my own"] if samples else ["Upload my own"], horizontal=True)

if source == "Try a sample image":
    label = lambda p: p.stem.rsplit("_test", 1)[0].replace("_", " ").capitalize() + f" (test image {p.stem.rsplit('_test', 1)[-1]})"
    choice = st.selectbox("Sample from the RFMiD test set (not used in training)", samples, format_func=label)
    raw = Image.open(choice).convert("RGB")
    st.caption("Sample images: RFMiD test set (Pachade et al., 2021). The label in the name is the experts' diagnosis.")
else:
    uploaded = st.file_uploader("Upload a retinal fundus photograph", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is None:
        st.info("Fundus photos are taken with a fundus camera and show the back of the eye (retina, optic disc and blood vessels). "
                "Other kinds of photos will still get a score, but it will be meaningless.")
        st.stop()
    try:
        raw = Image.open(uploaded).convert("RGB")
    except UnidentifiedImageError:
        st.error("That file couldn't be read as an image.")
        st.stop()

size = (bin_cfg or ml_cfg)["img_size"]
img = preprocess_image(raw, size)
show_cam = st.toggle("Show where the model looked (Grad-CAM)", value=True)

col1, col2 = st.columns(2)
col1.image(img.astype("uint8"), caption="Cropped and resized input", width="stretch")

if bin_models:
    p = float(ensemble_predict(bin_cfg, bin_models, img)[0])
    refer = p >= bin_cfg["threshold"]
    st.subheader("Disease risk")
    st.metric("Estimated probability of retinal disease", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    if refer:
        st.error(f"**Above the screening threshold ({bin_cfg['threshold']:.0%}).** In a screening setting this image would be flagged for review by an ophthalmologist.")
    else:
        st.success(f"**Below the screening threshold ({bin_cfg['threshold']:.0%}).** No disease pattern detected by the model. This does not rule out eye disease.")
    if show_cam:
        from iluminado.gradcam import gradcam, overlay
        name, model = best_model(bin_cfg, bin_models)
        col2.image(overlay(img, gradcam(model, img)), caption=f"Grad-CAM ({name}): red = most influential", width="stretch")

if ml_models:
    probs = ensemble_predict(ml_cfg, ml_models, img)
    st.subheader("Specific conditions")
    st.caption("Each condition is assessed independently; an eye can have more than one.")
    for j, (label, pj) in enumerate(zip(ml_cfg["labels"], probs)):
        flag = "⚑ above threshold" if pj >= ml_cfg["thresholds"][j] else ""
        st.markdown(f"**{DISEASE_NAMES.get(label, label)} ({label})**: {pj:.0%} {flag}")
        st.progress(float(min(max(pj, 0.0), 1.0)))
    st.caption("The model only checks these three conditions; RFMiD labels 45, so other diseases are not assessed here.")

st.divider()
st.caption("Research prototype trained on 3,200 public images (RFMiD); not clinically validated or a medical device.")
