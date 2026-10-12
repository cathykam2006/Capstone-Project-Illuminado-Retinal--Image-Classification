"""Grad-CAM heatmaps: which part of the retina drove the prediction.

For a medical screening tool this is the main sanity check that the model is
looking at lesions, the optic disc or vessels, and not at camera borders,
text overlays or lighting artefacts.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def _split_model(model):
    names = [layer.name for layer in model.layers]
    idx = names.index("backbone")
    before = model.layers[1:idx]       # preprocessing layers (if any)
    backbone = model.layers[idx]
    after = model.layers[idx + 1:]     # gap, dropout, head
    return before, backbone, after


def gradcam(model, image_0_255: np.ndarray, output_index: int = 0) -> np.ndarray:
    """Heatmap in [0, 1], same height/width as the input image.

    image_0_255: one preprocessed image, shape (H, W, 3), values 0-255.
    output_index: which sigmoid output to explain (0 for disease risk;
    0/1/2 for DR/MH/ODC in the multi-label model).
    """
    before, backbone, after = _split_model(model)
    x = tf.convert_to_tensor(image_0_255[None], dtype=tf.float32)
    for layer in before:
        x = layer(x)
    with tf.GradientTape() as tape:
        feats = backbone(x, training=False)
        tape.watch(feats)
        y = feats
        for layer in after:
            y = layer(y, training=False)
        score = y[:, output_index]
    grads = tape.gradient(score, feats)[0]          # (h, w, c)
    weights = tf.reduce_mean(grads, axis=(0, 1))    # channel importance
    cam = tf.nn.relu(tf.reduce_sum(feats[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], image_0_255.shape[:2])[..., 0]
    return cam.numpy()


def overlay(image_0_255: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend the heatmap onto the image; returns uint8 RGB."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3] * 255
    out = (1 - alpha) * image_0_255 + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)
