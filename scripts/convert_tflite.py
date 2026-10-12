"""Convert the demo's Keras models to TensorFlow Lite so the app doesn't need TensorFlow.

TensorFlow alone uses about 640 MB of memory before a model is even loaded, which
is close to the limit of free hosting. The TensorFlow Lite runtime (ai-edge-litert)
needs a small fraction of that.

Each .tflite model returns two things for a 384x384 RGB image with values 0-255:
  probs: (1, n_outputs) disease probabilities, identical to the Keras model
  cam:   (1, h, w, n_outputs) heatmap per output

Why no gradients are needed for the heatmap: the model ends in global average
pooling followed by one dense layer, so the Grad-CAM channel weights are the dense
layer's weights times a positive constant. After normalising, Grad-CAM equals the
class activation map relu(sum_k w_k * A_k), which this graph computes directly.

Usage (from the repo root):  python scripts/convert_tflite.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models"


def convert(keras_path: Path, tflite_path: Path, size: int) -> None:
    import tempfile

    import keras
    import tensorflow as tf
    from iluminado.gradcam import _split_model
    from iluminado.models import load

    model = load(str(keras_path))
    before, backbone, after = _split_model(model)
    gap, head = after[0], after[-1]
    n = head.units

    # A plain functional model with two outputs, reusing the trained layers.
    inp = keras.Input((size, size, 3), name="image")
    x = inp
    for layer in before:
        x = layer(x)
    feats = backbone(x)
    probs = head(gap(feats))
    cam_dense = keras.layers.Dense(n, use_bias=False, name="cam_weights")   # same weights as the head, applied per pixel
    cam = keras.layers.ReLU(name="cam")(cam_dense(feats))
    wrapper = keras.Model(inp, {"probs": probs, "cam": cam})
    cam_dense.set_weights([np.asarray(head.kernel)])

    with tempfile.TemporaryDirectory() as tmp:
        wrapper.export(tmp, format="tf_saved_model", verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
        tflite_path.write_bytes(converter.convert())


def verify(keras_path: Path, tflite_path: Path, images: np.ndarray) -> float:
    """Max |difference| between Keras and TFLite probabilities on real images."""
    from ai_edge_litert.interpreter import Interpreter
    from iluminado.models import load

    model = load(str(keras_path))
    runner = Interpreter(model_path=str(tflite_path)).get_signature_runner()
    worst = 0.0
    for img in images:
        k = np.asarray(model(img[None], training=False))
        out = runner(image=img[None].astype(np.float32))
        t = next(v for name, v in out.items() if v.ndim == 2)          # probs: (1, n)
        cam = next(v for name, v in out.items() if v.ndim == 4)        # cam: (1, h, w, n)
        if not (np.isfinite(t).all() and np.isfinite(cam).all()):
            return float("inf")                                       # NaN/inf output = broken conversion
        worst = max(worst, float(np.abs(k - t).max()))
    return worst


def main():
    from PIL import Image
    from iluminado.data import preprocess_image

    samples = sorted((ROOT / "app" / "samples").glob("*.jpg"))
    for cfg_path in sorted(MODEL_DIR.glob("*_ensemble.json")):
        cfg = json.loads(cfg_path.read_text())
        size, (name, fname) = cfg["img_size"], next(iter(cfg["models"].items()))
        keras_path, tflite_path = MODEL_DIR / fname, MODEL_DIR / (Path(fname).stem + ".tflite")
        convert(keras_path, tflite_path, size)
        imgs = np.stack([preprocess_image(Image.open(p).convert("RGB"), size) for p in samples]) if samples else \
            np.random.default_rng(0).uniform(0, 255, (2, size, size, 3)).astype("float32")
        diff = verify(keras_path, tflite_path, imgs)
        if not diff <= 1e-3:
            raise RuntimeError(f"{tflite_path.name}: TFLite output differs from Keras by {diff:.4f}")
        cfg["tflite"] = tflite_path.name
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"{tflite_path.name}: {tflite_path.stat().st_size / 1e6:.1f} MB, max difference vs Keras {diff:.1e}")


if __name__ == "__main__":
    main()
