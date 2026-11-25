"""Compare tflite_numpy.py against TensorFlow Lite on the sample images.

    pip install ai-edge-litert pillow numpy pandas
    python check_numpy_models.py /path/to/main/checkout
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "../Capstone-Project-Illuminado-Retinal--Image-Classification")
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))
from iluminado.data import preprocess_image  # noqa: E402
from tflite_numpy import Model  # noqa: E402

worst = 0.0
for task in ("binary", "multilabel"):
    cfg = json.loads((SRC / "models" / f"{task}_ensemble.json").read_text())
    runner = Interpreter(model_path=str(SRC / "models" / cfg["tflite"])).get_signature_runner()
    ours = Model(HERE / "models_numpy" / f"{task}.json", HERE / "models_numpy" / f"{task}.npz")
    for img_path in sorted((SRC / "app" / "samples").glob("*.jpg")):
        img = preprocess_image(Image.open(img_path).convert("RGB"), cfg["img_size"])[None]
        ref = runner(image=img.astype(np.float32))
        ref_p = next(v for v in ref.values() if v.ndim == 2)[0]
        ref_c = next(v for v in ref.values() if v.ndim == 4)[0]
        t0 = time.time()
        out = ours(img)
        dt = time.time() - t0
        p = next(v for v in out.values() if v.ndim == 2)[0]
        c = next(v for v in out.values() if v.ndim == 4)[0]
        dp = float(np.abs(p - ref_p).max())
        dc = float(np.abs(c - ref_c).max() / (np.abs(ref_c).max() + 1e-8))
        worst = max(worst, dp)
        print(f"{task:10} {img_path.stem:32} tflite={np.round(ref_p, 4)} numpy={np.round(p, 4)} "
              f"max|dp|={dp:.1e} heatmap rel diff={dc:.1e} ({dt:.2f}s)")
print("worst probability difference:", worst)
sys.exit(0 if worst < 1e-4 else 1)
