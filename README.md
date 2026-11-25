# Browser version of the demo (GitHub Pages)

This branch holds an always-on copy of the Iluminado demo that runs entirely in
the visitor's browser with [stlite](https://github.com/whitphx/stlite). There is
no server, so it never goes to sleep, and uploaded photos never leave the
visitor's device. The Streamlit Community Cloud app keeps running from `main`.

TensorFlow Lite can't run in a browser, so the two models run with
`tflite_numpy.py`, a small numpy runner for the operators these EfficientNet-B0
models use. `export_numpy_models.py` turns each `.tflite` file into a graph
(`models_numpy/*.json`) and its weights (`models_numpy/*.npz`).

Checked against TensorFlow Lite on the four sample images: probabilities agree
to within 0.000003 and heatmaps to within 0.0004%. The app's screening results,
metrics and messages are identical to the Streamlit version.

## After changing the app or the models on main

```bash
pip install tflite numpy ai-edge-litert pillow pandas
python build_static.py /path/to/main/checkout
python check_numpy_models.py /path/to/main/checkout
git add -A && git commit -m "Rebuild browser copy" && git push
```
