"""A deliberately simple baseline the CNNs have to beat.

34 interpretable statistics per image (colour distribution, edge strength,
bright/dark pixel shares, brightness by ring, sharpness), fed to three classical
classifiers whose probabilities are averaged. No deep learning involved. If the
CNN ensemble isn't clearly better than this, the extra complexity isn't earning
its keep.
"""
from __future__ import annotations

import warnings

import numpy as np
from PIL import Image


def load_images(paths, size: int) -> np.ndarray:
    out = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    for k, p in enumerate(paths):
        with Image.open(p) as im:
            out[k] = np.asarray(im.convert("RGB").resize((size, size), Image.BOX))
    return out


def handcrafted_features(images: np.ndarray) -> np.ndarray:
    """(n, h, w, 3) uint8 -> (n, 34) features, measured inside the retina (non-black area)."""
    n, h, w, _ = images.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) / (min(h, w) / 2)
    rings = [(0, .25), (.25, .5), (.5, .75), (.75, 1.0)]
    out = []
    for img in images:
        x = img.astype(np.float32)
        mask = (x @ np.array([0.299, 0.587, 0.114], np.float32)) > 12
        if mask.sum() < 50:
            mask = np.ones(mask.shape, bool)
        row = []
        for c in range(3):                                      # colour distribution
            v = x[..., c][mask]
            row += [v.mean(), v.std(), *np.percentile(v, [5, 25, 50, 75, 95])]
        row += [row[0] / (row[7] + 1), row[0] / (row[14] + 1)]  # red/green, red/blue balance
        g = x[..., 1]                                           # green: best vessel/lesion contrast
        gy, gx = np.gradient(g)
        mag = np.hypot(gx, gy)[mask]
        row += [mag.mean(), mag.std(), *np.percentile(mag, [90, 99])]
        gv = g[mask]
        row += [(gv > 200).mean(), (gv < 40).mean()]
        for lo, hi in rings:
            m = mask & (radius >= lo) & (radius < hi)
            row.append(float(g[m].mean()) if m.any() else 0.0)
        lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        row.append(float(lap[mask[1:-1, 1:-1]].var()))          # sharpness (low = blurry/hazy)
        out.append(row)
    return np.asarray(out, dtype=np.float64)


def features_for(paths, size: int = 224, batch: int = 200) -> np.ndarray:
    return np.vstack([handcrafted_features(load_images(paths[i:i + batch], size)) for i in range(0, len(paths), batch)])


def _members(seed: int = 42):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return [
        make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=seed),
    ]


def fit_predict(X_train, Y_train, X_eval_list) -> list[np.ndarray]:
    """Fit the 3-model classical ensemble per label column; return averaged probabilities
    for each array in X_eval_list, shaped (n, n_labels)."""
    Y_train = np.asarray(Y_train)
    Y_train = Y_train[:, None] if Y_train.ndim == 1 else Y_train
    outs = [np.zeros((len(X), Y_train.shape[1])) for X in X_eval_list]
    for j in range(Y_train.shape[1]):
        for est in _members():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X_train, Y_train[:, j])
            for o, X in zip(outs, X_eval_list):
                o[:, j] += est.predict_proba(X)[:, 1] / 3
    return outs
