"""Evaluation that answers the questions a clinician would actually ask.

Accuracy alone was misleading in the original project: 79% of images are
diseased, so "always say diseased" scores 79%. These helpers report AUC,
sensitivity/specificity at a threshold chosen on the validation set, the
naive baseline for comparison, and bootstrap confidence intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)


def predict(model, ds, tta: bool = True) -> np.ndarray:
    """Predictions in dataset order (the dataset must NOT be shuffled).

    With tta=True the prediction is averaged over the image and its horizontal
    and vertical flips (test-time augmentation): a cheap way to smooth out
    predictions that depend on orientation, which a fundus photo has no
    natural version of.
    """
    import tensorflow as tf

    p = np.asarray(model.predict(ds, verbose=0), dtype=np.float64)
    if not tta:
        return p
    for flip in (tf.image.flip_left_right, tf.image.flip_up_down):
        p += np.asarray(model.predict(ds.map(lambda x, y, f=flip: (f(x), y)), verbose=0), dtype=np.float64)
    return p / 3


def threshold_for_sensitivity(y_true, y_prob, target: float = 0.90) -> float:
    """Highest threshold that still catches `target` of diseased eyes.

    For screening, a missed disease costs more than a false alarm, so we fix
    sensitivity and take the best specificity we can get. Always chosen on the
    validation set, then applied unchanged to test.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= target)[0]
    return float(min(thr[ok[0]], 1.0))


def threshold_for_f1(y_true, y_prob) -> float:
    """Threshold maximising F1 (used per disease in the multi-label task)."""
    grid = np.unique(np.round(y_prob, 4))
    if len(grid) > 400:
        grid = np.quantile(y_prob, np.linspace(0.01, 0.99, 400))
    scores = [f1_score(y_true, y_prob >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def binary_report(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "threshold": threshold,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "PPV": tp / max(tp + fp, 1),
        "NPV": tn / max(tn + fn, 1),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": (tp + tn) / len(y_true),
        "majority_baseline_acc": max(y_true.mean(), 1 - y_true.mean()),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def multilabel_report(y_true, y_prob, thresholds, names) -> pd.DataFrame:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rows = []
    for j, name in enumerate(names):
        t, p = y_true[:, j].astype(int), y_prob[:, j]
        pred = (p >= thresholds[j]).astype(int)
        tn, fp, fn, tp = confusion_matrix(t, pred, labels=[0, 1]).ravel()
        rows.append({
            "disease": name, "n_positive": int(t.sum()),
            "AUC": roc_auc_score(t, p), "PR-AUC": average_precision_score(t, p),
            "prevalence (PR-AUC of random guess)": t.mean(),
            "threshold": thresholds[j], "F1": f1_score(t, pred, zero_division=0),
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        })
    df = pd.DataFrame(rows).set_index("disease")
    df.loc["macro avg"] = df.drop(columns=["threshold"]).mean(numeric_only=True)
    return df


def bootstrap_ci(y_true, y_prob, metric=roc_auc_score, n: int = 1000, seed: int = 42):
    """95% CI by resampling test images. With 640 test images, a 0.02 AUC
    difference between models is often within noise; the CI shows that."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric(y_true[idx], y_prob[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_bootstrap_diff(y_true, p_a, p_b, metric=roc_auc_score, n: int = 1000, seed: int = 42):
    """metric(A) - metric(B), with a 95% CI from resampling the SAME images for both.

    If the interval excludes 0, A is better (or worse) than B by more than
    sampling noise. Used to check whether the ensemble really beats its best
    single member; if not, the simpler single model is just as good.
    """
    rng = np.random.default_rng(seed)
    y_true, p_a, p_b = np.asarray(y_true), np.asarray(p_a), np.asarray(p_b)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt = y_true[idx]
        if (yt.ndim == 1 and len(np.unique(yt)) < 2) or (yt.ndim == 2 and (yt.min(0) == yt.max(0)).any()):
            continue
        diffs.append(metric(yt, p_a[idx]) - metric(yt, p_b[idx]))
    d = float(metric(y_true, p_a) - metric(y_true, p_b))
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    verdict = "A clearly better" if lo > 0 else "A clearly worse" if hi < 0 else "no clear difference"
    return d, lo, hi, verdict


def ensemble(probs: dict, weights: dict | None = None) -> np.ndarray:
    """Average the probabilities of several models (optionally weighted)."""
    names = list(probs)
    w = np.array([1.0 if weights is None else weights[k] for k in names])
    stacked = np.stack([probs[k] for k in names])
    return np.tensordot(w / w.sum(), stacked, axes=1)
