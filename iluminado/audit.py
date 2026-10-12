"""Data audits: near-duplicate images across splits, and camera-vs-label shortcuts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: str, hash_size: int = 16) -> np.ndarray:
    """256-bit difference hash: near-identical images give near-identical hashes."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((hash_size + 1, hash_size), Image.BOX)).astype(int)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def near_duplicate_pairs(paths, max_hamming: int = 6, chunk: int = 256) -> list[tuple[int, int, int]]:
    """All pairs (i, j, distance), i < j, whose hashes differ in at most max_hamming of 256 bits."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as pool:
        H = np.stack(list(pool.map(dhash, paths)))
    pairs = []
    for a in range(0, len(H), chunk):
        d = _POPCOUNT[H[a:a + chunk, None, :] ^ H[None, :, :]].sum(-1)
        for ii, jj in zip(*np.where(d <= max_hamming)):
            i, j = a + int(ii), int(jj)
            if i < j:
                pairs.append((i, j, int(d[ii, jj])))
    return pairs


def duplicate_report(manifest: pd.DataFrame, max_hamming: int = 6) -> pd.DataFrame:
    """Near-duplicate pairs with their splits. Pairs that cross splits would leak
    training images into validation or test and inflate the scores."""
    m = manifest.reset_index(drop=True)
    rows = [{"split_a": m.split[i], "ID_a": m.ID[i], "split_b": m.split[j], "ID_b": m.ID[j], "distance": d,
             "crosses splits": m.split[i] != m.split[j]}
            for i, j, d in near_duplicate_pairs(m.path.tolist(), max_hamming)]
    return pd.DataFrame(rows, columns=["split_a", "ID_a", "split_b", "ID_b", "distance", "crosses splits"])


def camera_group(w: int, h: int) -> str:
    """Camera proxy from the original image shape.

    RFMiD's three cameras produce 2144x1424 and 4288x2848 (both 3:2) and
    2048x1536 (4:3) images. If only downscaled copies exist, the two 3:2
    cameras can't be told apart, so the aspect ratio is used as a fallback.
    """
    if (w, h) in {(2144, 1424), (4288, 2848), (2048, 1536)}:
        return f"{w}x{h}"
    return "3:2 camera" if abs(w / h - 1.5) < 0.05 else "4:3 camera" if abs(w / h - 4 / 3) < 0.05 else f"other ({w}x{h})"


def camera_vs_label(cameras: pd.Series, labels: pd.Series) -> tuple[pd.DataFrame, float, float]:
    """Disease rate per camera, chi-square p-value and Cramer's V.

    Cramer's V near 0: the camera says little about the label (good).
    A large V means a model could score well just by recognising the camera.
    """
    from scipy.stats import chi2_contingency

    table = pd.DataFrame({"camera": cameras.values, "y": labels.values})
    summary = table.groupby("camera")["y"].agg(images="size", disease_rate="mean").round(3)
    ct = pd.crosstab(table.camera, table.y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return summary, float("nan"), float("nan")
    chi2, p, *_ = chi2_contingency(ct)
    v = float(np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1))))
    return summary, float(p), v
