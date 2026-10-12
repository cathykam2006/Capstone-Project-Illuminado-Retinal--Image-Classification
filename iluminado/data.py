"""Data loading for RFMiD.

Replaces the old "move files into class folders" approach. Images stay where
they are; a manifest (one row per image: split, path, labels) drives
everything. This keeps the multi-label annotations intact and makes the
feature/label pairing impossible to scramble.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SPLITS = ("train", "val", "test")
LABEL_CSVS = {
    "train": "RFMiD_Training_Labels.csv",
    "val": "RFMiD_Validation_Labels.csv",
    "test": "RFMiD_Testing_Labels.csv",
}
# The three diseases with the most examples, as in the original project.
TARGET_DISEASES = ["DR", "MH", "ODC"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _split_from_path(path: Path) -> str | None:
    """Work out which official split an image belongs to from its folder names.

    Handles the Kaggle layout (Training_Set/.../Training, Evaluation_Set/.../Validation,
    Test_Set/.../Test) and simple layouts (train/, val/, test/).
    """
    parts = " ".join(p.lower() for p in path.parent.parts)
    if re.search(r"\btest", parts):
        return "test"
    if re.search(r"valid|evaluation|\bval\b", parts):
        return "val"
    if re.search(r"train", parts):
        return "train"
    return None


def index_images(image_root: str | os.PathLike) -> pd.DataFrame:
    """Find every image under image_root and tag it with split and numeric ID."""
    rows = []
    for p in Path(image_root).rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS or not p.stem.isdigit():
            continue
        split = _split_from_path(p.relative_to(image_root))
        if split is not None:
            rows.append({"split": split, "ID": int(p.stem), "path": str(p)})
    df = pd.DataFrame(rows, columns=["split", "ID", "path"])
    dupes = df.duplicated(["split", "ID"], keep=False)
    if dupes.any():
        raise ValueError(
            "Same image ID found twice in one split; check image_root:\n"
            + df[dupes].sort_values(["split", "ID"]).head(10).to_string()
        )
    return df


def build_manifest(image_root: str | os.PathLike, label_dir: str | os.PathLike) -> pd.DataFrame:
    """Join the label CSVs to the image files. Fails loudly on missing images."""
    images = index_images(image_root)
    frames = []
    for split, csv in LABEL_CSVS.items():
        labels = pd.read_csv(Path(label_dir) / csv)
        labels.insert(0, "split", split)
        frames.append(labels)
    labels = pd.concat(frames, ignore_index=True)
    manifest = labels.merge(images, on=["split", "ID"], how="left")
    missing = manifest["path"].isna()
    if missing.any():
        counts = manifest[missing].groupby("split").size().to_dict()
        raise FileNotFoundError(
            f"{missing.sum()} labelled images not found under {image_root} "
            f"(by split: {counts}). Check that IMAGE_ROOT points at the RFMiD images."
        )
    return manifest


# ---------------------------------------------------------------- preprocessing

def crop_fundus(img: Image.Image, threshold: int = 12) -> Image.Image:
    """Trim the black border around the circular fundus and pad to a square.

    RFMiD comes from three cameras with different resolutions and border sizes.
    Cropping to the fundus makes every image the same framing, so resizing does
    not waste pixels on black background.
    """
    arr = np.asarray(img.convert("RGB"))
    mask = arr.max(axis=2) > threshold
    if mask.sum() < 0.05 * mask.size:  # nearly all dark: leave as is
        cropped = arr
    else:
        rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
        cropped = arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    h, w = cropped.shape[:2]
    side = max(h, w)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    square[top : top + h, left : left + w] = cropped
    return Image.fromarray(square)


def preprocess_image(img: Image.Image, size: int) -> np.ndarray:
    """The single preprocessing path used for training AND the app.

    Returns float32 pixels in 0-255, shape (size, size, 3). Model-specific
    scaling happens inside each model, so callers never have to remember it.
    """
    img = crop_fundus(img).resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def cache_images(manifest: pd.DataFrame, cache_dir: str | os.PathLike, size: int = 512) -> pd.DataFrame:
    """Crop and shrink every image once (about 40 GB down to under 1 GB).

    Training then reads small JPEGs instead of 2-4 MP PNGs, which is what makes
    Colab training practical. Returns a copy of the manifest whose `path`
    column points at the cached files. Re-running skips files already done.
    """
    cache_dir = Path(cache_dir)
    out = manifest.copy()
    new_paths = []
    for split, image_id, src in manifest[["split", "ID", "path"]].itertuples(index=False):
        dst = cache_dir / split / f"{image_id}.jpg"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                crop_fundus(im).resize((size, size), Image.BILINEAR).save(dst, quality=95)
        new_paths.append(str(dst))
    out["path"] = new_paths
    return out


# ---------------------------------------------------------------- tf.data

def make_dataset(paths, labels, img_size: int, batch_size: int, training: bool, seed: int = 42):
    """Build a tf.data pipeline that keeps each image paired with its label.

    Shuffling happens on (path, label) pairs together, so they can never drift
    apart (the bug in the original notebooks). Augmentation is random on every
    epoch, and only on the training split.
    """
    import tensorflow as tf

    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim == 1:
        labels = labels[:, None]
    ds = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if training:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img.set_shape((img_size, img_size, 3))
        return img, label

    def augment(img, label):
        # Fundus images have no natural "up", so flips and 90-degree turns are safe.
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        img = tf.image.random_brightness(img, 0.1 * 255)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        return tf.clip_by_value(img, 0.0, 255.0), label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
