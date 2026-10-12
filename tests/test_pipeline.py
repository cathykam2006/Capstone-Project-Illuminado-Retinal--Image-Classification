"""Smoke tests on synthetic images: no dataset, GPU or pretrained weights needed.

Run:  python -m pytest tests -q     (or: python tests/test_pipeline.py)

Checks the things that went wrong in the original version:
- images and labels stay paired through shuffling
- every labelled image is found, in the right split
- models train, save, reload and give identical predictions
- evaluation and Grad-CAM run
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iluminado import data as D, evaluate as E, models as M  # noqa: E402
from iluminado.gradcam import gradcam, overlay  # noqa: E402

N = {"train": 12, "val": 6, "test": 6}
KAGGLE_DIRS = {"train": "Training_Set/Training_Set/Training",
               "val": "Evaluation_Set/Evaluation_Set/Validation",
               "test": "Test_Set/Test_Set/Test"}


def fake_fundus(brightness: int, w: int = 300, h: int = 200) -> Image.Image:
    """Orange disc on a black background, like a fundus photo."""
    img = Image.new("RGB", (w, h))
    ImageDraw.Draw(img).ellipse([60, 10, 240, 190], fill=(brightness, brightness // 2, 20))
    return img


def make_fake_rfmid(root: Path) -> tuple[Path, Path]:
    img_root, label_dir = root / "images", root / "labels"
    label_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for split, n in N.items():
        folder = img_root / KAGGLE_DIRS[split]
        folder.mkdir(parents=True)
        rows = []
        for i in range(1, n + 1):
            risk = int(i % 2 == 0)
            fake_fundus(200 if risk else 90).save(folder / f"{i}.png")
            row = {"ID": i, "Disease_Risk": risk}
            row.update({d: int(risk and rng.random() < 0.5) for d in D.TARGET_DISEASES})
            row["ARMD"] = 0
            rows.append(row)
        pd.DataFrame(rows).to_csv(label_dir / D.LABEL_CSVS[split], index=False)
    return img_root, label_dir


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_root, label_dir = make_fake_rfmid(tmp)

        manifest = D.build_manifest(img_root, label_dir)
        assert manifest["split"].value_counts().to_dict() == N
        assert manifest["path"].notna().all()

        cached = D.cache_images(manifest, tmp / "cache", size=64)
        assert all(Path(p).exists() for p in cached["path"])
        with Image.open(cached["path"].iloc[0]) as im:
            assert im.size == (64, 64)

        # Crop removes the black border: the disc should fill most of the frame.
        arr = D.preprocess_image(fake_fundus(200), 64)
        assert arr.shape == (64, 64, 3) and arr.dtype == np.float32
        assert (arr.max(axis=2) > 12).mean() > 0.7

        # Pairing survives shuffling: bright images are exactly the label-1 images.
        tr = cached[cached.split == "train"]
        ds = D.make_dataset(tr.path, tr.Disease_Risk, 64, 4, training=True)
        for _ in range(2):
            for x, y in ds:
                centre_red = x.numpy()[:, 28:36, 28:36, 0].mean(axis=(1, 2))
                assert ((centre_red > 145) == (y.numpy()[:, 0] == 1)).all()

        for n_out, cols in ((1, ["Disease_Risk"]), (3, D.TARGET_DISEASES)):
            split = {s: cached[cached.split == s] for s in N}
            y = {s: split[s][cols].to_numpy("float32") for s in split}
            train_ds = D.make_dataset(split["train"].path, y["train"], 64, 4, training=True)
            val_ds = D.make_dataset(split["val"].path, y["val"], 64, 4, training=False)
            model = M.build_model("efficientnet_b0", 64, n_out, weights=None)
            ckpt = str(tmp / f"m{n_out}.keras")
            M.train(model, train_ds, val_ds, M.positive_weights(y["train"]), n_out, ckpt,
                    head_epochs=1, finetune_epochs=1)
            reloaded = M.load(ckpt)
            p1, p2 = E.predict(reloaded, val_ds), E.predict(reloaded, val_ds)
            assert p1.shape == (N["val"], n_out) and np.allclose(p1, p2)

            img = D.preprocess_image(fake_fundus(200), 64)
            cam = gradcam(reloaded, img, output_index=n_out - 1)
            assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1.0 + 1e-6
            assert overlay(img, cam).dtype == np.uint8


def test_evaluation_helpers():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.4 + rng.random(400) * 0.6, 0, 1)
    t = E.threshold_for_sensitivity(y, p, 0.9)
    rep = E.binary_report(y, p, t)
    assert rep["sensitivity"] >= 0.9 and 0.5 < rep["AUC"] <= 1
    lo, hi = E.bootstrap_ci(y, p, n=200)
    assert lo < rep["AUC"] < hi
    Y = rng.integers(0, 2, (400, 3)); P = np.clip(Y * 0.5 + rng.random((400, 3)) * 0.5, 0, 1)
    ths = [E.threshold_for_f1(Y[:, j], P[:, j]) for j in range(3)]
    df = E.multilabel_report(Y, P, ths, D.TARGET_DISEASES)
    assert list(df.index) == D.TARGET_DISEASES + ["macro avg"]
    ens = E.ensemble({"a": P, "b": 1 - P}, {"a": 3.0, "b": 1.0})
    assert np.allclose(ens, 0.75 * P + 0.25 * (1 - P))
    assert np.allclose(M.positive_weights(np.array([1, 1, 1, 0])), [1 / 3])


def test_huggingface_export(monkeypatch=None):
    """fetch_hf on a fake in-memory mirror: images exported, labels parsed, sizes recorded."""
    import datasets
    from iluminado import download as DL

    def fake_split(n, offset):
        risk = [i % 2 for i in range(n)]
        labels = ["DR, ODC" if r and i % 4 == 1 else ("MH" if r else "") for i, r in enumerate(risk)]
        imgs = [fake_fundus(200 if r else 90, 600, 400) for r in risk]
        return datasets.Dataset.from_dict({"image": imgs, "Disease_Risk": risk, "label": labels},
                                          features=datasets.Features({"image": datasets.Image(),
                                                                      "Disease_Risk": datasets.Value("int64"),
                                                                      "label": datasets.Value("string")}))
    fake = datasets.DatasetDict({"train": fake_split(8, 0), "validation": fake_split(4, 0), "test": fake_split(4, 0)})
    orig = datasets.load_dataset
    datasets.load_dataset = lambda *a, **k: fake
    try:
        with tempfile.TemporaryDirectory() as tmp:
            img_root, label_dir = DL.fetch_hf(tmp, max_side=300, log_every=0)
            DL.fetch_hf(tmp, max_side=300, log_every=0)          # second run: skips, keeps sizes
            manifest = D.build_manifest(img_root, label_dir)
            assert manifest["split"].value_counts().to_dict() == {"train": 8, "val": 4, "test": 4}
            row = manifest[(manifest.split == "train") & (manifest.ID == 2)].iloc[0]
            assert row.DR == 1 and row.ODC == 1 and row.MH == 0 and row.Disease_Risk == 1
            assert manifest.loc[manifest.Disease_Risk == 0, D.TARGET_DISEASES].to_numpy().sum() == 0
            info = pd.read_csv(Path(label_dir) / "image_info.csv")
            assert len(info) == 16 and (info.orig_w == 600).all()
            with Image.open(manifest.path.iloc[0]) as im:
                assert max(im.size) == 300
    finally:
        datasets.load_dataset = orig
    try:
        DL.label_frame([0], ["DR"])
        raise AssertionError("healthy image with a disease label should be rejected")
    except ValueError:
        pass


def test_audits_and_baseline():
    from iluminado import audit as A, baseline as B
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = []
        for i, seed in enumerate([1, 2, 2, 3]):                          # images 1 and 2 are identical
            arr = np.random.default_rng(seed).integers(0, 255, (60, 60, 3), dtype=np.uint8)
            p = tmp / f"{i}.png"; Image.fromarray(arr).resize((240, 240)).save(p); paths.append(str(p))
        m = pd.DataFrame({"split": ["train", "train", "test", "val"], "ID": [1, 2, 3, 4], "path": paths})
        rep = A.duplicate_report(m)
        assert len(rep) == 1 and rep.iloc[0]["crosses splits"]           # images 1 and 2 are identical
        assert A.camera_group(2144, 1424) == "2144x1424" and A.camera_group(1024, 768) == "4:3 camera"
        summary, p, v = A.camera_vs_label(pd.Series(["a"] * 50 + ["b"] * 50), pd.Series([1] * 50 + [0] * 50))
        assert v > 0.9
        X = B.features_for(paths, size=64)
        assert X.shape == (4, 34) and np.isfinite(X).all()
    rng = np.random.default_rng(0)
    Xtr, ytr = rng.normal(size=(60, 34)), np.r_[np.zeros(30), np.ones(30)]
    Xtr[ytr == 1, 0] += 3
    (pe,) = B.fit_predict(Xtr, ytr, [Xtr])
    assert pe.shape == (60, 1) and E.binary_report(ytr, pe[:, 0], 0.5)["AUC"] > 0.9
    d, lo, hi, verdict = E.paired_bootstrap_diff(ytr, pe[:, 0], rng.random(60), n=200)
    assert d > 0 and lo > 0 and verdict == "A clearly better"


if __name__ == "__main__":
    test_evaluation_helpers()
    test_huggingface_export()
    test_audits_and_baseline()
    test_end_to_end()
    print("All tests passed.")
