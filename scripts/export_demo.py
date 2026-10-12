"""Package a small, deployable version of the trained models for the Streamlit demo.

Run in Colab after notebooks 03 and 04 (see README, "Deploy the demo"):

    python scripts/export_demo.py --project-dir /content/drive/MyDrive/iluminado --out /content/iluminado-demo.zip

What it does
1. Takes the EfficientNet-B0 model from each task (about 17 MB each, small enough for
   GitHub; the full three-model ensemble is not).
2. Rebuilds each model in float32 (training used mixed precision for GPU speed) and saves it
   without optimizer state, so it loads quickly on a CPU-only server.
3. Recomputes thresholds for this single model from its saved VALIDATION predictions
   (the ensemble's thresholds don't transfer to a single model), and its TEST metrics.
4. Adds a few test-set sample images so visitors can try the app without their own photos.
5. Writes app/requirements.txt pinned to the TensorFlow/Keras versions that saved the models.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BACKBONE = "efficientnet_b0"
TASKS = {"binary": ["Disease_Risk"], "multilabel": ["DR", "MH", "ODC"]}
SAMPLES = [("healthy", lambda d: d.Disease_Risk == 0),
           ("diabetic_retinopathy", lambda d: (d.DR == 1) & (d.MH == 0) & (d.ODC == 0)),
           ("media_haze", lambda d: (d.MH == 1) & (d.DR == 0) & (d.ODC == 0)),
           ("optic_disc_cupping", lambda d: (d.ODC == 1) & (d.DR == 0) & (d.MH == 0))]


def flat_layers(model):
    import keras
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            yield from flat_layers(layer)
        else:
            yield layer


def real_val_images(manifest: pd.DataFrame, size: int, n: int = 48):
    """Up to n validation images, from Drive or from a local cache made earlier in this session."""
    from PIL import Image
    imgs = []
    for path, image_id in manifest.loc[manifest.split == "val", ["path", "ID"]].itertuples(index=False):
        for cand in (path, f"/content/local_cache_512/val/{image_id}.jpg", f"/content/local_cache/val/{Path(path).name}"):
            if Path(cand).exists():
                with Image.open(cand) as im:
                    imgs.append(np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype="float32"))
                break
        if len(imgs) >= n:
            break
    return np.stack(imgs) if imgs else None


def export_models(project: Path, out: Path) -> dict:
    import keras
    from sklearn.metrics import roc_auc_score
    from iluminado import evaluate as E, models as M

    keras.mixed_precision.set_global_policy("float32")
    manifest = pd.read_csv(project / "manifest.csv")
    summary = {}
    for task, labels in TASKS.items():
        cfg_in = json.loads((project / "models" / f"{task}_ensemble.json").read_text())
        size = int(cfg_in["img_size"])
        trained = M.load(str(project / "models" / f"{task}_{BACKBONE}.keras"))
        # Build with weights="imagenet": Keras's EfficientNet adds an extra input-scaling
        # layer only in that mode, so the graph must be built the same way as in training.
        # The downloaded ImageNet weights are then overwritten by the trained ones.
        clean = M.build_model(BACKBONE, size, len(labels), weights="imagenet")
        if [type(l).__name__ for l in flat_layers(trained)] != [type(l).__name__ for l in flat_layers(clean)]:
            raise RuntimeError(f"{task}: rebuilt model has a different structure from the trained model")
        clean.set_weights(trained.get_weights())
        x = real_val_images(manifest, size)
        if x is None:
            raise RuntimeError("No validation images found to verify the exported model; run notebook 03 or 04 in this session first.")
        a_, b_ = trained.predict(x, verbose=0).astype("float32"), clean.predict(x, verbose=0)
        mean_diff, max_diff = float(np.abs(a_ - b_).mean()), float(np.abs(a_ - b_).max())
        print(f"{task}: float32 copy vs trained model on {len(x)} real validation images: "
              f"mean difference {mean_diff:.4f}, max {max_diff:.4f}")
        # Half vs full precision should agree to about 0.01 on real images; a real
        # weight-mapping error gives differences of 0.3 or more.
        if mean_diff > 0.02 or max_diff > 0.1:
            raise RuntimeError(f"{task}: the float32 copy does not reproduce the trained model; not exporting it.")
        fname = f"{task}_{BACKBONE}.keras"
        clean.save(out / "models" / fname)

        y = {s: manifest[manifest.split == s][labels].to_numpy("float32") for s in ("val", "test")}
        p = {s: np.load(project / "results" / f"{task}_{BACKBONE}_{s}.npy") for s in ("val", "test")}
        cfg = {"task": task, "img_size": size, "labels": labels, "models": {BACKBONE: fname},
               "weights": {BACKBONE: 1.0}, "note": "Demo: EfficientNet-B0 only (the full ensemble is too large for GitHub)."}
        if task == "binary":
            cfg["threshold"] = E.threshold_for_sensitivity(y["val"][:, 0], p["val"], 0.90)
            cfg["threshold_rule"] = "90% sensitivity on validation set"
            cfg["test_auc"] = float(roc_auc_score(y["test"][:, 0], p["test"]))
            rep = E.binary_report(y["test"][:, 0], p["test"], cfg["threshold"])
            summary[task] = {k: round(float(rep[k]), 3) for k in ("AUC", "sensitivity", "specificity", "PPV", "NPV")}
        else:
            cfg["thresholds"] = [E.threshold_for_f1(y["val"][:, j], p["val"][:, j]) for j in range(len(labels))]
            cfg["threshold_rule"] = "max F1 per disease on validation set"
            cfg["test_macro_auc"] = float(roc_auc_score(y["test"], p["test"], average="macro"))
            summary[task] = {d: round(float(roc_auc_score(y["test"][:, j], p["test"][:, j])), 3) for j, d in enumerate(labels)}
        (out / "models" / f"{task}_ensemble.json").write_text(json.dumps(cfg, indent=2))
        print(f"{task}: saved {fname} ({(out / 'models' / fname).stat().st_size / 1e6:.1f} MB), test {summary[task]}")
    return summary


def export_samples(project: Path, out: Path) -> None:
    from PIL import Image
    from iluminado.data import crop_fundus

    manifest = pd.read_csv(project / "manifest.csv")
    raw = pd.read_csv(project / "manifest_raw.csv") if (project / "manifest_raw.csv").exists() else None
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    hf_test = None
    (out / "app" / "samples").mkdir(parents=True, exist_ok=True)
    for name, cond in SAMPLES:
        row = test[cond(test)].iloc[0]
        img = None
        for path in [row.path] + ([] if raw is None else raw[(raw.split == "test") & (raw.ID == row.ID)].path.tolist()):
            if Path(path).exists():
                img = Image.open(path).convert("RGB")
                break
        if img is None:   # Drive copy incomplete: fetch the test split from Hugging Face (about 2.5 GB)
            if hf_test is None:
                from datasets import load_dataset
                from iluminado.download import HF_REPO
                print("Sample images not on Drive; downloading the test split from Hugging Face ...")
                hf_test = load_dataset(HF_REPO, split="test")
            img = hf_test[int(row.ID) - 1]["image"].convert("RGB")
        crop_fundus(img).resize((512, 512), Image.BILINEAR).save(out / "app" / "samples" / f"{name}_test{row.ID}.jpg", quality=90)
    print("samples:", sorted(p.name for p in (out / "app" / "samples").iterdir()))


def write_requirements(out: Path) -> None:
    import keras
    import tensorflow as tf

    (out / "app" / "requirements.txt").write_text(
        "# Pinned to the versions that saved the models (written by scripts/export_demo.py)\n"
        f"tensorflow-cpu=={tf.__version__}\nkeras=={keras.__version__}\n"
        "streamlit>=1.50\nnumpy\npandas\npillow\nmatplotlib\nscikit-learn\n")
    print(f"requirements pinned to tensorflow-cpu {tf.__version__}, keras {keras.__version__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default="/content/drive/MyDrive/iluminado")
    ap.add_argument("--out", default="/content/iluminado-demo.zip")
    a = ap.parse_args()
    project, zip_path = Path(a.project_dir), Path(a.out)
    stage = zip_path.with_suffix("")
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "models").mkdir(parents=True)
    summary = export_models(project, stage)
    export_samples(project, stage)
    write_requirements(stage)
    (stage / "models" / "demo_test_metrics.json").write_text(json.dumps(summary, indent=2))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in stage.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(stage))
    print(f"\nWrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB). Unzip it into the repo folder and push.")


if __name__ == "__main__":
    main()
