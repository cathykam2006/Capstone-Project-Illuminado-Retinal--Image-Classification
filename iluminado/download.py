"""Download RFMiD from the Hugging Face Hub (no login needed).

The Hugging Face mirror stores each image with a `Disease_Risk` flag and a text
`label` column listing its diseases. This module exports it to the same folder
layout as the Kaggle copy, and writes label CSVs built from the mirror itself,
so every image travels with its own labels (no reliance on row order matching
the official CSVs). It also records each image's original size, which is the
best available proxy for which of the three cameras took it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HF_REPO = "bumbledeep/rfmid"

CODE_NAMES = {
    "DR": "diabetic retinopathy", "ARMD": "age-related macular degeneration", "MH": "media haze",
    "DN": "drusens", "MYA": "myopia", "BRVO": "branch retinal vein occlusion", "TSLN": "tessellation",
    "ERM": "epiretinal membrane", "LS": "laser scars", "MS": "macular scar",
    "CSR": "central serous retinopathy", "ODC": "optic disc cupping",
    "CRVO": "central retinal vein occlusion", "TV": "tortuous vessels", "AH": "asteroid hyalosis",
    "ODP": "optic disc pallor", "ODE": "optic disc edema", "ST": "optociliary shunt",
    "AION": "anterior ischemic optic neuropathy", "PT": "parafoveal telangiectasia",
    "RT": "retinal traction", "RS": "retinitis", "CRS": "chorioretinitis", "EDN": "exudation",
    "RPEC": "retinal pigment epithelium changes", "MHL": "macular hole", "RP": "retinitis pigmentosa",
    "CWS": "cotton-wool spots", "CB": "coloboma", "ODPM": "optic disc pit maculopathy",
    "PRH": "preretinal hemorrhage", "MNF": "myelinated nerve fibers", "HR": "hemorrhagic retinopathy",
    "CRAO": "central retinal artery occlusion", "TD": "tilted disc", "CME": "cystoid macular edema",
    "PTCR": "post-traumatic choroidal rupture", "CF": "choroidal folds", "VH": "vitreous hemorrhage",
    "MCA": "macroaneurysm", "VS": "vasculitis", "BRAO": "branch retinal artery occlusion",
    "PLQ": "plaque", "HPED": "hemorrhagic pigment epithelial detachment", "CL": "collateral",
}
CODES = list(CODE_NAMES)
NAME_TO_CODE = {v: k for k, v in CODE_NAMES.items()}
EMPTY_TOKENS = {"", "nan", "none", "null", "normal", "healthy", "no disease", "no_disease", "[]"}

# HF split name -> (our split, Kaggle-style folder, official CSV name)
SPLIT_MAP = {
    "train": ("train", "Training_Set/Training", "RFMiD_Training_Labels.csv"),
    "validation": ("val", "Evaluation_Set/Validation", "RFMiD_Validation_Labels.csv"),
    "test": ("test", "Test_Set/Test", "RFMiD_Testing_Labels.csv"),
}


def _pieces(value) -> list[str]:
    if value is None or (isinstance(value, float) and value != value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value]
    s = str(value).strip().strip("[]{}")
    return [p.strip().strip("'\"").strip() for p in re.split(r"[,;|/\n]+", s)]


def _is_multihot(value) -> bool:
    """A 45-long list of 0/1 flags, one per disease in official column order."""
    toks = [t for t in _pieces(value) if t != ""]
    return len(toks) == len(CODES) and all(t in ("0", "1") for t in toks)


def _infer_mode(values) -> str:
    vals = list(values)
    if vals and all(_is_multihot(v) for v in vals):
        return "multihot"
    tokens = [t for v in values for t in _pieces(v) if t.lower() not in EMPTY_TOKENS]
    if tokens and not any(t.upper() in CODE_NAMES or t.lower() in NAME_TO_CODE for t in tokens) \
            and all(t.isdigit() for t in tokens):
        return "index"
    return "names"


def parse_label(value, mode: str) -> list[str]:
    if mode == "multihot":
        flags = [t for t in _pieces(value) if t != ""]
        return [c for c, f in zip(CODES, flags) if f == "1"]
    out = []
    for t in _pieces(value):
        tl = t.lower()
        if tl in EMPTY_TOKENS:
            continue
        if mode == "index" and t.isdigit():
            i = int(t)
            if not 0 <= i < len(CODES):
                raise ValueError(f"Label index {i} is outside 0..{len(CODES) - 1}")
            out.append(CODES[i])
        elif t.upper() in CODE_NAMES:
            out.append(t.upper())
        elif tl in NAME_TO_CODE:
            out.append(NAME_TO_CODE[tl])
        else:
            m = re.search(r"\(([A-Za-z]+)\)\s*$", t)   # e.g. "Diabetic retinopathy (DR)"
            if m and m.group(1).upper() in CODE_NAMES:
                out.append(m.group(1).upper())
            else:
                raise ValueError(f"Cannot interpret label token {t!r} (from {value!r}).")
    return sorted(set(out))


def label_frame(disease_risk, labels) -> pd.DataFrame:
    """Official-style label table. Stops if an image marked healthy carries a disease (= misread encoding)."""
    mode = _infer_mode(labels)
    parsed = [parse_label(v, mode) for v in labels]
    dr = pd.Series(list(disease_risk), dtype="int64")
    bad = [i for i, (d, p) in enumerate(zip(dr, parsed)) if d == 0 and p]
    if bad:
        raise ValueError(f"{len(bad)} images have Disease_Risk=0 but disease labels (e.g. row {bad[0]}: "
                         f"{labels[bad[0]]!r}); the label encoding is not what the parser assumes.")
    frame = pd.DataFrame({"ID": range(1, len(dr) + 1), "Disease_Risk": dr})
    for c in CODES:
        frame[c] = [int(c in p) for p in parsed]
    return frame


def fetch_hf(out_dir: str | Path, repo: str = HF_REPO, max_side: int | None = 1024, log_every: int = 400,
             official_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Download and export RFMiD. Returns (image_root, label_dir).

    Re-running skips images already exported. max_side shrinks images on export
    (1024 keeps more detail than training uses and needs ~3 GB instead of ~15 GB).
    """
    from datasets import load_dataset

    out_dir = Path(out_dir)
    image_root, label_dir = out_dir / "images", out_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {repo} from the Hugging Face Hub (first run downloads about 8 GB) ...")
    dd = load_dataset(repo)
    missing = [k for k in SPLIT_MAP if k not in dd]
    if missing:
        raise KeyError(f"The mirror has splits {list(dd)}; expected {list(SPLIT_MAP)}.")
    sizes = []
    for hf_name, (split, folder, csv_name) in SPLIT_MAP.items():
        ds = dd[hf_name]
        for col in ("image", "Disease_Risk", "label"):
            if col not in ds.column_names:
                raise KeyError(f"Expected column {col!r}; the mirror has {ds.column_names}.")
        frame = label_frame(ds["Disease_Risk"], ds["label"])
        if official_dir is not None:
            official = pd.read_csv(Path(official_dir) / csv_name)
            cols = [c for c in official.columns if c != "ID"]
            diff = (frame[cols].sum() - official[cols].sum())
            if len(frame) != len(official) or (diff != 0).any():
                raise ValueError(f"{hf_name}: parsed labels don't match the official RFMiD label totals "
                                 f"(differences: {diff[diff != 0].to_dict()}). The mirror's label format may have "
                                 "changed; use SOURCE = 'kaggle' or 'local' instead.")
        frame.to_csv(label_dir / csv_name, index=False)
        dst_dir = image_root / folder
        dst_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exporting {hf_name}: {len(ds)} images")
        for i in range(len(ds)):
            dst = dst_dir / f"{i + 1}.png"
            if dst.exists() and (label_dir / "image_info.csv").exists():
                continue
            img = ds[i]["image"].convert("RGB")
            sizes.append({"split": split, "ID": i + 1, "orig_w": img.width, "orig_h": img.height})
            if not dst.exists():
                if max_side and max(img.size) > max_side:
                    s = max_side / max(img.size)
                    img = img.resize((round(img.width * s), round(img.height * s)))
                img.save(dst)
            if log_every and (i + 1) % log_every == 0:
                print(f"  {i + 1}/{len(ds)}")
    if sizes:
        info_path = label_dir / "image_info.csv"
        new = pd.DataFrame(sizes)
        if info_path.exists():
            new = pd.concat([pd.read_csv(info_path), new]).drop_duplicates(["split", "ID"], keep="last")
        new.sort_values(["split", "ID"]).to_csv(info_path, index=False)
    return image_root, label_dir


def compare_with_official(label_dir: str | Path, official_dir: str | Path) -> pd.DataFrame:
    """Per split: do the downloaded labels match the official CSVs shipped in the repo?"""
    rows = []
    for split, _, csv_name in SPLIT_MAP.values():
        a = pd.read_csv(Path(label_dir) / csv_name).set_index("ID")
        b = pd.read_csv(Path(official_dir) / csv_name).set_index("ID")
        cols = [c for c in b.columns if c in a.columns]
        common = a.index.intersection(b.index)
        same_rows = (a.loc[common, cols] == b.loc[common, cols]).all(axis=1).mean()
        same_totals = bool((a[cols].sum() == b[cols].sum()).all())
        rows.append({"split": split, "images": len(a), "official": len(b),
                     "column totals match": same_totals, "identical rows (same ID)": round(float(same_rows), 4)})
    return pd.DataFrame(rows).set_index("split")
