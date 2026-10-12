# Iluminado: Retinal Disease Screening with Deep Learning

Transfer-learning ensemble that screens retinal fundus photographs for disease, and flags three specific conditions: diabetic retinopathy (DR), media haze (MH) and optic disc cupping (ODC). Built on the public RFMiD dataset, with a Streamlit demo app.

> **Version 2 (2026).** This is a full rebuild of my 2023 data science capstone. A code review found that the original training pipeline paired images with the wrong labels, so the original results were not valid. Every issue found and how it was fixed is listed under [What changed in v2](#what-changed-in-v2).

## Why this project

The WHO estimates that at least 2.2 billion people live with vision impairment, and that at least 1 billion of those cases could have been prevented or have yet to be addressed ([WHO](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment)). A major barrier is the shortage of trained eye care providers. Automated screening of fundus photos, which can be taken with portable cameras by trained non-specialists, is one way to prioritise who needs an ophthalmologist first.

I had implantable collamer lens (ICL) surgery and saw first-hand how much imaging and software contribute to diagnosis. This project is my attempt to apply data science to that problem.

**Scope:** this is a research prototype trained on 3,200 images. It is not a medical device and has not been clinically validated.

## Dataset

[Retinal Fundus Multi-Disease Image Dataset (RFMiD)](https://www.mdpi.com/2306-5729/6/2/14): 3,200 fundus images from three cameras (2009-2020), annotated by two senior retinal experts with 45 disease labels. Official split: 1,920 training, 640 validation, 640 test images. Low-quality images are included on purpose.

- About **79% of images show disease**, so "always predict diseased" is 79% accurate. Accuracy alone is therefore a misleading metric here.
- Images often carry **several labels**: 458 training images have two or more diseases.

The label CSVs are in `data/`. The images (~40 GB) are not in this repo; notebook 01 downloads them from Kaggle.

## Approach

```
fundus photo ──► crop to fundus, square, resize ──► 3 CNN backbones ──► weighted average ──► probabilities
                 (identical in training & app)      EfficientNet-B0       (weights =          ├─ disease risk (yes/no)
                                                     DenseNet-121          validation AUC)     └─ DR / MH / ODC (independent)
                                                     ConvNeXt-Tiny
```

1. **Preprocessing.** Each image is cropped to the circular fundus and padded to a square, removing camera-specific black borders, then resized to 384 × 384.
2. **Models.** ImageNet-pretrained backbones with a new sigmoid head. Training is two-stage: the head first, then the top 30% of the backbone at a low learning rate. Augmentation (flips, 90° rotations, brightness, contrast) is fresh every epoch.
3. **Imbalance.** A class-weighted loss for both tasks.
4. **Model selection.** Early stopping, ensemble weights and decision thresholds all use the **validation** set. The test set is used once, at the end.
5. **Thresholds.** For disease risk, the threshold is set to catch 90% of diseased eyes on validation (screening setting, where a miss costs more than a false alarm). For DR/MH/ODC, each disease gets the threshold that maximises F1 on validation.
6. **Explainability.** Grad-CAM heatmaps show which regions of the retina drove each prediction.

## Results

> Fill these in from the last cell of notebooks 03 and 04 after training. All numbers are on the held-out test set (640 images).

**Disease risk (healthy vs diseased)**

| Model | AUC (95% CI) | Sensitivity | Specificity | PPV | NPV |
|---|---|---|---|---|---|
| EfficientNet-B0 | | | | | |
| DenseNet-121 | | | | | |
| ConvNeXt-Tiny | | | | | |
| **Ensemble** | | | | | |
| Always "diseased" | 0.500 | 1.000 | 0.000 | 0.791 | — |

**Disease type (ensemble, per disease)**

| Disease | Test positives | AUC | PR-AUC | Prevalence | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| DR | 124 | | | 0.194 | | | |
| MH | 104 | | | 0.163 | | | |
| ODC | 91 | | | 0.142 | | | |

<!-- After training, copy results/*.png from your Drive into docs/ and uncomment:
![ROC and confusion matrix](docs/binary_roc_cm.png)
![Grad-CAM examples](docs/binary_gradcam.png)
-->

## What changed in v2

| # | Issue in v1 | Effect | Fix |
|---|---|---|---|
| 1 | Features came from a shuffled generator, labels from `.labels` in folder order | Every model trained on scrambled labels; validation AUC ≈ 0.5 | `tf.data` keeps each image and its label together through shuffling (covered by a test) |
| 2 | Test set used as `validation_data` with early stopping | Test scores were optimistic | Official validation split for all tuning; test used once |
| 3 | Multi-class models all reused InceptionV3 features | "Six architectures" were one feature extractor | Each backbone trained end to end |
| 4 | Images moved into one folder per disease | Multi-label images lost labels; ~150 target-disease images dropped | Manifest keeps all labels; multi-label sigmoid model |
| 5 | Disease model trained only on images with DR/MH/ODC | Couldn't learn "none of these" | Trained on all images |
| 6 | `rescale=1/255` for every backbone | EfficientNet/ResNeXt collapsed to constant "diseased" | Each model does its own correct scaling internally |
| 7 | Augmented features extracted once | One frozen augmentation per image | Fresh augmentation every epoch |
| 8 | Declared input sizes didn't match the 256 px generator default | Unclear what actually ran | One `IMG_SIZE`, used everywhere |
| 9 | Accuracy as headline; no baseline | Results (0.73-0.78) were below the 0.79 do-nothing baseline | AUC, sensitivity/specificity, bootstrap CIs, baseline row |
| 10 | "Ensemble" in title, no ensemble in code | Claim not supported | Validation-AUC-weighted ensemble |
| 11 | App loaded a non-existent `model.keras`, wrong size, no scaling, mismatched backbone/head | App could not give valid predictions | App shares the training preprocessing and loads the saved ensemble |
| 12 | App saved every upload to disk, even without consent | Privacy risk for medical images | Uploads processed in memory only, never stored |
| 13 | No disclaimer; raw probability array shown | Could be mistaken for a diagnosis | Clear "not a medical device" notice and plain-language output |
| 14 | Absolute `/Users/...` paths, `cd` cells, no requirements | Not reproducible | Config cells, `requirements.txt`, runs on Colab |
| 15 | README images pointed at `localhost` | Broken for everyone else | Images stored in the repo |
| 16 | "SE-ResNeXt" was plain ResNeXt-50 | Mislabelled | Replaced with clearly named backbones |

## Repository layout

```
iluminado/                      shared code used by notebooks and app
  data.py                       manifest, fundus crop, caching, tf.data pipeline
  models.py                     backbones, weighted loss, two-stage training
  evaluate.py                   thresholds, clinical metrics, bootstrap CIs, ensembling
  gradcam.py                    Grad-CAM heatmaps
notebooks/
  01_data_preparation.ipynb     download, manifest, crop + cache
  02_eda.ipynb                  balance, baseline, co-occurrence, camera check
  03_disease_risk_binary.ipynb  healthy vs diseased + ensemble
  04_disease_type_multilabel.ipynb  DR / MH / ODC + ensemble
app/streamlit_app.py            demo app
data/                           RFMiD label CSVs
models/                         trained models + ensemble JSON (added after training)
tests/test_pipeline.py          synthetic-data tests (no dataset or GPU needed)
```

## How to run

**Train (Google Colab, free GPU)**
1. Open each notebook in Colab (File → Open notebook → GitHub → this repo) and set Runtime → Change runtime type → GPU.
2. Run 01 → 02 → 03 → 04. Outputs are saved to `MyDrive/iluminado/`, so a disconnect doesn't lose finished models.
3. Notebook 01 downloads RFMiD with `kagglehub` (needs a free Kaggle API token), or point it at your own copy.

**Run the app**
```bash
pip install -r requirements.txt
# copy *.keras and *_ensemble.json from MyDrive/iluminado/models/ into models/
streamlit run app/streamlit_app.py
```
GitHub rejects files over 100 MB. EfficientNet-B0 and DenseNet-121 fit; ConvNeXt-Tiny (~110 MB) needs [Git LFS](https://git-lfs.com/) or can be left out. The app skips missing models and reweights the rest.

**Tests**
```bash
python -m pytest tests -q
```

## Limitations and next steps

- 3,200 images from one dataset; performance on other cameras and populations is unknown. External validation (for example on ODIR or APTOS) is the most important next step.
- Only 3 of 45 conditions are classified individually.
- Labels come from two graders, so label noise is likely for subtle cases.
- Grad-CAM shows where the model looked, not whether its reasoning is clinically correct.

## References

1. Pachade et al. (2021). [Retinal Fundus Multi-Disease Image Dataset (RFMiD)](https://www.mdpi.com/2306-5729/6/2/14). *Data*, 6(2), 14.
2. World Health Organization. [Blindness and vision impairment](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment).
3. Selvaraju et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
