# mri-anomaly-ood-calibration

Does a brain-MRI anomaly detector stay reliable under scanner / dataset shift? Fixed vs adaptive thresholds.

**Research question:** Does your anomaly detector stay reliable under distribution shift?

**Paper:** _arXiv link — TBD (add here as soon as the preprint is posted)_

## Goal

Take the ViT / f-AnoGAN brain-MRI anomaly detectors and test them across scanners or datasets. Report AUROC, calibration (Platt-scaled on the source domain) and false-positive drift; compare a fixed threshold with an adaptive threshold (the link to SAAD). Output: a cleaned repo plus one extended paper merging the ICCIT and SAAD work.

## Data

- **IXI** (healthy, multi-scanner: Guys/HH/IOP), **BraTS** (tumors), **fastMRI** brain (multi-contrast / field strength).
- Anomaly-detection convention: healthy scans = label 0, pathological = label 1. Be explicit about what is the source domain and what is shifted; dataset identity is confounded with pathology if normals and anomalies come from different datasets, so report within-site comparisons where possible.
- Download from the official sites (registration/terms apply) into `data/`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (one command)

```bash
make run        # evaluate scores/<domain>.csv files -> results/
make smoke      # synthetic sanity check
```

`make smoke` runs the whole pipeline end-to-end on synthetic data in under a minute and checks that the code works. The numbers it produces are NOT results.

All experiments use fixed seeds (`seed: 42` in the configs, applied through `utils.set_seed`). After your final run, lock the environment with `pip freeze > requirements.lock.txt`.

## Repository layout

```
mri_ood/
  models.py          scorer interface (plug in your ViT / f-AnoGAN)
  extract_scores.py  writes scores/<domain>.csv  (to be implemented for your preprocessing)
  calibration.py     Platt scaling on source validation
  thresholds.py      fixed vs adaptive (robust-z) thresholds
  metrics.py         AUROC, AUPRC, ECE, Brier, TPR/FPR/F1
  evaluate.py        results table + shift figures
configs/ results/ paper/
```

## Results

### Shift results (TBD after real runs)

| domain | n | AUROC | AUPRC | ECE | TPR/FPR fixed | TPR/FPR adaptive |
|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Main figure

![shift](results/shift_main.png)

## Limitations

- `thresholds.adaptive_threshold` is a generic robust-z rule; replace it with the SAAD rule before reporting.
- `extract_scores.py` and `models.py` are stubs: they need your trained checkpoints and preprocessing.

## Citation

See `CITATION.cff`. A BibTeX entry will be added once the arXiv paper is out.
