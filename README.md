# Exoplanet Transit Detection with Deep Learning

A complete pipeline that detects exoplanets from real NASA Kepler telescope
data — from raw light curves to a trained convolutional neural network —
following the architecture introduced in Shallue & Vanderburg (2018),
*"Identifying Exoplanets with Deep Learning: A Five-Planet Resonant Chain
around Kepler-80 and an Eighth Planet around Kepler-90."*

## What this does

Planets are detected using the **transit method**: when a planet passes in
front of its host star (from our point of view), it blocks a tiny fraction
of the star's light, causing a small, periodic dip in brightness. This
project builds the full pipeline to find and classify those signals:

```
Raw Kepler light curve
        │
        ▼
Box Least Squares (BLS) baseline — classical period search
        │
        ▼
Preprocessing — detrending, phase-folding, global/local view binning
        │
        ▼
Labeled dataset — 232 real KOIs (confirmed planets + false positives)
        │
        ▼
Dual-branch CNN (PyTorch) — Astronet-style global + local view classifier
```

## Pipeline stages

| Step | File | What it does |
|------|------|---------------|
| 1 | `step1_fetch_kepler10.py` | Pulls a real Kepler-10 light curve via `lightkurve`, plots raw + detrended flux |
| 2 | `step2_bls_search.py` | Runs a blind Box Least Squares period search; recovers Kepler-10b's known 0.8375-day orbital period from noise alone |
| 3 | `step3_preprocessing_pipeline.py` | Reusable function that turns any flattened light curve into fixed-length global (2001-bin) and local (201-bin) views, matching the Astronet paper's input format |
| 4 | `step4_build_dataset.py` | Queries NASA's Exoplanet Archive KOI catalog, downloads and processes light curves for a balanced sample of confirmed planets and false positives |
| 5 | `step5_train_cnn.py` | Trains a two-branch 1D CNN (global view + local view) in PyTorch and evaluates it |

## Results

Trained on 232 real KOI examples (118 confirmed planets, 114 false
positives; 186 train / 46 validation):

| Metric | Score |
|---|---|
| Accuracy | 0.717 |
| Precision | 0.724 |
| Recall | 0.808 |

Confusion matrix (validation set):

|  | Predicted: False Pos | Predicted: Confirmed |
|---|---|---|
| **True: False Pos** | 12 | 8 |
| **True: Confirmed** | 5 | 21 |

### BLS baseline sanity check

Before any ML, a classical Box Least Squares search on Kepler-10 alone
(no label given to the algorithm) found a best-fit period of **0.8375
days** — matching the published value of 0.837491 days for Kepler-10b to
within 0.03%. This confirms the preprocessing and folding logic is
correct before it ever reaches the CNN.

## Limitations & honest notes

- **Dataset size is small for a CNN.** 186 training examples is far
  below the ~15,000 used in the original Astronet paper. Training loss
  dropped to ~0.01 while validation accuracy plateaued around 0.72 — a
  clear sign of overfitting that more data would likely narrow.
- **This is a from-scratch, educational reproduction**, not a
  production-grade detector. It's meant to demonstrate understanding of
  the full pipeline (data access → classical baseline → ML) rather than
  to compete with NASA's own published results.

## Setup

```bash
pip install -r requirements.txt
```

Run the steps in order (1 → 5). Steps 1-4 require internet access to
query NASA's MAST archive and Exoplanet Archive; step 5 only needs the
`.npz` dataset produced by step 4.

## Reference

Shallue, C. J., & Vanderburg, A. (2018). *Identifying Exoplanets with Deep
Learning: A Five-Planet Resonant Chain around Kepler-80 and an Eighth
Planet around Kepler-90.* The Astronomical Journal, 155(2), 94.
