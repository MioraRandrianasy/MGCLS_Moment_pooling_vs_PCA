# Anomaly Detection on MGCLS — Progress Summary
**Internship · University of the Western Cape · 2026**

---

## Context

The goal is to detect anomalous radio sources in the MGCLS dataset without human labels,
and benchmark those methods against **Protege** — a human-in-the-loop active learning
system developed by the supervisor. BYOL features (512-dim self-supervised embeddings)
are provided as the input representation.

---

## What Was Done

### Baseline
Loaded the Protege PCA score from the catalogue as the gold-standard reference.
Evaluation uses four metrics throughout: **ROC-AUC**, **PR-AUC**, **Recall@100**, and
**Spearman correlation** against the raw 1–5 human scores.

---

### Step 1 — Moment Pooling + Detectors
Implemented **Moment Pooling** (arXiv:2403.08854): compress BYOL features with PCA
(`latent_dim` components), then expand with polynomial features (`order` degree) to
capture cross-moment statistics. Five detectors were applied to the resulting features:

- **L2 norm** — distance from the origin in moment space
- **Isolation Forest** (300 trees, negated `score_samples`)
- **Extended Isolation Forest / EIF** (arXiv:2110.13402) — random hyperplane cuts
  to fix the axis-aligned bias of standard IF
- **ECOD** — parameter-free empirical tail-probability scoring
- **COPOD** — copula-based joint distribution scoring

A **hyperparameter sweep** over `latent_dim` ∈ {4, 8, 16} and `order` ∈ {2, 3}
was run with ECOD as the downstream detector, producing heatmaps of all four metrics.

---

### Step 2 — Raw BYOL Features
Skipped the PCA bottleneck entirely and ran ECOD and COPOD directly on the
standardised BYOL features after a variance-threshold filter. This tested whether
Moment Pooling's compression step helps or discards anomaly-relevant signal.

---

### Step 3 — Score Ensemble
Rank-normalised every detector score to [0, 1] (verified direction with ROC-AUC > 0.5
check) and combined them into three ensemble variants:

- **Equal-weight** — simple mean of all normalised scores
- **AUC-weighted** — each detector votes proportionally to its ROC-AUC
- **Top-3** — only the three highest-AUC detectors, equal weight

The top-3 individual methods and their ensemble were plotted alongside Protege.

---

### Step 4 — Full Comparison
All 11 methods from Steps 1–3 collected into one ranked evaluation table and
one cumulative discovery curve — the weekly-meeting summary figure.

---

### Step 5 — DeepSVDD
Implemented **Deep Support Vector Data Description** (Ruff et al., 2018) from scratch
in PyTorch — the first method that *learns* from the data rather than applying fixed rules.

Key design decisions:
- No bias terms and no BatchNorm (both cause hypersphere collapse)
- Centre `c` initialised as the mean embedding after one forward pass
- Two variants: BYOL input (512-dim, 4-layer encoder) and Moment Pooling input (44-dim, 3-layer)

Improvements over the basic version:
- `n_epochs=100` with **cosine LR annealing** (1e-3 → 1e-5)
- **Gradient clipping** (max norm = 1.0) to prevent training spikes
- **Collapse guard** — warns if loss hits zero before epoch 10
- Configurable encoder depth via `depth` parameter

Training loss curves and a 2D PCA embedding visualisation (coloured by true label
vs SVDD score) were added to diagnose convergence and verify the learned space.

---

### Final Plot
After Step 5, all 13 methods are ranked by ROC-AUC. The top-3 (excluding Protege)
are plotted alongside the Protege baseline — a dynamic cell that automatically
updates to whichever methods perform best after each run.

---

## Results (summary)

| Result | Observation |
|--------|-------------|
| Protege wins all metrics | Expected — built with real human labels and active learning |
| Best unsupervised method | Determined by the top-3 cell after each run |
| Spearman ≈ 0 for all methods | None track the full 1–5 scale; they only partially separate score ≥ 4 |
| Ensemble ≥ best individual | Rank-normalised averaging consistently improves over single detectors |

---

## Next Steps

- **PatchCore** — multi-layer BYOL features for richer local representations
- **Simulated Protege** — reproduce the active learning loop with catalogue labels as oracle
  to quantify how much human feedback contributes
