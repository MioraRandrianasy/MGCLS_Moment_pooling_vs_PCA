# Anomaly Detection Evaluation Pipeline — Technical Explanation

This document explains the methodology, design choices, and results of
`scripts/evaluation.ipynb`, which benchmarks anomaly detection strategies
on MGCLS (MeerKAT Galaxy Cluster Legacy Survey) radio-source features
extracted with BYOL (Bootstrap Your Own Latent).

---

## 1. Data & Ground Truth

**Input features** (`X`): BYOL embeddings — one 512-dimensional vector per
radio source.  
**Labels** (`labels`): Expert-assigned interest scores from 1 (ordinary) to 5
(highly anomalous).  
**Binary anomaly flag** (`y_interesting`): sources with score ≥ 4 are treated
as anomalies (positive class).

The dataset is highly imbalanced: anomalies (score 4–5) represent a small
fraction of the full catalogue, which is why PR-AUC and Recall@100 carry more
diagnostic weight than ROC-AUC alone.

---

## 2. Evaluation Metrics

| Metric | Definition | Why it matters |
|---|---|---|
| **ROC-AUC (4–5)** | Area under the ROC curve, anomaly = score ≥ 4 | Global ranking quality |
| **PR-AUC (4–5)** | Area under the Precision-Recall curve | Robust to class imbalance |
| **Recall@100 (4–5)** | Fraction of true anomalies in the top-100 predictions | Operational: how many rare objects are surfaced in a short review list |
| **Spearman (1–5)** | Rank correlation between anomaly score and full expert label | Sensitivity across the whole 1–5 scale, not just the binary threshold |

---

## 3. Feature Representations

### 3.1 PCA (Protege baseline)

The **Protege** pipeline compresses the 512-dim BYOL vectors using Principal
Component Analysis (PCA) and then scores each source by its **reconstruction
error** — the L2 distance between the original embedding and its PCA
approximation. A large reconstruction error indicates that the source does not
fit the principal subspace defined by the bulk of the population, and is
therefore anomalous.

PCA is a linear, closed-form method: no training loop is required once the
projection matrix has been fitted on the data.

### 3.2 Raw BYOL Features

Some detectors are applied directly to the full 512-dim BYOL vectors without
any dimensionality reduction. This tests whether the high-dimensional space
already separates anomalies well enough for a detector to exploit.

### 3.3 Moment Pooling (MP)

Moment Pooling compresses the BYOL representation by computing **statistical
moments** of feature groups:

* **Order 2** (mean + variance per group) → 2 × `latent_dim` features  
* **Order 3** (mean + variance + skewness per group) → 3 × `latent_dim` features

The result is a compact, statistically rich vector of dimension
`order × latent_dim`.

A hyperparameter sweep was run over:

| `latent_dim` | `order` | Output dim |
|---|---|---|
| 4 | 2 | 8 |
| 4 | 3 | 12 |
| 8 | 2 | 16 |
| 8 | 3 | 24 |
| 16 | 2 | 32 |
| **16** | **3** | **44** |

The best configuration (`latent_dim=16`, `order=3`) yielding a **44-dim**
representation was selected and used for all subsequent Moment Pooling
experiments.

---

## 4. Anomaly Detectors

### 4.1 Static (Training-Free) Detectors

These methods compute anomaly scores analytically from the feature matrix —
no iterative training loop is needed.

| Detector | Input | How it scores anomalies |
|---|---|---|
| **L2 distance** | MP features | Euclidean distance of each point from the dataset centroid |
| **ECOD** | Raw BYOL or MP | Empirical Cumulative Distribution functions; estimates tail probability in each dimension |
| **COPOD** | Raw BYOL or MP | Copula-based tail probability; models feature dependence before estimating outlierness |

### 4.2 Detectors Requiring a Training Phase

These methods fit a model on the unlabelled data before scoring.

| Detector | Input | Training objective |
|---|---|---|
| **Isolation Forest (IsoForest)** | MP features | Random recursive partitioning; anomalies are isolated in fewer splits |
| **Extended Isolation Forest (EIF)** | MP features | Like IsoForest but uses random hyperplane cuts, reducing bias near the boundary |
| **DeepSVDD (BYOL)** | Raw BYOL (512-dim) | Neural network maps inputs to a hypersphere; distance from centre = anomaly score |
| **DeepSVDD (MP)** | MP features (44-dim) | Same objective on the compressed representation |

#### DeepSVDD Architecture

A bias-free MLP (no bias terms, no BatchNorm) is trained to map all sources
into a compact hypersphere centred at `c`. Three rules prevent trivial
hypersphere collapse:

1. **No bias terms** — a bias could shift all outputs to `c`, zeroing the loss
   without learning.
2. **No BatchNorm** — batch normalisation re-centres activations, causing the
   same collapse.
3. **Centre `c` ≠ origin** — if `c = 0`, the network maps everything to zero
   trivially. `c` is initialised as the mean embedding after one random-weight
   forward pass.

---

## 5. Ensemble Scoring & Rank Normalisation

A naïve average of raw anomaly scores across methods is **not valid**: ECOD
returns log-probabilities (negative, near zero), Isolation Forest returns
signed path-length scores (~0.5 for inliers), and DeepSVDD returns Euclidean
distances. These live on completely different scales and distributions.

### Why Rank Normalisation

Rank normalisation converts each method's scores to **uniform percentile
ranks** before combining them:

```
rank_i = rank(score_i) / N
```

This makes every method's output distribution identical (uniform on [0, 1]),
so a simple average is meaningful.

### Top-3 Ensemble (Committee Approach)

Rather than averaging all methods indiscriminately, only the **top-3
performing methods** (by ROC-AUC) are included in the ensemble. Blending
weak detectors with strong ones degrades the ensemble. The rank-normalised
scores of the top-3 are averaged to produce the final ensemble anomaly score.

---

## 6. Results

### 6.1 Full Comparison Table

| Method | ROC-AUC (4–5) | PR-AUC (4–5) | Recall@100 (4–5) | Spearman (1–5) |
|---|---|---|---|---|
| **PCA (Protege)** | **0.8674** | **0.1052** | **0.1744** | 0.0174 |
| Raw BYOL + ECOD | 0.5892 | 0.0213 | 0.0349 | 0.0073 |
| DeepSVDD (BYOL) | 0.5990 | 0.0198 | 0.0349 | — |
| MP + IsoForest | 0.5592 | 0.0195 | 0.0349 | 0.0219 |
| MP + ECOD | 0.5585 | 0.0187 | 0.0465 | 0.0224 |
| MP + COPOD | 0.5550 | 0.0191 | 0.0465 | 0.0213 |
| MP + EIF | 0.5503 | 0.0216 | **0.0581** | 0.0181 |
| DeepSVDD (MP) | 0.5476 | 0.0213 | 0.0349 | — |
| Raw BYOL + COPOD | 0.4655 | 0.0130 | 0.0000 | 0.0204 |
| MP + L2 | 0.4518 | 0.0124 | 0.0000 | 0.0178 |

### 6.2 Key Observations

* **PCA (Protege) dominates** on all primary metrics. Its reconstruction-error
  score is a powerful proxy for anomalousness in this dataset.
* **Raw BYOL + ECOD** is the strongest unsupervised alternative at ROC-AUC
  0.5892, but its Spearman is the worst of the group, suggesting it captures a
  binary "outlier vs. inlier" split rather than a graded ranking.
* **Moment Pooling variants** cluster around ROC-AUC 0.55. The Spearman
  values for MP methods (0.018–0.022) are modestly higher than PCA's (0.017),
  hinting that the compressed moments carry some graded ranking signal across
  the 1–5 scale — but this does not translate into better binary separation.
* **MP + EIF achieves the best Recall@100 among non-PCA methods** (0.0581),
  making it the most operationally useful challenger if the goal is to surface
  rare sources in a short review list.
* **DeepSVDD** performs comparably to the statistical methods without learning
  an obviously better representation, possibly because 150 training epochs on
  CPU are insufficient or because the unlabelled training set contains too
  many inliers for the hypersphere to find a meaningful boundary.
* **MP + L2 and Raw BYOL + COPOD** both achieve Recall@100 = 0, indicating
  their ranking places no true anomaly in the top 100 reviewed sources — they
  are unsuitable for operational use in this form.

---

## 7. Operational Recommendation

For the current stage of the MGCLS anomaly-detection effort:

1. **Use PCA (Protege) as the primary scorer.** It substantially outperforms
   all benchmarked alternatives.
2. **Recall@100 should be the primary success metric** for future method
   development: the operational task is to find rare, scientifically
   interesting sources in the shortest possible review queue.
3. **MP + EIF is the best secondary method** to run alongside PCA. Its
   complementary Recall@100 may catch anomalies that PCA misses.
4. **DeepSVDD is promising** but needs further investigation: more training
   epochs, GPU acceleration, or pretraining on a self-supervised objective
   before fine-tuning with the SVDD loss.

---

## 8. Next Steps

- [ ] Evaluate an **ensemble of PCA + MP + EIF** using rank normalisation to
  check whether blending catches additional anomalies missed by PCA alone.
- [ ] Investigate **DeepSVDD with more epochs** (GPU) or a two-phase
  pretraining strategy.
- [ ] Pivot the hyperparameter sweep to maximise **Recall@100** directly,
  rather than ROC-AUC.
- [ ] Extend the benchmark with **Local Outlier Factor (LOF)** and
  **Autoencoder reconstruction error** on MP features.
