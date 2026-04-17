# Anomaly Detection Evaluation (anomaly-mgcls)

This repository evaluates and compares static anomaly detection scoring methods on a labelled dataset originally developed through the Protege active learning workflow. Note that this evaluation is a static ranking comparison intended to observe which method better identifies \"interesting\" anomalies.

## Methods Evaluated
1. **PCA (Baseline):** Pre-computed scores loaded directly from `protege_catalogue.csv`.
2. **PCA (Isotree):** Isolation Forest baseline trained on standard PCA components.
3. **Moment Pooling:** Computes summary statistics (higher-order polynomial feature expansions evaluated via L2 norm) over the dataset to act as a static anomaly metric.
4. **Moment Pooling (Isotree):** Isolation Forest trained on the expanded Moment Pooling features.

## Results Overview
Based on evaluation against the interesting anomaly label subset (human surveyor scores of 4 to 5):
* **PCA (Baseline)** significantly outperforms all newly implemented metrics, including Moment Pooling and Isolation Forest variations.
* **PCA (Baseline) Metrics:** ~0.867 ROC-AUC, ~0.105 PR-AUC, ~17.4% Recall@100.
* **Moment Pooling (L2) Metrics:** ~0.490 ROC-AUC, ~0.013 PR-AUC, 0.0% Recall@100.
* **Moment Pooling (Isotree) Metrics:** ~0.464 ROC-AUC, ~0.012 PR-AUC, 0.0% Recall@100.
* **PCA (Isotree) Metrics:** ~0.477 ROC-AUC, ~0.012 PR-AUC, 0.0% Recall@100.

The detailed visual tracking of cumulative recall indicates that the original PCA method correctly prioritizes interesting anomalies, whereas Moment Pooling and Isolation Forest fail to detect them accurately.

## Key Files
* `scripts/evaluation.ipynb` - The primary Jupyter Notebook serving as the all-in-one visualizer, computing component comparisons, printing metrics (ROC-AUC, PR-AUC, Recall), and plotting the runtime and score distributions. All experiment logic (such as models like Isolation Forest) runs locally in this notebook.
* `scripts/utils.py` - Consolidated utility file containing all python functionality, including the data loading logic, metric calculations, and the base Moment Pooling class implementation.
* `data/` - Holds the dataset, specifically the catalog `protege_catalogue.csv` and the features `mgcls_byol_features.parquet`.

## Quick Start
To interactively reproduce the analyses, visualizations, and metrics, simply execute the cells within the provided notebook:

```bash
cd scripts
jupyter notebook evaluation.ipynb
```
