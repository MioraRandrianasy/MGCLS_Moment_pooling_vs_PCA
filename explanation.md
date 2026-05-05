
**Why we standardise first (`StandardScaler`)**

BYOL features are 512-dimensional vectors whose individual dimensions have very different scales: some might range from 0 to 0.01, others from -100 to 100. Nearly every anomaly detector (Isolation Forest, ECOD, COPOD, PCA) is sensitive to scale. If we don't normalise, the large-scale dimensions dominate all distance and covariance calculations and the small-scale ones are invisible. `StandardScaler` subtracts the mean and divides by the standard deviation per dimension, bringing everything to the same playing field.

---

**What Moment Pooling actually does (and why we use it)**

The paper arXiv:2403.08854 proposes this as a smarter alternative to just using PCA components as features. The logic is:

1. First, PCA compresses 512 dimensions down to `latent_dim` (note: 8) components keeping only the directions of maximum variance.
2. Then, `PolynomialFeatures` with `order=2` expands those 8 values into all their combinations: the original 8 values, their squares (z0², z1², …), and their cross-products (z0·z1, z0·z2, …). That gives 44 features.

Those cross-products are what matter. A plain PCA score only captures "is this object far from the mean in the big directions?" Moment Pooling also captures "does this object have an unusual *relationship* between dimensions?" — which is closer to what the human annotators are labelling as interesting. The bias column (the constant 1 term) is dropped before scoring because it carries no information about the object.

The `fit`/`transform` split added to `utils.py` is important for correctness: you fit PCA on the training data only, then apply the same learned transformation to new data. Without that split, if you ever evaluate on a held-out set, you'd be leaking information from the test set into the PCA axes.

---

**Why L2 alone underperforms**

After Moment Pooling you get a 44-dimensional vector per object. Taking the L2 norm of that vector means "how far is this object from the origin in moment space?" That works if anomalies are statistically extreme — very large or very small values. But many interesting radio sources are unusual because of their *shape*, not because their pixel statistics are extreme. BYOL encodes this semantic information in specific directions in feature space, but those directions may not align with the polynomial axes that L2 is measuring. That's what the Summary section's "why Moment Pooling alone underperforms" was pointing at.

---

**Why Isolation Forest, and why `score_samples` not `fit_predict`**

Isolation Forest works by building random trees that try to isolate each point. Anomalies are isolated in fewer splits — they end up with shorter path lengths. The key fix from previous notebook: `fit_predict` returns -1/1 class labels, which throws away all the continuous information and makes metrics like ROC-AUC meaningless. `score_samples` returns the raw anomaly score (negative, so we negate it: more anomalous = higher score), which is what ROC-AUC and PR-AUC need.

We also increased `n_estimators` from the default 100 to 300, because with fewer trees the scores are unstable, running twice gives different rankings.

---

**Why Extended Isolation Forest (EIF)**

Standard IF has a documented geometric bias: because it only makes axis-aligned cuts, points that are anomalous but happen to lie near the centre of the distribution along most axes are hard to isolate. EIF (arXiv:2110.13402) uses random hyperplane cuts instead — any direction, not just horizontal/vertical. For radio morphologies that are unusual in oblique directions through feature space, this can recover anomalies that standard IF misses.

---

**Why ECOD and COPOD (PyOD)**

These are the "statistical anomaly detection on pre-computed features" approach. ECOD estimates, for each dimension independently, how extreme a value is by looking at its empirical cumulative distribution (what fraction of the data is more extreme in the left or right tail). It then combines these per-dimension tail probabilities. It's parameter-free, fast, and interpretable, you can see which dimensions made a point look anomalous. COPOD does something similar but models the joint distribution using copulas, which can capture dependencies between dimensions that ECOD misses.

Both work directly on the Moment Pooling features without needing any hyperparameter tuning, which is why they're good early candidates.

---

**Supervised vs Unsupervised: When does the algorithm "train"?**

One crucial detail in this notebook is understanding *when* and *how* the algorithms learn. In traditional supervised machine learning, you split your data into a training set and a testing set, and you feed the algorithm both the features and the labels so it can learn to predict the specific anomalies you are looking for.

In this notebook, **none of the models are trained with labels**. We are doing purely **unsupervised** anomaly detection. 
- The human labels (`y_interesting`, derived from `evaluation_subset_author_ML_score`) are separated at the very beginning and are never passed to any model.
- **The "Training" (Fitting) phase**: When we call `.fit()` or `.fit_transform()` on `PCA`, `MomentPooling`, `IsolationForest`, `ECOD`, or `COPOD`, we only provide the features (e.g. `X` or `X_mp_clean`). The algorithms "train" by looking at the entire dataset to learn its underlying statistical structure (such as the principal components, the expected distribution, or the best ways to isolate points). They don't know which points are actual anomalies; they simply map out what the "normal" background looks like.
- **The "Scoring" phase**: When we extract the anomaly scores (e.g., via `.score_samples()` or `.decision_scores_`), the models are purely measuring how much each object deviates from that learned normal background.
- **The "Evaluation" phase**: We only bring the labels back at the very end to evaluate the results. Functions like `compute_metrics` compare the unsupervised scores against the hidden human labels to see how well the algorithms' mathematical definition of "unusual" aligns with the human annotators' definition of "interesting" (giving us our ROC-AUC and PR-AUC scores).

Because the algorithms are unsupervised (they don't "cheat" by looking at the answers), it's acceptable to evaluate their performance on the same dataset they were fitted on for this exploratory analysis. However, as noted in the Moment Pooling section, if we want to deploy this to production to score *new* radio sources, we would only `.fit()` on historical data and use `.transform()` or `.score_samples()` on the new data.

---

**Step 1: The hyperparameter sweep**

`latent_dim` controls how many PCA components survive before polynomial expansion. Too small (4) and you lose signal; too large (16) and you keep noise, plus the polynomial expansion grows as O(d^order) so 16 components at order 3 would give hundreds of features. `order=2` vs `order=3` tests whether cubic terms (which approximate skewness) add anything over just means, variances, and covariances. The heatmap makes it easy to see whether the two knobs interact, maybe `latent_dim=8, order=2` is better than `latent_dim=16, order=3` even though more features sounds better.

We run this sweep only with ECOD as the downstream detector, because ECOD is parameter-free,if we swept both the preprocessing and the detector simultaneously we'd have too many combinations and couldn't isolate which change helped.

---

**Step 2: Raw BYOL features (no PCA)**

This is the critical test of whether Moment Pooling's PCA bottleneck is helping or hurting. PCA maximises variance retained, but anomalies are rare, their variance contribution is tiny, and PCA might actively discard the directions where anomalies live. By running ECOD and COPOD directly on the full feature space (after a variance threshold filter to remove near-constant dimensions that add noise), we can see whether the compression step is a net win or a net loss.

`VarianceThreshold(threshold=0.01)` removes dimensions where the variance across all objects is below 0.01 on the standardised data, these dimensions are essentially constant and contribute nothing to anomaly detection.

---

**Step 3: The ensemble**

Individual detectors disagree on borderline cases. Averaging their scores is a well-known way to reduce variance. But you can't average raw scores because ECOD scores and IsoForest scores are on completely different scales. Rank normalisation fixes that: it maps each score to its rank divided by n-1, giving everyone a [0,1] value where 0 means "most normal" and 1 means "most anomalous". After that, averaging is meaningful.

The three ensemble variants test different hypotheses: does giving more weight to the better-performing detectors help, or is their "better performance" just noise? The top-3 ensemble specifically tests whether a smaller, curated committee beats an indiscriminate average of everything.

---

**Step 4: The full comparison table**

All methods in one place, sorted by ROC-AUC, with green highlighting on the best value per column.

---

The overall philosophy throughout is: each new method is either testing a different *representation* (Moment Pooling vs raw BYOL), a different *detector* (statistical vs tree-based), or a different *combination strategy* (ensemble), and each one is evaluated with the same four metrics so results are directly comparable. 