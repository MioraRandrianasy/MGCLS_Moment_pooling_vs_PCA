import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations_with_replacement
from sklearn.metrics import roc_auc_score, average_precision_score

# Automatically resolves project root
BASE_DIR = Path(__file__).resolve().parent.parent

def load_features():
    path = BASE_DIR / "data" / "mgcls_byol_features.parquet"
    df = pd.read_parquet(path)
    print("Features loaded:", df.shape)
    return df

def load_catalogue():
    path = BASE_DIR / "data" / "protege_catalogue.csv"
    df = pd.read_csv(path)
    print("Catalogue loaded:", df.shape)
    return df

def compute_metrics(y_true, scores):
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores)
    }

def topk_recall(labels, scores, k=100):
    ranked = scores.sort_values(ascending=False).index[:k]
    return labels.loc[ranked].sum() / labels.sum()

class MomentPooling:
    def __init__(self, latent_dim=4, order=3, include_bias=True):
        self.latent_dim = latent_dim
        self.order = order
        self.include_bias = include_bias
        self.pca = PCA(n_components=self.latent_dim)
        self.poly = PolynomialFeatures(degree=self.order, include_bias=self.include_bias, interaction_only=False)

    def _make_feature_names(self, n_features):
        names = []
        if self.include_bias:
            names.append("bias")

        for degree in range(1, self.order + 1):
            for comb in combinations_with_replacement(range(n_features), degree):
                names.append("*".join([f"z{i}" for i in comb]))

        return names

    def fit_transform(self, X):
        Z = self.pca.fit_transform(X)
        Z_poly = self.poly.fit_transform(Z)
        feature_names = self._make_feature_names(Z.shape[1])
        return pd.DataFrame(Z_poly, index=X.index, columns=feature_names)
