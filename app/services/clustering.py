from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

def auto_kmeans(X: np.ndarray, k_min: int = 4, k_max: int = 10) -> Tuple[int, float, np.ndarray]:
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1.0
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_k is None:
        km = KMeans(n_clusters=min(4, len(X)), n_init="auto", random_state=42)
        best_labels = km.fit_predict(X)
        best_k = len(set(best_labels))
        best_score = -1.0
    return best_k, best_score, best_labels

def pca_2d(X: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=42).fit_transform(X)

def tsne_2d(X: np.ndarray) -> np.ndarray:
    return TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(X)
