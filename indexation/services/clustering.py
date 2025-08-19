from __future__ import annotations

"""Clustering helpers."""

from typing import Tuple, Iterable
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def auto_kmeans(X: np.ndarray, k_min: int = 4, k_max: int = 10) -> Tuple[int, float, np.ndarray]:
    """Try different k and pick the best silhouette score."""
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_k is None:
        km = KMeans(n_clusters=min(4, len(X)), n_init="auto", random_state=42)
        best_labels = km.fit_predict(X)
        best_k = len(set(best_labels))
        best_score = -1
    return best_k, best_score, best_labels


def pca_2d(X: np.ndarray) -> np.ndarray:
    p = PCA(n_components=2, random_state=42)
    return p.fit_transform(X)
