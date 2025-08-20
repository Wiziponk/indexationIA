from __future__ import annotations

from typing import Tuple, Literal, List
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


# ---------------- Existing helpers (kept) ----------------

def auto_kmeans(X: np.ndarray, k_min: int = 4, k_max: int = 10) -> Tuple[int, float, np.ndarray]:
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        # need at least 2 clusters to compute silhouette
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1.0
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_k is None:
        km = KMeans(n_clusters=min(max(2, X.shape[0] // 2), 4), n_init="auto", random_state=42)
        best_labels = km.fit_predict(X)
        best_k = len(set(best_labels))
        best_score = -1.0
    return best_k, best_score, best_labels


def pca_2d(X: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=42).fit_transform(X)


def tsne_2d(X: np.ndarray) -> np.ndarray:
    return TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(X)


# ---------------- New wrappers expected by routers ----------------

Proj = Literal["pca", "tsne"]

def project_points(X: np.ndarray, proj_choice: Proj = "pca") -> np.ndarray:
    """
    2D projection dispatcher used by the routers.
    """
    if proj_choice == "tsne":
        return tsne_2d(X)
    # default
    return pca_2d(X)


def kmeans_auto_or_k(X: np.ndarray, k_choice: str = "auto") -> Tuple[np.ndarray, int, float]:
    """
    If k_choice == 'auto': run auto_kmeans.
    Else: treat k_choice as an int K and cluster with KMeans.
    Returns: (labels, k, silhouette)
    """
    if str(k_choice).strip().lower() == "auto":
        k, sil, labels = auto_kmeans(X)
        return labels, int(k), float(sil)

    try:
        k = int(k_choice)
    except Exception:
        k = 6  # sensible default
    km = KMeans(n_clusters=max(2, k), n_init="auto", random_state=42)
    labels = km.fit_predict(X)

    if len(set(labels)) >= 2:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = -1.0
    else:
        sil = -1.0

    return labels, int(k), sil


def dbscan_cluster(X: np.ndarray, eps: float = 0.8, min_samples: int = 10) -> np.ndarray:
    """
    Return labels from DBSCAN. Noise points are labeled -1.
    """
    db = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric="euclidean", n_jobs=None)
    labels = db.fit_predict(X)
    return labels
