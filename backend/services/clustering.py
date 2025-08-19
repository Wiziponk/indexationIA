from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def kmeans_auto(emb: np.ndarray, k_min: int, k_max: int) -> Tuple[np.ndarray, float, int]:
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(max(2, k_min), max(3, k_max+1)):
        km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(emb)
        labels = km.labels_
        if len(set(labels)) < 2:
            score = -1.0
        else:
            score = silhouette_score(emb, labels, metric='cosine')
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    return best_labels, best_score, best_k if best_k else max(2, k_min)

def dbscan_cluster(emb: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, Optional[float], int]:
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(emb)
    labels = db.labels_
    unique = set(labels) - {-1}
    score = None
    if len(unique) >= 2:
        score = silhouette_score(emb[labels!=-1], labels[labels!=-1], metric='cosine')
    n_clusters = len(unique)
    return labels, score, n_clusters

def project_2d(emb: np.ndarray) -> np.ndarray:
    p = PCA(n_components=2, random_state=42).fit_transform(emb)
    return p
