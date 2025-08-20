from __future__ import annotations

import json
import uuid
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Form, UploadFile, File, HTTPException

from ..config import DATA_DIR, TITLE_COL_CANDIDATES, PROJECTION_DEFAULT
from ..services.clustering import auto_kmeans, pca_2d, tsne_2d
from ..services.embeddings import suggest_cluster_names  # reuse helper for naming

router = APIRouter()

@router.post("/cluster")
async def run_cluster(
    existing_uid: Optional[str] = Form(None),
    k_choice: str = Form("auto"),
    algo_choice: str = Form("kmeans"),
    proj_choice: str = Form(PROJECTION_DEFAULT),
    db_eps: float = Form(0.8),
    db_min_samples: int = Form(10),
    dataset_file: UploadFile = File(default=None),
    embedding_file: UploadFile = File(default=None),
):
    if existing_uid:
        df_path = DATA_DIR / f"raw_{existing_uid}.parquet"
        emb_path = DATA_DIR / f"emb_{existing_uid}.npy"
        if not df_path.exists() or not emb_path.exists():
            raise HTTPException(400, "Selected dataset not found.")
    else:
        if not dataset_file or not embedding_file:
            raise HTTPException(400, "Provide dataset (.parquet) and embeddings (.npy).")
        uid = uuid.uuid4().hex[:8]
        df_path = DATA_DIR / f"upload_{uid}.parquet"
        emb_path = DATA_DIR / f"upload_{uid}.npy"
        df_path.write_bytes(await dataset_file.read())
        emb_path.write_bytes(await embedding_file.read())

    df = pd.read_parquet(df_path)
    X = np.load(emb_path)
    if X.shape[0] != len(df):
        raise HTTPException(400, "Embedding size mismatch with dataset.")

    # Clustering
    if algo_choice == "dbscan":
        from sklearn.cluster import DBSCAN
        try:
            db = DBSCAN(eps=float(db_eps), min_samples=int(db_min_samples))
        except Exception:
            db = DBSCAN()
        labels = db.fit_predict(X)
        k = len(set(labels)) - (1 if -1 in labels else 0)
        sil = -1.0
    else:
        if k_choice == "auto":
            k, sil, labels = auto_kmeans(X)
        else:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            try:
                k = int(k_choice)
                km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = km.fit_predict(X)
                sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0
            except Exception:
                k, sil, labels = auto_kmeans(X)

    # Projection
    coords = tsne_2d(X) if proj_choice == "tsne" else pca_2d(X)
    df["_cluster"] = labels
    df["_x"] = coords[:, 0]
    df["_y"] = coords[:, 1]

    uid = existing_uid or uuid.uuid4().hex[:8]
    out_name = f"result_{uid}.parquet"
    df.to_parquet(DATA_DIR / out_name, index=False)

    title_col = next((c for c in TITLE_COL_CANDIDATES if c in df.columns), None)
    cluster_names = suggest_cluster_names(df, title_col) if title_col else {}

    payload = {
        "meta": {
            "k": int(k),
            "silhouette": float(sil),
            "points": int(len(df)),
            "algo": algo_choice,
            "projection": proj_choice,
            "cluster_names": cluster_names,
        },
        "points": [
            {
                "id": str(row.get("id", idx)),
                "title": str(row[title_col]) if title_col else "",
                "cluster": int(row["_cluster"]),
                "x": float(row["_x"]),
                "y": float(row["_y"]),
            }
            for idx, row in df.iterrows()
        ],
        "download": {"parquet": f"/api/download/{out_name}", "embeddings": f"/api/download/{emb_path.name}"},
    }
    return payload
