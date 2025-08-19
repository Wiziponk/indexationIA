from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd
from flask import Blueprint, flash, redirect, render_template, request, url_for
from sklearn.cluster import DBSCAN, KMeans

from .config import DATA_DIR, TITLE_COL_CANDIDATES
from .helpers import suggest_cluster_names
from .services.clustering import auto_kmeans, pca_2d

cluster_bp = Blueprint("cluster", __name__)


@cluster_bp.get("/cluster")
def cluster_page():
    existing = []
    for p in DATA_DIR.glob("raw_*.parquet"):
        uid = p.stem.split("_", 1)[1]
        emb = DATA_DIR / f"emb_{uid}.npy"
        if emb.exists():
            existing.append({"uid": uid, "parquet": p.name, "emb": emb.name})
    preselect = request.args.get("uid", "")
    return render_template("cluster.html", existing=existing, preselect=preselect)


@cluster_bp.post("/cluster")
def run_cluster():
    existing_uid = request.form.get("existing_uid", "").strip()
    k_choice = request.form.get("k_choice", "auto").strip()
    algo_choice = request.form.get("algo_choice", "kmeans").strip()

    if existing_uid:
        df_path = DATA_DIR / f"raw_{existing_uid}.parquet"
        emb_path = DATA_DIR / f"emb_{existing_uid}.npy"
        if not df_path.exists() or not emb_path.exists():
            flash("Selected dataset not found.")
            return redirect(url_for("cluster.cluster_page"))
    else:
        df_file = request.files.get("dataset_file")
        emb_file = request.files.get("embedding_file")
        if not df_file or not emb_file:
            flash("Please provide dataset and embedding files.")
            return redirect(url_for("cluster.cluster_page"))
        uid = uuid.uuid4().hex[:8]
        df_path = DATA_DIR / f"upload_{uid}.parquet"
        emb_path = DATA_DIR / f"upload_{uid}.npy"
        df_file.save(df_path)
        emb_file.save(emb_path)

    df = pd.read_parquet(df_path)
    X = np.load(emb_path)
    if X.shape[0] != len(df):
        flash("Embedding file size mismatch with dataset.")
        return redirect(url_for("cluster.cluster_page"))

    if algo_choice == "dbscan":
        db = DBSCAN()
        labels = db.fit_predict(X)
        k = len(set(labels)) - (1 if -1 in labels else 0)
        sil = -1
    else:
        if k_choice == "auto":
            k, sil, labels = auto_kmeans(X)
        else:
            try:
                k = int(k_choice)
                km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = km.fit_predict(X)
                sil = -1
            except Exception:
                k, sil, labels = auto_kmeans(X)

    coords = pca_2d(X)
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
            "cluster_names": cluster_names,
        },
        "points": [
            {
                "id": str(row.get("id", row.index)),
                "title": str(row[title_col]) if title_col else "",
                "cluster": int(row["_cluster"]),
                "x": float(row["_x"]),
                "y": float(row["_y"]),
            }
            for _, row in df.iterrows()
        ],
        "columns": {"title": title_col},
        "download": {
            "parquet": url_for("common.download_result", name=out_name),
            "embeddings": url_for("common.download_result", name=emb_path.name),
        },
    }

    return render_template("results.html", payload_json=json.dumps(payload))
