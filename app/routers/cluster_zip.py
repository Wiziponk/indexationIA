from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Form, UploadFile, File, HTTPException

from ..services.clustering import project_points, kmeans_auto_or_k, dbscan_cluster
from ..config import DATA_DIR

router = APIRouter()


def _read_npy_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as f:
        data = f.read()
    return np.load(io.BytesIO(data), allow_pickle=False)


def _read_lines_from_zip(zf: zipfile.ZipFile, name: str) -> List[Dict[str, Any]]:
    lines = []
    with zf.open(name) as f:
        for raw in f:
            try:
                lines.append(json.loads(raw.decode("utf-8")))
            except Exception:
                pass
    return lines


@router.post("/cluster/clips")
async def cluster_clips(
    packages: List[UploadFile] = File(...),
    algo_choice: str = Form("kmeans"),  # kmeans|dbscan
    k_choice: str = Form("auto"),
    proj_choice: str = Form("pca"),
    db_eps: float = Form(0.8),
    db_min_samples: int = Form(10),
    name_clusters: bool = Form(False),
):
    """
    Cluster clip embeddings from uploaded program ZIPs (emb_clips.npy),
    keeping emission affiliation (pk).
    """
    rows = []
    for p in packages:
        with zipfile.ZipFile(io.BytesIO(await p.read())) as zf:
            try:
                emb = _read_npy_from_zip(zf, "emb_clips.npy")
            except KeyError:
                raise HTTPException(400, f"{p.filename} does not contain emb_clips.npy")
            program_meta = json.loads(zf.read("program.json").decode("utf-8"))
            pk = str(program_meta.get("pk_value"))

            meta = _read_lines_from_zip(zf, "clips_meta.jsonl")
            titles = [m.get("title") for m in meta] if meta else [None] * emb.shape[0]
            texts = [m.get("text") for m in meta] if meta else [None] * emb.shape[0]

            for i in range(emb.shape[0]):
                rows.append({"pk": pk, "idx": i + 1, "title": titles[i], "text": texts[i], "vec": emb[i, :]})

    if not rows:
        raise HTTPException(400, "No clips found in uploaded ZIPs.")

    X = np.vstack([r["vec"] for r in rows])

    if algo_choice == "kmeans":
        labels, k, sil = kmeans_auto_or_k(X, k_choice)
    else:
        labels = dbscan_cluster(X, eps=db_eps, min_samples=db_min_samples)
        k = int(len(set([l for l in labels if l != -1])))
        sil = -1.0

    proj = project_points(X, proj_choice)
    points = []
    for r, c, xy in zip(rows, labels, proj):
        points.append({
            "pk": r["pk"],
            "clip_index": r["idx"],
            "title": r["title"],
            "cluster": int(c),
            "x": float(xy[0]),
            "y": float(xy[1]),
        })

    meta = {"k": k, "silhouette": sil, "cluster_names": {}}
    if name_clusters and k > 0:
        # optional: name clusters using titles as exemplars
        try:
            from ..services.embeddings import name_clusters_via_chat  # if you add later
        except Exception:
            meta["cluster_names"] = {}
    return {"points": points, "meta": meta, "download": {}}


@router.post("/cluster/emissions")
async def cluster_emissions(
    packages: List[UploadFile] = File(...),
    algo_choice: str = Form("kmeans"),
    k_choice: str = Form("auto"),
    proj_choice: str = Form("pca"),
    db_eps: float = Form(0.8),
    db_min_samples: int = Form(10),
    name_clusters: bool = Form(False),
):
    """
    Cluster program embeddings from uploaded program ZIPs (emb_program.npy).
    """
    rows = []
    for p in packages:
        with zipfile.ZipFile(io.BytesIO(await p.read())) as zf:
            try:
                emb = _read_npy_from_zip(zf, "emb_program.npy")
            except KeyError:
                raise HTTPException(400, f"{p.filename} does not contain emb_program.npy")
            program_meta = json.loads(zf.read("program.json").decode("utf-8"))
            pk = str(program_meta.get("pk_value"))
            rows.append({"pk": pk, "vec": emb})

    if not rows:
        raise HTTPException(400, "No program embeddings found.")

    X = np.vstack([r["vec"] for r in rows])

    if algo_choice == "kmeans":
        labels, k, sil = kmeans_auto_or_k(X, k_choice)
    else:
        labels = dbscan_cluster(X, eps=db_eps, min_samples=db_min_samples)
        k = int(len(set([l for l in labels if l != -1])))
        sil = -1.0

    proj = project_points(X, proj_choice)
    points = []
    for r, c, xy in zip(rows, labels, proj):
        points.append({"pk": r["pk"], "cluster": int(c), "x": float(xy[0]), "y": float(xy[1])})

    meta = {"k": k, "silhouette": sil, "cluster_names": {}}
    return {"points": points, "meta": meta, "download": {}}
