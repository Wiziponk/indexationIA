import os, io, json, time, uuid
import numpy as np
import pandas as pd
from rq import get_current_job
from typing import Dict, Any, List
from .config import settings
from .services.api_client import fetch_programs, fields_from_records
from .services.embeddings import build_texts, embed_texts
from .services.clustering import kmeans_auto, dbscan_cluster, project_2d
from .services.utils import ensure_dir

DATA_DIR = settings.DATA_DIR

def _set_progress(p: float, msg: str):
    job = get_current_job()
    if job:
        job.meta["progress"] = p
        job.meta["message"] = msg
        job.save_meta()

def generate_task(params: Dict[str, Any]) -> Dict[str, Any]:
    _set_progress(0.0, "Fetching records")
    base = params.get("base")
    primary_key = params["primary_key"]
    fields = params["fields"]

    records = get_records_with_optional_excel(base, primary_key, params.get("excel_token"))
    if not records:
        raise RuntimeError("No records fetched")

    _set_progress(0.2, "Building texts")
    texts = build_texts(records, fields)

    _set_progress(0.4, "Embedding")
    emb = embed_texts(texts)
    uid = uuid.uuid4().hex[:8]
    raw_path = os.path.join(DATA_DIR, f"raw_{uid}.parquet")
    emb_path = os.path.join(DATA_DIR, f"emb_{uid}.npy")
    ensure_dir(DATA_DIR)

    _set_progress(0.8, "Saving artifacts")
    df = pd.DataFrame(records)
    df.to_parquet(raw_path, index=False)
    np.save(emb_path, emb)

    _set_progress(1.0, "Done")
    return {"raw_path": raw_path, "emb_path": emb_path, "count": len(records)}

def get_records_with_optional_excel(base: str, primary_key: str, excel_token: str = None):
    # For brevity: just fetch programs; if excel token exists, filter ids
    import json, os
    from .services.storage import load_preview_token
    temp_dir = os.path.join(settings.DATA_DIR, "_tmp")
    records =  __import__("asyncio").get_event_loop().run_until_complete(fetch_programs(base, settings.MAX_ITEMS))
    if excel_token:
        payload = load_preview_token(temp_dir, excel_token)
        ids = [str(r.get(primary_key)) for r in payload.get("head", []) if primary_key in r]
        idset = set(ids)
        records = [r for r in records if str(r.get(primary_key)) in idset]
    return records

def cluster_task(params: Dict[str, Any]) -> Dict[str, Any]:
    _set_progress(0.0, "Loading data")
    raw_path = params["raw_path"]
    emb_path = params["emb_path"]
    algo = params.get("algorithm", "kmeans")

    emb = np.load(emb_path)
    _set_progress(0.2, "Clustering")
    if algo == "kmeans":
        labels, score, k = kmeans_auto(emb, params.get("k_min", 6), params.get("k_max", 12))
        n_clusters = len(set(labels))
    else:
        labels, score, n_clusters = dbscan_cluster(emb, params.get("eps", 0.5), params.get("min_samples", 5))

    _set_progress(0.6, "Projecting 2D")
    xy = project_2d(emb)

    _set_progress(0.8, "Saving result")
    import pandas as pd
    df = pd.read_parquet(raw_path)
    df["_cluster"] = labels
    df["_x"] = xy[:,0]
    df["_y"] = xy[:,1]

    uid = os.path.basename(emb_path).split("_",1)[1].split(".")[0]
    result_path = os.path.join(settings.DATA_DIR, f"result_{uid}.parquet")
    df.to_parquet(result_path, index=False)

    _set_progress(1.0, "Done")
    return {"result_path": result_path, "silhouette": (None if score is None else float(score)), "n_clusters": int(n_clusters)}
