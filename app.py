import os
import uuid
import json
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from openai import OpenAI

# -------------------
# Setup
# -------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")  # replace in prod

# -------------------
# Helpers
# -------------------
EMBED_MODEL = "text-embedding-3-small"
ID_COL_CANDIDATES = ["id", "video_id", "uid"]
TITLE_COL_CANDIDATES = ["title", "name", "programme_title"]
TEXT_COL_CANDIDATES = ["transcript_text", "transcript", "text", "summary"]
PATH_COL_CANDIDATES = ["transcript_path", "path"]

def pick_column(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def load_text_from_row(row, text_col, path_col, base_dir=None):
    # Prefer inline text
    if text_col and isinstance(row.get(text_col), str) and row[text_col].strip():
        return row[text_col]
    # Fallback: read file path
    if path_col and isinstance(row.get(path_col), str) and row[path_col].strip():
        p = row[path_col]
        if base_dir and not os.path.isabs(p):
            p = os.path.join(base_dir, p)
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""
    return ""

def batch_embed(texts, model=EMBED_MODEL):
    # OpenAI supports batching by passing a list to input
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def auto_kmeans(X, k_min=4, k_max=10):
    best_k, best_score, best_labels = None, -1, None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        # guard for tiny datasets
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_k is None:
        # fallback
        km = KMeans(n_clusters=min(4, len(X)), n_init="auto", random_state=42)
        best_labels = km.fit_predict(X)
        best_k = len(set(best_labels))
        best_score = -1
    return best_k, best_score, best_labels

def pca_2d(X):
    p = PCA(n_components=2, random_state=42)
    coords = p.fit_transform(X)
    return coords

# -------------------
# Routes
# -------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Handle upload, embedding, clustering, and prepare visualization.
    Synchronous for simplicity. For larger catalogs, consider background jobs.
    """
    file = request.files.get("catalog")
    base_dir = request.form.get("base_dir", "").strip() or None
    k_choice = request.form.get("k_choice", "auto").strip()

    if not file or file.filename == "":
        flash("Please upload a CSV or Excel file with your catalog.")
        return redirect(url_for("index"))

    # Save upload to a temp path
    uid = str(uuid.uuid4())[:8]
    in_name = f"catalog_{uid}_{file.filename}"
    in_path = DATA_DIR / in_name
    file.save(in_path)

    # Read CSV or Excel
    try:
        if in_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(in_path)
        else:
            df = pd.read_csv(in_path)
    except Exception as e:
        flash(f"Could not read file: {e}")
        return redirect(url_for("index"))

    # Identify columns
    id_col = pick_column(df, ID_COL_CANDIDATES, default=df.columns[0])
    title_col = pick_column(df, TITLE_COL_CANDIDATES, default=None)
    text_col = pick_column(df, TEXT_COL_CANDIDATES, default=None)
    path_col = pick_column(df, PATH_COL_CANDIDATES, default=None)

    # Build text column
    texts = []
    for _, row in df.iterrows():
        texts.append(load_text_from_row(row, text_col, path_col, base_dir))
    df["_text_for_embedding"] = texts
    df = df[df["_text_for_embedding"].astype(str).str.strip() != ""].copy()

    if df.empty:
        flash("No rows with transcript text found. Ensure your file has a transcript column or a transcript_path.")
        return redirect(url_for("index"))

    # Embeddings (batched for speed)
    # You can slice into chunks if you expect very large inputs; for now, one shot:
    try:
        embs = batch_embed(df["_text_for_embedding"].tolist(), model=EMBED_MODEL)
    except Exception as e:
        flash(f"Embedding error: {e}")
        return redirect(url_for("index"))

    X = np.vstack(embs)

    # Clustering
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

    # 2D projection
    coords = pca_2d(X)
    df["_cluster"] = labels
    df["_x"] = coords[:, 0]
    df["_y"] = coords[:, 1]

    # Save processed parquet and a JSON snapshot for the frontend
    out_name = f"result_{uid}.parquet"
    out_path = DATA_DIR / out_name
    df.to_parquet(out_path, index=False)

    # Build lightweight JSON payload for client chart
    payload = {
        "meta": {
            "k": int(k),
            "silhouette": float(sil),
            "points": len(df)
        },
        "points": [
            {
                "id": str(row.get(id_col, "")),
                "title": str(row.get(title_col, "")) if title_col else "",
                "cluster": int(row["_cluster"]),
                "x": float(row["_x"]),
                "y": float(row["_y"]),
            }
            for _, row in df.iterrows()
        ],
        "columns": {
            "id": id_col,
            "title": title_col,
        },
        "download": {
            "parquet": url_for("download_result", name=out_name)
        }
    }

    return render_template("results.html", payload_json=json.dumps(payload))

@app.route("/download/<name>", methods=["GET"])
def download_result(name):
    # In dev this will serve from /data via send_from_directory
    # Keep it simple: we only allow parquet files previously saved
    from flask import send_from_directory, abort
    candidate = DATA_DIR / name
    if not candidate.exists() or not candidate.name.endswith(".parquet"):
        abort(404)
    return send_from_directory(DATA_DIR, candidate.name, as_attachment=True)

# -------------------
# Entrypoint
# -------------------
if __name__ == "__main__":
    app.run(debug=True)


