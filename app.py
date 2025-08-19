import json
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   url_for)
from sklearn.cluster import DBSCAN, KMeans

from indexation.config import (API_BASE, DATA_DIR, ID_GUESS, SECRET_KEY,
                               TITLE_COL_CANDIDATES)
from indexation.helpers import (auto_kmeans, batch_embed,
                                build_text_from_fields, discover_fields,
                                ensure_list, fetch_all_programs,
                                get_nested_value, load_transcripts, pca_2d,
                                suggest_cluster_names)

load_dotenv()

app = Flask(__name__)
app.secret_key = SECRET_KEY

# token -> {"path": Path, "ts": float}
PREVIEW_INDEX = {}


def cleanup_previews(max_age=3600):
    """Remove preview files older than max_age seconds."""
    now = time.time()
    for token, info in list(PREVIEW_INDEX.items()):
        if now - info["ts"] > max_age:
            try:
                info["path"].unlink(missing_ok=True)
            except Exception:
                pass
            PREVIEW_INDEX.pop(token, None)


# -----------------------------------------------------
# Landing and common API endpoints
# -----------------------------------------------------


@app.get("/healthz")
def healthz():
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        sha = "unknown"
    return {"ok": True, "version": sha}


@app.get("/")
def landing():
    return render_template("home.html")


@app.get("/api/fields")
def api_fields():
    fields = discover_fields()
    return jsonify(fields=fields, api_base=API_BASE)


@app.get("/api/sample")
def api_sample():
    pk = request.args.get("primary_key", "").strip()
    fields = ensure_list(request.args.get("fields", "").split(","))
    try:
        programs = fetch_all_programs(max_pages=1)
    except Exception as e:
        return jsonify(error=str(e)), 500
    if not programs:
        return jsonify(error="No data"), 404
    sample = programs[0]
    out = {}
    if pk:
        out[pk] = get_nested_value(sample, pk)
    for f in fields:
        out[f] = get_nested_value(sample, f)
    return jsonify(sample=out)


# -----------------------------------------------------
# Generation
# -----------------------------------------------------


def build_dataset(
    primary_key, embed_fields, mode, excel_token, excel_id_col, transcripts_files
):
    """Fetch programs, filter and attach transcripts. Returns (df, pk_col)."""
    try:
        programs = fetch_all_programs()
    except Exception as e:
        raise RuntimeError(f"API error: {e}")

    if not programs:
        raise RuntimeError("No data returned by the API")

    wanted_ids = None
    if mode == "excel":
        info = PREVIEW_INDEX.pop(excel_token or "", None)
        if not info or not info["path"].exists():
            raise RuntimeError("Uploaded Excel token expired; please re-upload")

        tmp = info["path"]
        if tmp.suffix.lower() in [".xlsx", ".xls"]:
            df_ids = pd.read_excel(tmp)
        else:
            df_ids = pd.read_csv(tmp)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        if excel_id_col not in df_ids.columns:
            raise RuntimeError("Chosen ID column not found in uploaded file")
        wanted_ids = set(df_ids[excel_id_col].astype(str).str.strip().tolist())

    df = pd.DataFrame(programs)
    if "." in primary_key or primary_key not in df.columns:
        df["_pk"] = df.apply(
            lambda r: get_nested_value(r.to_dict(), primary_key), axis=1
        )
        pk_col = "_pk"
        if df[pk_col].isna().all():
            raise RuntimeError(f"Primary key '{primary_key}' not found in API data")
    else:
        pk_col = primary_key

    if wanted_ids is not None:
        df = df[df[pk_col].astype(str).str.strip().isin(wanted_ids)].copy()
        if df.empty:
            raise RuntimeError("None of the provided IDs matched the API data")

    transcripts_map = load_transcripts(transcripts_files)
    if transcripts_map:
        df["transcript_text"] = df[pk_col].astype(str).map(transcripts_map).fillna("")
        if "transcript_text" not in embed_fields:
            embed_fields.append("transcript_text")

    return df, pk_col


@app.get("/generate")
def generate_page():
    fields = discover_fields()
    return render_template(
        "generate.html", api_base=API_BASE, fields=fields, id_guess=ID_GUESS
    )


@app.post("/generate/preview")
def generate_preview():
    mode = request.form.get("mode", "api")
    primary_key = request.form.get("primary_key", "").strip()
    embed_fields = ensure_list(request.form.getlist("embed_fields"))
    excel_token = request.form.get("excel_token", "")
    excel_id_col = request.form.get("excel_id_col", "")

    if not primary_key or not embed_fields:
        return jsonify(error="Missing primary key or fields"), 400

    try:
        df, pk_col = build_dataset(
            primary_key,
            embed_fields.copy(),
            mode,
            excel_token,
            excel_id_col,
            request.files.getlist("transcripts"),
        )
    except RuntimeError as e:
        return jsonify(error=str(e)), 400

    title_col = next((c for c in TITLE_COL_CANDIDATES if c in df.columns), None)
    prev = df[[pk_col] + ([title_col] if title_col else [])].copy()
    prev["has_transcript"] = df.get("transcript_text", "").astype(str).str.strip() != ""
    return jsonify(rows=prev.head(50).to_dict(orient="records"))


@app.post("/generate")
def run_generate():
    mode = request.form.get("mode", "api")
    primary_key = request.form.get("primary_key", "").strip()
    embed_fields = ensure_list(request.form.getlist("embed_fields"))
    excel_token = request.form.get("excel_token", "")
    excel_id_col = request.form.get("excel_id_col", "")

    if not primary_key:
        flash("Please choose a primary key from the API fields.")
        return redirect(url_for("generate_page"))
    if not embed_fields:
        flash("Select at least one field to embed.")
        return redirect(url_for("generate_page"))

    try:
        df, pk_col = build_dataset(
            primary_key,
            embed_fields,
            mode,
            excel_token,
            excel_id_col,
            request.files.getlist("transcripts"),
        )
    except RuntimeError as e:
        flash(str(e))
        return redirect(url_for("generate_page"))

    texts = [
        build_text_from_fields(row.to_dict(), embed_fields) for _, row in df.iterrows()
    ]
    df["_text_for_embedding"] = texts
    df = df[df["_text_for_embedding"].astype(str).str.strip() != ""]
    if df.empty:
        flash("Nothing to embed after field selection. Check your embed fields.")
        return redirect(url_for("generate_page"))

    try:
        embs = batch_embed(df["_text_for_embedding"].tolist())
    except Exception as e:
        flash(f"Embedding error: {e}")
        return redirect(url_for("generate_page"))
    X = np.vstack(embs)

    uid = str(uuid.uuid4())[:8]
    raw_name = f"raw_{uid}.parquet"
    emb_name = f"emb_{uid}.npy"
    df.to_parquet(DATA_DIR / raw_name, index=False)
    np.save(DATA_DIR / emb_name, X)

    return render_template(
        "generate_done.html",
        uid=uid,
        parquet=url_for("download_result", name=raw_name),
        embeddings=url_for("download_result", name=emb_name),
    )


# -----------------------------------------------------
# Clustering
# -----------------------------------------------------


@app.get("/cluster")
def cluster_page():
    existing = []
    for p in DATA_DIR.glob("raw_*.parquet"):
        uid = p.stem.split("_", 1)[1]
        emb = DATA_DIR / f"emb_{uid}.npy"
        if emb.exists():
            existing.append({"uid": uid, "parquet": p.name, "emb": emb.name})
    preselect = request.args.get("uid", "")
    return render_template("cluster.html", existing=existing, preselect=preselect)


@app.post("/cluster")
def run_cluster():
    existing_uid = request.form.get("existing_uid", "").strip()
    k_choice = request.form.get("k_choice", "auto").strip()
    algo_choice = request.form.get("algo_choice", "kmeans").strip()

    if existing_uid:
        df_path = DATA_DIR / f"raw_{existing_uid}.parquet"
        emb_path = DATA_DIR / f"emb_{existing_uid}.npy"
        if not df_path.exists() or not emb_path.exists():
            flash("Selected dataset not found.")
            return redirect(url_for("cluster_page"))
    else:
        df_file = request.files.get("dataset_file")
        emb_file = request.files.get("embedding_file")
        if not df_file or not emb_file:
            flash("Please provide dataset and embedding files.")
            return redirect(url_for("cluster_page"))
        uid = uuid.uuid4().hex[:8]
        df_path = DATA_DIR / f"upload_{uid}.parquet"
        emb_path = DATA_DIR / f"upload_{uid}.npy"
        df_file.save(df_path)
        emb_file.save(emb_path)

    df = pd.read_parquet(df_path)
    X = np.load(emb_path)
    if X.shape[0] != len(df):
        flash("Embedding file size mismatch with dataset.")
        return redirect(url_for("cluster_page"))

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
            "parquet": url_for("download_result", name=out_name),
            "embeddings": url_for("download_result", name=emb_path.name),
        },
    }

    return render_template("results.html", payload_json=json.dumps(payload))


# -----------------------------------------------------
# Utilities
# -----------------------------------------------------


@app.post("/preview-excel")
def preview_excel():
    cleanup_previews()
    f = request.files.get("excel")
    if not f or f.filename == "":
        return jsonify(error="No file uploaded"), 400
    token = uuid.uuid4().hex
    tmp = DATA_DIR / f"preview_{token}{Path(f.filename).suffix}"
    f.save(tmp)
    try:
        if tmp.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(tmp)
        else:
            df = pd.read_csv(tmp)
    except Exception as e:
        return jsonify(error=f"Cannot read file: {e}"), 400

    preview = df.head(50)
    cols = list(df.columns)
    PREVIEW_INDEX[token] = {"path": tmp, "ts": time.time()}
    return jsonify(columns=cols, preview=preview.to_dict(orient="records"), token=token)


@app.get("/download/<name>")
def download_result(name):
    from flask import abort, send_from_directory

    candidate = DATA_DIR / name
    if not candidate.exists() or not candidate.name.endswith(
        tuple([".parquet", ".npy"])
    ):
        abort(404)
    return send_from_directory(DATA_DIR, candidate.name, as_attachment=True)
