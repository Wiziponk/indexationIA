import uuid
import json
from pathlib import Path

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from sklearn.cluster import KMeans

from indexation.config import (
    API_BASE,
    DATA_DIR,
    TITLE_COL_CANDIDATES,
    ID_GUESS,
    SECRET_KEY,
)
from indexation.helpers import (
    discover_fields,
    ensure_list,
    fetch_all_programs,
    build_text_from_fields,
    batch_embed,
    auto_kmeans,
    pca_2d,
)

app = Flask(__name__)
app.secret_key = SECRET_KEY


@app.get("/")
def home():
    fields = discover_fields()
    return render_template("index.html", api_base=API_BASE, fields=fields, id_guess=ID_GUESS)


@app.get("/api/fields")
def api_fields():
    fields = discover_fields()
    return jsonify(fields=fields, api_base=API_BASE)


@app.post("/preview-excel")
def preview_excel():
    f = request.files.get("excel")
    if not f or f.filename == "":
        return jsonify(error="No file uploaded"), 400
    tmp = DATA_DIR / f"preview_{uuid.uuid4().hex}{Path(f.filename).suffix}"
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
    return jsonify(
        columns=cols,
        preview=preview.to_dict(orient="records"),
        token=str(tmp.name),
    )


@app.post("/ingest")
def ingest():
    mode = request.form.get("mode", "api")
    primary_key = request.form.get("primary_key", "").strip()
    embed_fields = ensure_list(request.form.getlist("embed_fields"))
    k_choice = request.form.get("k_choice", "auto").strip()

    if not primary_key:
        flash("Please choose a primary key from the API fields.")
        return redirect(url_for("home"))
    if not embed_fields:
        flash("Select at least one field to embed.")
        return redirect(url_for("home"))

    try:
        programs = fetch_all_programs()
    except Exception as e:
        flash(f"API error: {e}")
        return redirect(url_for("home"))

    if not programs:
        flash("No data returned by the API.")
        return redirect(url_for("home"))

    wanted_ids = None
    if mode == "excel":
        token = request.form.get("excel_token", "")
        excel_id_col = request.form.get("excel_id_col", "")
        if not token or not excel_id_col:
            flash("Excel filter selected but missing file preview token or ID column.")
            return redirect(url_for("home"))

        tmp = Path(token)
        if not tmp.exists():
            flash("Uploaded Excel token expired; please re-upload.")
            return redirect(url_for("home"))

        if tmp.suffix.lower() in [".xlsx", ".xls"]:
            df_ids = pd.read_excel(tmp)
        else:
            df_ids = pd.read_csv(tmp)

        if excel_id_col not in df_ids.columns:
            flash("Chosen ID column not found in uploaded file.")
            return redirect(url_for("home"))

        wanted_ids = set(df_ids[excel_id_col].astype(str).str.strip().tolist())

    df = pd.DataFrame(programs)
    if primary_key not in df.columns:
        flash(f"Primary key '{primary_key}' not found in API data.")
        return redirect(url_for("home"))

    if wanted_ids is not None:
        df = df[df[primary_key].astype(str).str.strip().isin(wanted_ids)].copy()
        if df.empty:
            flash("None of the provided IDs matched the API data.")
            return redirect(url_for("home"))

    texts = []
    for _, row in df.iterrows():
        texts.append(build_text_from_fields(row.to_dict(), embed_fields))
    df["_text_for_embedding"] = texts
    df = df[df["_text_for_embedding"].astype(str).str.strip() != ""]
    if df.empty:
        flash("Nothing to embed after field selection. Check your embed fields.")
        return redirect(url_for("home"))

    try:
        embs = batch_embed(df["_text_for_embedding"].tolist())
    except Exception as e:
        flash(f"Embedding error: {e}")
        return redirect(url_for("home"))
    X = np.vstack(embs)

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

    uid = str(uuid.uuid4())[:8]
    out_name = f"result_{uid}.parquet"
    out_path = DATA_DIR / out_name
    df.to_parquet(out_path, index=False)

    title_col = next((c for c in TITLE_COL_CANDIDATES if c in df.columns), None)

    payload = {
        "meta": {
            "k": int(k),
            "silhouette": float(sil),
            "points": int(len(df)),
            "primary_key": primary_key,
            "embed_fields": embed_fields,
            "mode": mode,
        },
        "points": [
            {
                "id": str(row[primary_key]),
                "title": str(row[title_col]) if title_col else "",
                "cluster": int(row["_cluster"]),
                "x": float(row["_x"]),
                "y": float(row["_y"]),
            }
            for _, row in df.iterrows()
        ],
        "columns": {"id": primary_key, "title": title_col},
        "download": {"parquet": url_for("download_result", name=out_name)},
    }

    return render_template("results.html", payload_json=json.dumps(payload))


@app.get("/download/<name>")
def download_result(name):
    from flask import send_from_directory, abort

    candidate = DATA_DIR / name
    if not candidate.exists() or not candidate.name.endswith(".parquet"):
        abort(404)
    return send_from_directory(DATA_DIR, candidate.name, as_attachment=True)
