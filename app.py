import os
import uuid
import json
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from openai import OpenAI
import requests

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

# --------- CONFIG (change if needed) ----------
API_BASE = os.getenv("EDUC_API_BASE", "https://educ.arte.tv/api/list/programs")
API_TOKEN = os.getenv("EDUC_API_TOKEN", None)  # if your API requires auth
TIMEOUT = 20

EMBED_MODEL = "text-embedding-3-small"

# column heuristics for titles/text
TITLE_COL_CANDIDATES = ["title", "name", "programme_title", "titre"]
TEXT_COL_CANDIDATES = ["transcript_text", "transcript", "text", "summary", "synopsis", "description"]
ID_GUESS = ["code_emission", "codeEmission", "id", "video_id", "uid"]


# -------------------
# Helpers
# -------------------
def auth_headers():
    h = {"Accept": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h

def fetch_all_programs(max_pages=50, page_param="page", start_page=1):
    """
    Very defensive fetcher:
    - Tries simple GET on API_BASE
    - If 'next' is present in JSON, follow it
    - Else, try page pagination ?page=2,3...
    """
    items = []
    url = API_BASE
    seen_urls = set()
    for i in range(max_pages):
        if not url or url in seen_urls:
            break
        seen_urls.add(url)
        r = requests.get(url, headers=auth_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            # common shapes
            if "results" in data and isinstance(data["results"], list):
                items.extend(data["results"])
            elif "data" in data and isinstance(data["data"], list):
                items.extend(data["data"])
            elif "items" in data and isinstance(data["items"], list):
                items.extend(data["items"])
            elif isinstance(data.get("programs"), list):
                items.extend(data["programs"])
            else:
                # maybe the dict itself is a program? (unlikely) or wrong shape
                pass

            # try cursor-style
            next_url = data.get("next") or data.get("links", {}).get("next")
            if next_url:
                url = next_url
                continue

            # if no next, try numeric pagination
            # detect current page and build next
            if page_param in (url.split("?")[-1] if "?" in url else ""):
                # if page already in URL, increment naïvely
                # find current page int
                import re
                import urllib.parse as up
                parsed = up.urlparse(url)
                qs = dict(up.parse_qsl(parsed.query))
                try:
                    cur = int(qs.get(page_param, start_page))
                    qs[page_param] = str(cur + 1)
                    url = parsed._replace(query=up.urlencode(qs)).geturl()
                    # basic stop condition: if no new items for several loops, break
                    continue
                except Exception:
                    pass

            # final fallback: try adding ?page=2,3...
            if "?" in url:
                url = url + f"&{page_param}={start_page+1}"
            else:
                url = url + f"?{page_param}={start_page+1}"
            start_page += 1
            continue

        elif isinstance(data, list):
            items.extend(data)
            break  # assume no pagination if array

        else:
            break

    # de-duplicate by dict identity best-effort
    # (optional)
    return items

def discover_fields():
    try:
        r = requests.get(API_BASE, headers=auth_headers(), timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        # Take first item whatever the shape is
        if isinstance(data, dict):
            sample = None
            for key in ["results", "data", "items", "programs"]:
                if isinstance(data.get(key), list) and data[key]:
                    sample = data[key][0]
                    break
            if sample is None and len(data.keys()) > 0:
                # maybe it's a dict program (edge case)
                sample = data
        elif isinstance(data, list) and data:
            sample = data[0]
        else:
            sample = {}

        if not isinstance(sample, dict):
            return []

        fields = sorted(sample.keys())
        return fields
    except Exception as e:
        return []

def ensure_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    return [val]

def build_text_from_fields(row, fields):
    # concatenate selected fields into one string for embedding
    parts = []
    for f in fields:
        v = row.get(f, "")
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        parts.append(str(v))
    return "\n".join([p for p in parts if p and p.strip()])

def batch_embed(texts, model=EMBED_MODEL):
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def auto_kmeans(X, k_min=4, k_max=10):
    best_k, best_score, best_labels = None, -1, None
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

def pca_2d(X):
    p = PCA(n_components=2, random_state=42)
    return p.fit_transform(X)

# -------------------
# Routes
# -------------------
@app.get("/")
def home():
    # pre-populate field guesses for the UI
    fields = discover_fields()
    return render_template("index.html", api_base=API_BASE, fields=fields, id_guess=ID_GUESS)

@app.get("/api/fields")
def api_fields():
    fields = discover_fields()
    return jsonify(fields=fields, api_base=API_BASE)

@app.post("/preview-excel")
def preview_excel():
    """Upload Excel/CSV and return first rows + detected columns for mapping."""
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

    # limit preview to 50 rows
    preview = df.head(50)
    cols = list(df.columns)
    # stash the temp filename to reuse on ingest
    return jsonify(
        columns=cols,
        preview=preview.to_dict(orient="records"),
        token=str(tmp.name)  # return absolute path (kept server-side)
    )

@app.post("/ingest")
def ingest():
    """
    One endpoint for both modes:
    - mode=api : fetch all programs from API
    - mode=excel : upload token of previous preview + selected id column, fetch all, then filter
    Required form fields:
      - primary_key: the field used as unique id in the API objects (e.g., code_emission)
      - embed_fields[]: list of fields to concatenate for embedding
      - k_choice: "auto" or an int
      - excel_token (if mode=excel)
      - excel_id_col (if mode=excel) -> the column in the Excel containing the IDs to fetch
    """
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

    # 1) Fetch data from API
    try:
        programs = fetch_all_programs()
    except Exception as e:
        flash(f"API error: {e}")
        return redirect(url_for("home"))

    if not programs:
        flash("No data returned by the API.")
        return redirect(url_for("home"))

    # 2) Optionally filter by Excel list of IDs
    wanted_ids = None
    if mode == "excel":
        token = request.form.get("excel_token", "")
        excel_id_col = request.form.get("excel_id_col", "")
        if not token or not excel_id_col:
            flash("Excel filter selected but missing file preview token or ID column.")
            return redirect(url_for("home"))

        # Read back the previously uploaded file
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

    # Normalize to DataFrame
    df = pd.DataFrame(programs)
    if primary_key not in df.columns:
        flash(f"Primary key '{primary_key}' not found in API data.")
        return redirect(url_for("home"))

    if wanted_ids is not None:
        df = df[df[primary_key].astype(str).str.strip().isin(wanted_ids)].copy()
        if df.empty:
            flash("None of the provided IDs matched the API data.")
            return redirect(url_for("home"))

    # 3) Build text to embed from selected fields
    texts = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        texts.append(build_text_from_fields(row_dict, embed_fields))
    df["_text_for_embedding"] = texts
    df = df[df["_text_for_embedding"].astype(str).str.strip() != ""]
    if df.empty:
        flash("Nothing to embed after field selection. Check your embed fields.")
        return redirect(url_for("home"))

    # 4) Embeddings
    try:
        embs = batch_embed(df["_text_for_embedding"].tolist())
    except Exception as e:
        flash(f"Embedding error: {e}")
        return redirect(url_for("home"))
    X = np.vstack(embs)

    # 5) Clustering
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

    # 6) 2D projection
    coords = pca_2d(X)
    df["_cluster"] = labels
    df["_x"] = coords[:, 0]
    df["_y"] = coords[:, 1]

    # 7) Save parquet + build payload
    uid = str(uuid.uuid4())[:8]
    out_name = f"result_{uid}.parquet"
    out_path = DATA_DIR / out_name
    df.to_parquet(out_path, index=False)

    # Guess a title field if exists
    title_col = next((c for c in TITLE_COL_CANDIDATES if c in df.columns), None)

    payload = {
        "meta": {
            "k": int(k),
            "silhouette": float(sil),
            "points": int(len(df)),
            "primary_key": primary_key,
            "embed_fields": embed_fields,
            "mode": mode
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
        "columns": {
            "id": primary_key,
            "title": title_col,
        },
        "download": {
            "parquet": url_for("download_result", name=out_name)
        }
    }

    return render_template("results.html", payload_json=json.dumps(payload))

@app.get("/download/<name>")
def download_result(name):
    from flask import send_from_directory, abort
    candidate = DATA_DIR / name
    if not candidate.exists() or not candidate.name.endswith(".parquet"):
        abort(404)
    return send_from_directory(DATA_DIR, candidate.name, as_attachment=True)
