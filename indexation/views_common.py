from __future__ import annotations

from pathlib import Path
import subprocess
from flask import Blueprint, jsonify, render_template, request
import pandas as pd

import uuid

from .config import API_BASE, DATA_DIR
from .helpers import ensure_list, get_nested_value
from .services import preview_cache
from .services.api_client import discover_fields, fetch_all_programs

common_bp = Blueprint("common", __name__)


@common_bp.get("/healthz")
def healthz():
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        )
    except Exception:
        sha = "unknown"
    return {"ok": True, "version": sha}


@common_bp.get("/")
def landing():
    return render_template("home.html")


@common_bp.get("/api/fields")
def api_fields():
    fields = discover_fields()
    return jsonify(fields=fields, api_base=API_BASE)


@common_bp.get("/api/sample")
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


@common_bp.post("/preview-excel")
def preview_excel():
    preview_cache.cleanup_previews()
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
    preview_cache.register_preview(token, tmp)
    return jsonify(columns=cols, preview=preview.to_dict(orient="records"), token=token)


@common_bp.get("/download/<name>")
def download_result(name):
    from flask import abort, send_from_directory

    candidate = DATA_DIR / name
    if not candidate.exists() or not candidate.name.endswith(tuple([".parquet", ".npy"])):
        abort(404)
    return send_from_directory(DATA_DIR, candidate.name, as_attachment=True)
