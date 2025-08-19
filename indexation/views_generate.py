from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for

from .config import API_BASE, DATA_DIR, ID_GUESS, TITLE_COL_CANDIDATES
from .helpers import ensure_list, get_nested_value
from .services import preview_cache
from .services.api_client import discover_fields, fetch_all_programs
from .services.embedding import batch_embed, build_text_from_fields
from .services.transcripts import load_transcripts

generate_bp = Blueprint("generate", __name__)


def build_dataset(primary_key, embed_fields, mode, excel_token, excel_id_col, transcripts_files):
    """Fetch programs, filter and attach transcripts. Returns (df, pk_col)."""
    try:
        programs = fetch_all_programs()
    except Exception as e:
        raise RuntimeError(f"API error: {e}")

    if not programs:
        raise RuntimeError("No data returned by the API")

    wanted_ids = None
    if mode == "excel":
        info = preview_cache.pop_preview(excel_token or "")
        if not info or not Path(info["path"]).exists():
            raise RuntimeError("Uploaded Excel token expired; please re-upload")

        tmp = Path(info["path"])
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
        df["_pk"] = df.apply(lambda r: get_nested_value(r.to_dict(), primary_key), axis=1)
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


@generate_bp.get("/generate")
def generate_page():
    fields = discover_fields()
    return render_template("generate.html", api_base=API_BASE, fields=fields, id_guess=ID_GUESS)


@generate_bp.post("/generate/preview")
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


@generate_bp.post("/generate")
def run_generate():
    mode = request.form.get("mode", "api")
    primary_key = request.form.get("primary_key", "").strip()
    embed_fields = ensure_list(request.form.getlist("embed_fields"))
    excel_token = request.form.get("excel_token", "")
    excel_id_col = request.form.get("excel_id_col", "")

    if not primary_key:
        flash("Please choose a primary key from the API fields.")
        return redirect(url_for("generate.generate_page"))
    if not embed_fields:
        flash("Select at least one field to embed.")
        return redirect(url_for("generate.generate_page"))

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
        return redirect(url_for("generate.generate_page"))

    texts = [build_text_from_fields(row.to_dict(), embed_fields) for _, row in df.iterrows()]
    df["_text_for_embedding"] = texts
    df = df[df["_text_for_embedding"].astype(str).str.strip() != ""]
    if df.empty:
        flash("Nothing to embed after field selection. Check your embed fields.")
        return redirect(url_for("generate.generate_page"))

    try:
        embs = batch_embed(df["_text_for_embedding"].tolist())
    except Exception as e:
        flash(f"Embedding error: {e}")
        return redirect(url_for("generate.generate_page"))
    X = np.vstack(embs)

    uid = str(uuid.uuid4())[:8]
    raw_name = f"raw_{uid}.parquet"
    emb_name = f"emb_{uid}.npy"
    df.to_parquet(DATA_DIR / raw_name, index=False)
    np.save(DATA_DIR / emb_name, X)

    return render_template(
        "generate_done.html",
        uid=uid,
        parquet=url_for("common.download_result", name=raw_name),
        embeddings=url_for("common.download_result", name=emb_name),
    )
