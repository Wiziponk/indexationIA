from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ..config import API_BASE, DATA_DIR, ID_GUESS, TITLE_COL_CANDIDATES
from ..services.preview_cache import pop_preview
from ..services.api_client import discover_fields, fetch_all_programs
from ..services.embeddings import batch_embed, build_text_from_fields
from ..services.transcripts import load_transcripts
from ..services.storage import save_dataset_files

router = APIRouter()

@router.get("/sample")
def api_sample(primary_key: str, fields: str = ""):
    fields_list = [f for f in fields.split(",") if f.strip()]
    programs = fetch_all_programs(max_pages=1)
    if not programs:
        raise HTTPException(404, "No data")
    sample = programs[0]
    def get_nested_value(d, path):
        for part in path.split('.'):
            if isinstance(d, dict):
                d = d.get(part, "")
            else:
                return ""
        return d
    out = {}
    out[primary_key] = get_nested_value(sample, primary_key)
    for f in fields_list:
        out[f] = get_nested_value(sample, f)
    return {"sample": out, "api_base": API_BASE}

@router.post("/preview")
async def generate_preview(
    mode: str = Form("api"),
    primary_key: str = Form(...),
    embed_fields: List[str] = Form(...),
    excel_token: Optional[str] = Form(None),
    excel_id_col: Optional[str] = Form(None),
    transcripts: List[UploadFile] = File(default=[])
):
    if not primary_key or not embed_fields:
        raise HTTPException(400, "Missing primary key or fields")

    # Build dataset
    try:
        programs = fetch_all_programs()
    except Exception as e:
        raise HTTPException(400, f"API error: {e}")
    if not programs:
        raise HTTPException(400, "No data returned by the API")

    wanted_ids = None
    if mode == "excel":
        info = pop_preview(excel_token or "")
        if not info or not Path(info["path"]).exists():
            raise HTTPException(400, "Uploaded Excel token expired; please re-upload")
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
            raise HTTPException(400, "Chosen ID column not found in uploaded file")
        wanted_ids = set(df_ids[excel_id_col].astype(str).str.strip().tolist())

    df = pd.DataFrame(programs)
    # nested key support
    def get_nested_value(d, path):
        for part in path.split('.'):
            if isinstance(d, dict):
                d = d.get(part, "")
            else:
                return ""
        return d
    if "." in primary_key or primary_key not in df.columns:
        df["_pk"] = df.apply(lambda r: get_nested_value(r.to_dict(), primary_key), axis=1)
        pk_col = "_pk"
        if df[pk_col].isna().all():
            raise HTTPException(400, f"Primary key '{primary_key}' not found in API data")
    else:
        pk_col = primary_key

    if wanted_ids is not None:
        df = df[df[pk_col].astype(str).str.strip().isin(wanted_ids)].copy()
        if df.empty:
            raise HTTPException(400, "None of the provided IDs matched the API data")

    transcripts_map = await load_transcripts(transcripts)
    if transcripts_map:
        df["transcript_text"] = df[pk_col].astype(str).map(transcripts_map).fillna("")

    title_col = next((c for c in TITLE_COL_CANDIDATES if c in df.columns), None)
    prev = df[[pk_col] + ([title_col] if title_col else [])].copy()
    if "transcript_text" in df.columns:
        prev["has_transcript"] = df["transcript_text"].astype(str).str.strip() != ""
    else:
        prev["has_transcript"] = False

    return {"rows": prev.head(50).to_dict(orient="records")}

@router.post("/generate")
async def run_generate(
    mode: str = Form("api"),
    primary_key: str = Form(...),
    embed_fields: List[str] = Form(...),
    excel_token: Optional[str] = Form(None),
    excel_id_col: Optional[str] = Form(None),
    transcripts: List[UploadFile] = File(default=[])
):
    if not primary_key or not embed_fields:
        raise HTTPException(400, "Missing primary key or fields")

    try:
        programs = fetch_all_programs()
    except Exception as e:
        raise HTTPException(400, f"API error: {e}")
    if not programs:
        raise HTTPException(400, "No data returned by the API")

    wanted_ids = None
    if mode == "excel":
        info = pop_preview(excel_token or "")
        if not info or not Path(info["path"]).exists():
            raise HTTPException(400, "Uploaded Excel token expired; please re-upload")
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
            raise HTTPException(400, "Chosen ID column not found in uploaded file")
        wanted_ids = set(df_ids[excel_id_col].astype(str).str.strip().tolist())

    df = pd.DataFrame(programs)
    def get_nested_value(d, path):
        for part in path.split('.'):
            if isinstance(d, dict):
                d = d.get(part, "")
            else:
                return ""
        return d
    if "." in primary_key or primary_key not in df.columns:
        df["_pk"] = df.apply(lambda r: get_nested_value(r.to_dict(), primary_key), axis=1)
        pk_col = "_pk"
        if df[pk_col].isna().all():
            raise HTTPException(400, f"Primary key '{primary_key}' not found in API data")
    else:
        pk_col = primary_key

    if wanted_ids is not None:
        df = df[df[pk_col].astype(str).str.strip().isin(wanted_ids)].copy()
        if df.empty:
            raise HTTPException(400, "None of the provided IDs matched the API data")

    transcripts_map = await load_transcripts(transcripts)
    if transcripts_map:
        df["transcript_text"] = df[pk_col].astype(str).map(transcripts_map).fillna("")
        if "transcript_text" not in embed_fields:
            embed_fields.append("transcript_text")

    texts = [build_text_from_fields(row.to_dict(), embed_fields) for _, row in df.iterrows()]
    df["_text_for_embedding"] = texts
    df = df[df["_text_for_embedding"].astype(str).str.strip() != ""].copy()
    if df.empty:
        raise HTTPException(400, "Nothing to embed after field selection. Check your fields.")

    try:
        embs = await batch_embed(df["_text_for_embedding"].tolist())
    except Exception as e:
        raise HTTPException(400, f"Embedding error: {e}")
    X = np.vstack(embs)

    uid = str(uuid.uuid4())[:8]
    raw_name, emb_name = save_dataset_files(df, X, uid)
    return {"uid": uid, "parquet": f"/api/download/{raw_name}", "embeddings": f"/api/download/{emb_name}"}
