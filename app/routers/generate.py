from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Form, UploadFile, File, HTTPException

from ..config import API_BASE, DATA_DIR
from ..services.preview_cache import pop_preview, get_preview
from ..services.api_client import fetch_all_programs
from ..services.embeddings import batch_embed, build_text_from_fields
from ..services.transcripts import load_transcripts
from ..services.storage import save_dataset_files
from ..services.utils import get_nested_value

router = APIRouter()


def _read_ids_frame(tmp: Path) -> pd.DataFrame:
    if tmp.suffix.lower() in [".xlsx", ".xls"]:
        df_ids = pd.read_excel(tmp)
    else:
        df_ids = pd.read_csv(tmp)
    df_ids.columns = [str(c).strip() for c in df_ids.columns]
    return df_ids


def _resolve_id_column(df_ids: pd.DataFrame, excel_id_col: Optional[str]) -> str:
    if excel_id_col and excel_id_col in df_ids.columns:
        return excel_id_col
    for guess in ["id", "ID", "Id", "code_emission", "codeEmission", "video_id", "uid"]:
        if guess in df_ids.columns:
            return guess
    for col in df_ids.columns:
        if not df_ids[col].isna().all():
            return col
    return df_ids.columns[0]


def _assert_embed_fields_ok(embed_fields: List[str]) -> None:
    if not embed_fields or not any(str(f).strip() for f in embed_fields):
        raise HTTPException(400, "Please select at least one API field to embed.")


def _assert_api_base_configured() -> None:
    if not API_BASE or not str(API_BASE).strip():
        raise HTTPException(
            400,
            "EDUC_API_BASE is not configured on the server. "
            "Set it (and any token) and reload the app."
        )


@router.post("/preview")
async def generate_preview(
    mode: str = Form("api"),
    primary_key: str = Form(...),
    embed_fields: List[str] = Form(...),
    excel_token: Optional[str] = Form(None),
    excel_id_col: Optional[str] = Form(None),
    transcripts: List[UploadFile] = File(default=[]),
):
    if not primary_key or not str(primary_key).strip():
        raise HTTPException(400, "Please choose a Primary key.")
    _assert_embed_fields_ok(embed_fields)
    _assert_api_base_configured()

    try:
        programs = fetch_all_programs()
    except Exception as e:
        raise HTTPException(400, f"API error while fetching programs: {e}")
    if not programs:
        raise HTTPException(400, "The catalog API returned no data. Check API base and auth.")

    wanted_ids = None
    if mode == "excel":
        info = get_preview(excel_token or "")
        if not info or not Path(info["path"]).exists():
            raise HTTPException(400, "Uploaded Excel token expired; please re-upload the file.")
        tmp = Path(info["path"])
        try:
            df_ids = _read_ids_frame(tmp)
        except Exception as e:
            raise HTTPException(400, f"Failed to read the uploaded Excel/CSV: {e}")
        id_col = _resolve_id_column(df_ids, (excel_id_col or "").strip() or None)
        wanted_ids = set(df_ids[id_col].astype(str).str.strip().tolist())

    df = pd.DataFrame(programs)

    # primary key (supports dot-paths)
    if "." in primary_key or primary_key not in df.columns:
        df["_pk"] = df.apply(lambda r: get_nested_value(r.to_dict(), primary_key), axis=1)
        pk_col = "_pk"
        if df[pk_col].astype(str).str.strip().eq("").all():
            raise HTTPException(400, f"Primary key '{primary_key}' not found in API data.")
    else:
        pk_col = primary_key

    if wanted_ids is not None:
        df = df[df[pk_col].astype(str).str.strip().isin(wanted_ids)].copy()
        if df.empty:
            raise HTTPException(400, "None of the provided IDs matched the API data.")

    # transcripts (text + filename)
    t_map, t_name = await load_transcripts(transcripts)
    if t_map:
        df["transcript_text"] = df[pk_col].astype(str).map(t_map).fillna("")
        df["_transcript_name"] = df[pk_col].astype(str).map(t_name).fillna("")

    # Build preview rows
    rows = []
    for _, r in df.head(50).iterrows():
        d = r.to_dict()
        row = {"Primary Key": get_nested_value(d, primary_key)}
        for f in embed_fields:
            row[f] = get_nested_value(d, f)
        row["transcript"] = d.get("_transcript_name", "")
        rows.append(row)

    return {"rows": rows}


@router.post("/generate")
async def run_generate(
    mode: str = Form("api"),
    primary_key: str = Form(...),
    embed_fields: List[str] = Form(...),
    excel_token: Optional[str] = Form(None),
    excel_id_col: Optional[str] = Form(None),
    transcripts: List[UploadFile] = File(default=[]),
):
    if not primary_key or not str(primary_key).strip():
        raise HTTPException(400, "Please choose a Primary key.")
    _assert_embed_fields_ok(embed_fields)
    _assert_api_base_configured()

    try:
        programs = fetch_all_programs()
    except Exception as e:
        raise HTTPException(400, f"API error while fetching programs: {e}")
    if not programs:
        raise HTTPException(400, "The catalog API returned no data. Check API base and auth.")

    wanted_ids = None
    if mode == "excel":
        info = pop_preview(excel_token or "")
        if not info or not Path(info["path"]).exists():
            raise HTTPException(400, "Uploaded Excel token expired; please re-upload the file.")
        tmp = Path(info["path"])
        try:
            df_ids = _read_ids_frame(tmp)
        except Exception as e:
            raise HTTPException(400, f"Failed to read the uploaded Excel/CSV: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        id_col = _resolve_id_column(df_ids, (excel_id_col or "").strip() or None)
        wanted_ids = set(df_ids[id_col].astype(str).str.strip().tolist())

    df = pd.DataFrame(programs)

    # primary key (supports dot-paths)
    if "." in primary_key or primary_key not in df.columns:
        df["_pk"] = df.apply(lambda r: get_nested_value(r.to_dict(), primary_key), axis=1)
        pk_col = "_pk"
        if df[pk_col].astype(str).str.strip().eq("").all():
            raise HTTPException(400, f"Primary key '{primary_key}' not found in API data.")
    else:
        pk_col = primary_key

    if wanted_ids is not None:
        df = df[df[pk_col].astype(str).str.strip().isin(wanted_ids)].copy()
        if df.empty:
            raise HTTPException(400, "None of the provided IDs matched the API data.")

    # transcripts (text only for embeddings)
    t_map, _ = await load_transcripts(transcripts)
    if t_map:
        df["transcript_text"] = df[pk_col].astype(str).map(t_map).fillna("")
        if "transcript_text" not in embed_fields:
            embed_fields.append("transcript_text")

    texts = [build_text_from_fields(row.to_dict(), embed_fields) for _, row in df.iterrows()]
    df["_text_for_embedding"] = texts
    df = df[df["_text_for_embedding"].astype(str).str.strip() != ""].copy()
    if df.empty:
        raise HTTPException(400, "Nothing to embed after field selection.")

    embs = await batch_embed(df["_text_for_embedding"].tolist())
    X = np.vstack(embs)

    uid = str(uuid.uuid4())[:8]
    raw_name, emb_name = save_dataset_files(df, X, uid)
    return {
        "uid": uid,
        "parquet": f"/api/download/{raw_name}",
        "embeddings": f"/api/download/{emb_name}",
    }
