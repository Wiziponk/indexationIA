from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, UploadFile, HTTPException, File
from fastapi.responses import FileResponse

from ..config import API_BASE, DATA_DIR
from ..services.preview_cache import register_preview, pop_preview, cleanup_previews
from ..services.api_client import discover_fields

router = APIRouter()


@router.get("/health")
def health():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    return {"ok": True, "version": sha}


@router.get("/fields")
def api_fields():
    """
    Expose field names for the UI to populate the Primary Key and Embed Fields pickers.
    """
    return {"fields": discover_fields(), "api_base": API_BASE}


@router.post("/upload-ids")
async def upload_ids(excel: UploadFile = File(...)):
    """
    Accept an Excel/CSV upload of IDs, store temporarily, and return:
      - token: a short-lived token to reference the temp file later
      - columns: list of column names
      - preview: first 50 rows for display
    """
    cleanup_previews()

    if not excel or not excel.filename:
        raise HTTPException(400, "No file uploaded")

    suffix = Path(excel.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "Please upload a .csv, .xlsx, or .xls file")

    token = uuid.uuid4().hex
    tmp = DATA_DIR / f"preview_{token}{suffix}"

    try:
        tmp.write_bytes(await excel.read())
        if suffix in (".xlsx", ".xls"):
            # requires openpyxl for .xlsx and xlrd<2.0 for .xls
            df = pd.read_excel(tmp)
        else:
            df = pd.read_csv(tmp)
    except Exception as e:
        raise HTTPException(400, f"Cannot read file: {e}")

    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]

    register_preview(token, tmp)
    preview = df.head(50)
    return {
        "token": token,
        "columns": list(df.columns),
        "preview": preview.to_dict(orient="records"),
    }


@router.get("/download/{name}")
def download(name: str):
    """
    Serve generated artifacts (parquet / npy) from /data.
    """
    candidate = DATA_DIR / name
    if not candidate.exists() or candidate.suffix not in {".parquet", ".npy"}:
        raise HTTPException(404, "Not found")
    return FileResponse(candidate)
