from __future__ import annotations

import subprocess
from fastapi import APIRouter, UploadFile, HTTPException, File
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd

from ..config import API_BASE, DATA_DIR
from ..services.preview_cache import register_preview, cleanup_previews
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
    fields, source, sample_size, err = discover_fields()
    payload = {
        "fields": fields,
        "api_base": API_BASE,
        "source": source,            # 'direct' | 'fallback' | 'empty'
        "sample_size": sample_size,  # number of items inspected
    }
    if err:
        payload["note"] = err
    return payload


@router.post("/upload-ids")
async def upload_ids(excel: UploadFile = File(...)):
    cleanup_previews()

    if not excel or not excel.filename:
        raise HTTPException(400, "No file uploaded")

    suffix = Path(excel.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "Please upload a .csv, .xlsx, or .xls file")

    token_path = DATA_DIR / f"preview_{excel.filename}"
    token_path.write_bytes(await excel.read())

    # Read to get columns & preview
    try:
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(token_path)
        else:
            df = pd.read_csv(token_path)
    except Exception as e:
        raise HTTPException(400, f"Cannot read file: {e}")

    df.columns = [str(c).strip() for c in df.columns]

    # Use filename as token (sufficient for Codespaces), or use random if you prefer.
    token = token_path.stem

    register_preview(token, token_path)
    preview = df.head(50)
    return {
        "token": token,
        "columns": list(df.columns),
        "preview": preview.to_dict(orient="records"),
    }


@router.get("/download/{name}")
def download(name: str):
    candidate = DATA_DIR / name
    if not candidate.exists() or candidate.suffix not in {".parquet", ".npy", ".zip"}:
        raise HTTPException(404, "Not found")
    return FileResponse(candidate)
