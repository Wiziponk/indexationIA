from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, HTTPException, File
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd

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
    return {"fields": discover_fields(), "api_base": API_BASE}

@router.post("/upload-ids")
async def upload_ids(excel: UploadFile=File(...)):
    cleanup_previews()
    if not excel or not excel.filename:
        raise HTTPException(400, "No file uploaded")
    token = uuid.uuid4().hex
    tmp = DATA_DIR / f"preview_{token}{Path(excel.filename).suffix}"
    try:
        tmp.write_bytes(await excel.read())
        if tmp.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(tmp)
        else:
            df = pd.read_csv(tmp)
    except Exception as e:
        raise HTTPException(400, f"Cannot read file: {e}")
    register_preview(token, tmp)
    preview = df.head(50)
    return {"token": token, "columns": list(df.columns), "preview": preview.to_dict(orient="records")}

@router.get("/download/{name}")
def download(name: str):
    candidate = DATA_DIR / name
    if not candidate.exists() or not candidate.suffix in [".parquet", ".npy"]:
        raise HTTPException(404, "Not found")
    return FileResponse(candidate)
