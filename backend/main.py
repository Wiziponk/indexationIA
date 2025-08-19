import os, uuid, io, json
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
from rq import Queue
from redis import Redis

from .config import settings
from .models import GeneratePreviewRequest, GenerateRequest, GenerateResult, ClusterRequest, ClusterResult
from .services.api_client import fetch_programs, fields_from_records, sample_rows
from .services.storage import list_artifacts, save_preview_token, load_preview_token
from .services.utils import ensure_dir
from . import jobs as job_funcs

app = FastAPI(title="Embeddings & Clustering API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis = Redis.from_url(settings.REDIS_URL)
q = Queue("default", connection=redis)

@app.get("/healthz")
def healthz():
    # Soft check - presence of API key only
    ok = bool(settings.OPENAI_API_KEY)
    return {"ok": ok}

@app.get("/")
def root():
    return {"name": "Embeddings & Clustering", "data_dir": settings.DATA_DIR}

@app.get("/api/fields")
async def api_fields(base: str | None = None):
    recs = await fetch_programs(base or settings.API_BASE, max_items=50)
    fields = fields_from_records(recs)
    return {"base": base or settings.API_BASE, "fields": fields}

@app.get("/api/sample")
async def api_sample(base: str | None = None, fields: str = "id,title,description"):
    field_list = [f.strip() for f in fields.split(",") if f.strip()]
    recs = await fetch_programs(base or settings.API_BASE, max_items=10)
    return {"rows": sample_rows(recs, field_list, n=5)}

@app.post("/preview-excel")
async def preview_excel(file: UploadFile = File(...)):
    # Read small sample for preview; store the head+columns under a token
    content = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(content)) if file.filename.endswith(".xlsx") else pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Cannot read file: {e}")

    head = df.head(10).to_dict(orient="records")
    cols = list(df.columns)
    token = uuid.uuid4().hex[:8]
    temp_dir = os.path.join(settings.DATA_DIR, "_tmp")
    save_preview_token(temp_dir, token, {"columns": cols, "head": head})
    return {"token": token, "columns": cols, "head": head}

@app.get("/download/{name}")
def download(name: str):
    path = os.path.join(settings.DATA_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=name)

# ------- Job helpers (polling-based) -------
@app.post("/_jobs/generate")
def job_generate(req: GenerateRequest):
    job = q.enqueue(job_funcs.generate_task, req.model_dump(), job_timeout="2h")
    return {"job_id": job.id}

@app.post("/_jobs/cluster")
def job_cluster(req: ClusterRequest):
    job = q.enqueue(job_funcs.cluster_task, req.model_dump(), job_timeout="2h")
    return {"job_id": job.id}

@app.get("/_jobs/{job_id}")
def job_status(job_id: str):
    from rq.job import Job
    try:
        job = Job.fetch(job_id, connection=redis)
    except Exception:
        raise HTTPException(404, "Job not found")
    status = job.get_status()
    meta = getattr(job, "meta", {}) or {}
    if status == "finished":
        return {"status": status, "progress": 1.0, "result": job.result, "message": "Done"}
    elif status == "failed":
        return {"status": status, "progress": meta.get("progress", 0.0), "error": str(job.exc_info)}
    else:
        return {"status": status, "progress": meta.get("progress", 0.0), "message": meta.get("message", "Working")}

# ------- Legacy public endpoints kept for parity -------
@app.post("/generate/preview", response_model=dict)
async def generate_preview(req: GeneratePreviewRequest):
    recs = await fetch_programs(req.base or settings.API_BASE, max_items=25)
    rows = sample_rows(recs, [req.primary_key] + req.fields, n=5)
    return {"rows": rows}

@app.post("/generate", response_model=GenerateResult)
def generate(req: GenerateRequest):
    # Synchronous wrapper that waits for the job (for full parity). Not recommended for large runs.
    job = q.enqueue(job_funcs.generate_task, req.model_dump(), job_timeout="2h")
    from rq.job import Job
    import time
    while True:
        j = Job.fetch(job.id, connection=redis)
        if j.get_status() == "finished":
            return j.result
        if j.get_status() == "failed":
            raise HTTPException(500, "Generation failed")
        time.sleep(1.0)

@app.post("/cluster", response_model=ClusterResult)
def cluster(req: dict):
    # Allow both legacy and typed request
    from rq.job import Job
    job = q.enqueue(job_funcs.cluster_task, req, job_timeout="2h")
    import time
    while True:
        j = Job.fetch(job.id, connection=redis)
        if j.get_status() == "finished":
            return j.result
        if j.get_status() == "failed":
            raise HTTPException(500, "Clustering failed")
        time.sleep(1.0)

@app.get("/artifacts")
def artifacts():
    return {"files": list_artifacts(settings.DATA_DIR)}
