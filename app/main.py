from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import logging

from .routers.common import router as common_router
from .routers.generate import router as generate_router      # kept (dataset legacy)
from .routers.cluster import router as cluster_router        # kept (legacy clustering)
from .routers.clip import router as clip_router              # NEW: clips flow + batch zips
from .routers.cluster_zip import router as cluster_zip_router# NEW: cluster from zips
from .config import BASE_DIR

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Indexation v3", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        logger.exception("Unhandled error for %s %s", request.method, request.url)
        raise

app.include_router(common_router, prefix="/api", tags=["common"])
app.include_router(generate_router, prefix="/api", tags=["generate"])
app.include_router(cluster_router, prefix="/api", tags=["cluster"])
app.include_router(clip_router, prefix="/api", tags=["clips"])
app.include_router(cluster_zip_router, prefix="/api", tags=["cluster-zips"])

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))
