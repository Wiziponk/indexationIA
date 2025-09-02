from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import BASE_DIR
from .db import init_db
from .routers.clip import router as clip_router  # NEW: clips flow + batch zips
from .routers.cluster import router as cluster_router  # kept (legacy clustering)
from .routers.cluster_zip import router as cluster_zip_router  # NEW: cluster from zips
from .routers.common import router as common_router
from .routers.datasets import router as datasets_router  # NEW: datasets CRUD
from .routers.db import router as db_router  # NEW: DB CRUD
from .routers.generate import router as generate_router  # kept (dataset legacy)

logger = logging.getLogger("uvicorn.error")

tags_metadata = [
    {"name": "common", "description": "Health checks and helpers"},
    {"name": "generate", "description": "Legacy dataset generation"},
    {"name": "cluster", "description": "Legacy clustering endpoints"},
    {"name": "clips", "description": "Clip segmentation workflow"},
    {"name": "cluster-zips", "description": "Cluster embeddings from uploaded ZIPs"},
    {"name": "datasets", "description": "Dataset CRUD operations"},
    {"name": "db", "description": "Project and library database"},
]

app = FastAPI(title="Indexation v3", version="3.0.0", openapi_tags=tags_metadata)

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
app.include_router(datasets_router, prefix="/api", tags=["datasets"])
app.include_router(db_router, prefix="/api/db", tags=["db"])

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount(
    "/",
    StaticFiles(directory=str(BASE_DIR / "web" / "dist"), html=True),
    name="web",
)


@app.on_event("startup")
def _startup():
    # Create SQLite schema on boot
    init_db()
