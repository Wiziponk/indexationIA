from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from .routers.common import router as common_router
from .routers.generate import router as generate_router
from .routers.cluster import router as cluster_router
from .config import BASE_DIR

app = FastAPI(title="Indexation v2", version="1.0.0")

# CORS (relaxed for local usage)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(common_router, prefix="/api", tags=["common"])
app.include_router(generate_router, prefix="/api", tags=["generate"])
app.include_router(cluster_router, prefix="/api", tags=["cluster"])

# Static UI
app.mount("/static", StaticFiles(directory=str(BASE_DIR.parent / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(str(BASE_DIR.parent / "static" / "index.html"))
