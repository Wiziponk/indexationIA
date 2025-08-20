from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# External API
API_BASE = os.getenv("EDUC_API_BASE", "https://educ.arte.tv/api/list/programs")
API_TOKEN = os.getenv("EDUC_API_TOKEN", "")
TIMEOUT = int(os.getenv("TIMEOUT", "60"))

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "96"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "50000"))

# Projection
PROJECTION_DEFAULT = os.getenv("PROJECTION_DEFAULT", "pca")  # 'pca' or 'tsne' (UMAP optional)

# UI
TITLE_COL_CANDIDATES = ["title", "name", "programme_title", "titre"]
TEXT_COL_CANDIDATES = ["transcript_text", "transcript", "text", "summary", "synopsis", "description"]
ID_GUESS = ["code_emission", "codeEmission", "id", "video_id", "uid"]
