from __future__ import annotations
import os
from pathlib import Path
from openai import OpenAI

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# External API
API_BASE = os.getenv("EDUC_API_BASE", "").strip()
API_TOKEN = os.getenv("EDUC_API_TOKEN", "").strip()
TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))

# Allow disabling TLS verification if your API uses self-signed certs
VERIFY_SSL = (os.getenv("EDUC_VERIFY_SSL", "1").strip() not in {"0", "false", "False"})

# OpenAI
def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running.")
    return OpenAI(api_key=key)

# Embeddings for classic dataset
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "96"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "50000"))

# Segmentation / clip naming
SEG_EMBED_MODEL = os.getenv("SEG_EMBED_MODEL", "text-embedding-3-large")
SEG_GEN_MODEL = os.getenv("SEG_GEN_MODEL", "gpt-4o-mini")
CLIP_KEEP_RATIO_DEFAULT = float(os.getenv("CLIP_KEEP_RATIO", "0.6"))

# Clustering defaults
DEFAULT_PROJ = PROJECTION_DEFAULT = os.getenv("DEFAULT_PROJ", "pca")  # pca|tsne

# UI
TITLE_COL_CANDIDATES = ["title", "name", "programme_title", "titre"]
TEXT_COL_CANDIDATES = ["transcript_text", "transcript", "text", "summary", "synopsis", "description"]
ID_GUESS = ["code_emission", "codeEmission", "id", "video_id", "uid"]
