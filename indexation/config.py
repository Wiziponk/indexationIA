from __future__ import annotations

"""Application configuration and constants."""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# External services
API_BASE = os.getenv("EDUC_API_BASE", "https://educ.arte.tv/api/list/programs")
API_TOKEN = os.getenv("EDUC_API_TOKEN")
TIMEOUT = 60

# Embeddings
EMBED_MODEL = "text-embedding-3-small"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "test"))

# UI / heuristics
TITLE_COL_CANDIDATES = ["title", "name", "programme_title", "titre"]
TEXT_COL_CANDIDATES = ["transcript_text", "transcript", "text", "summary", "synopsis", "description"]
ID_GUESS = ["code_emission", "codeEmission", "id", "video_id", "uid"]

# Uploads
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "docx", "parquet", "npy"}


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret")
    MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH
    ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
    DATA_DIR = DATA_DIR
    RESULTS_DIR = RESULTS_DIR


class DevConfig(Config):
    DEBUG = True


class ProdConfig(Config):
    DEBUG = False


def get_config():
    env = os.getenv("APP_ENV", "dev").lower()
    return ProdConfig if env == "prod" else DevConfig
