import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# External services
API_BASE = os.getenv("EDUC_API_BASE", "https://educ.arte.tv/api/list/programs")
API_TOKEN = os.getenv("EDUC_API_TOKEN")
TIMEOUT = 60

# Embeddings
EMBED_MODEL = "text-embedding-3-small"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# UI / heuristics
TITLE_COL_CANDIDATES = ["title", "name", "programme_title", "titre"]
TEXT_COL_CANDIDATES = ["transcript_text", "transcript", "text", "summary", "synopsis", "description"]
ID_GUESS = ["code_emission", "codeEmission", "id", "video_id", "uid"]

# Flask
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret")
