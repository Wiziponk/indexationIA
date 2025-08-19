import os, json, tempfile, shutil, time
from typing import Dict, Any, List
from .utils import ensure_dir

def atomic_write_bytes(path: str, data: bytes):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

def atomic_write_text(path: str, text: str, encoding="utf-8"):
    atomic_write_bytes(path, text.encode(encoding))

def list_artifacts(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    files = []
    for f in os.listdir(data_dir):
        if f.endswith(".parquet") or f.endswith(".npy"):
            files.append(f)
    return sorted(files)

def save_preview_token(temp_dir: str, token: str, payload: Dict[str, Any]):
    ensure_dir(temp_dir)
    path = os.path.join(temp_dir, f"preview_{token}.json")
    atomic_write_text(path, json.dumps(payload))

def load_preview_token(temp_dir: str, token: str) -> Dict[str, Any]:
    path = os.path.join(temp_dir, f"preview_{token}.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Preview token not found")
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
