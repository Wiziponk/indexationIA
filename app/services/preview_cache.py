from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

PREVIEW_INDEX: Dict[str, Dict[str, Any]] = {}


def register_preview(token: str, path: Path) -> None:
    PREVIEW_INDEX[token] = {"path": path, "ts": time.time()}


def pop_preview(token: str):
    """Consume and remove a preview entry."""
    return PREVIEW_INDEX.pop(token, None)


def get_preview(token: str):
    """Peek at a preview entry without removing it."""
    return PREVIEW_INDEX.get(token)


def cleanup_previews(max_age: int = 3600) -> None:
    now = time.time()
    for token, info in list(PREVIEW_INDEX.items()):
        if now - info["ts"] > max_age:
            try:
                Path(info["path"]).unlink(missing_ok=True)
            except Exception:
                pass
            PREVIEW_INDEX.pop(token, None)
