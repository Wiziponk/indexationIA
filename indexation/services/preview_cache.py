from __future__ import annotations

"""Utility for temporary preview file handling."""

import time
from pathlib import Path
from typing import Dict

# token -> {"path": Path, "ts": float}
PREVIEW_INDEX: Dict[str, Dict[str, object]] = {}


def register_preview(token: str, path: Path) -> None:
    PREVIEW_INDEX[token] = {"path": path, "ts": time.time()}


def pop_preview(token: str):
    return PREVIEW_INDEX.pop(token, None)


def cleanup_previews(max_age: int = 3600) -> None:
    """Remove preview files older than ``max_age`` seconds."""
    now = time.time()
    for token, info in list(PREVIEW_INDEX.items()):
        if now - info["ts"] > max_age:
            try:
                Path(info["path"]).unlink(missing_ok=True)
            except Exception:
                pass
            PREVIEW_INDEX.pop(token, None)
