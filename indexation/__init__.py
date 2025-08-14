"""Core package for indexation helpers and configuration."""

from .config import (
    API_BASE,
    DATA_DIR,
    TITLE_COL_CANDIDATES,
    ID_GUESS,
    SECRET_KEY,
)
from .helpers import (
    discover_fields,
    ensure_list,
    fetch_all_programs,
    build_text_from_fields,
    batch_embed,
    auto_kmeans,
    pca_2d,
)

__all__ = [
    "API_BASE",
    "DATA_DIR",
    "TITLE_COL_CANDIDATES",
    "ID_GUESS",
    "SECRET_KEY",
    "discover_fields",
    "ensure_list",
    "fetch_all_programs",
    "build_text_from_fields",
    "batch_embed",
    "auto_kmeans",
    "pca_2d",
]
