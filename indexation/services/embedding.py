from __future__ import annotations

"""Embedding related helpers."""

import json
from typing import Iterable, List

from indexation.config import EMBED_MODEL, client

from ..helpers import get_nested_value


def build_text_from_fields(row: dict, fields: Iterable[str]) -> str:
    """Build a text snippet from the selected fields of a row."""
    parts = []
    for f in fields:
        v = get_nested_value(row, f)
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        parts.append(str(v))
    return "\n".join([p for p in parts if p and p.strip()])


def batch_embed(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """Embed a list of texts using the OpenAI API."""
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]
