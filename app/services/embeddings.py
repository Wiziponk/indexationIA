from __future__ import annotations

import json
from typing import Iterable, List, Dict

import numpy as np
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from ..config import EMBED_MODEL, EMBED_BATCH_SIZE, MAX_TEXT_CHARS
from .utils import get_nested_value

_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        import os
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        _client = OpenAI(api_key=key)
    return _client

def build_text_from_fields(row: dict, fields: Iterable[str]) -> str:
    parts = []
    for f in fields:
        v = get_nested_value(row, f)
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        parts.append(str(v))
    txt = "\n".join([p for p in parts if p and p.strip()])
    if len(txt) > MAX_TEXT_CHARS:
        txt = txt[:MAX_TEXT_CHARS]
    return txt

@retry(
    reraise=True,
    retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(6),
)
def _embed_batch(batch: List[str], model: str):
    return get_client().embeddings.create(model=model, input=batch)

async def batch_embed(texts: List[str], model: str = EMBED_MODEL, batch_size: int = EMBED_BATCH_SIZE) -> List[List[float]]:
    cleaned = [str(t or "").strip() for t in texts]
    cleaned = [t for t in cleaned if t]
    if not cleaned:
        return []
    out: List[List[float]] = []
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i+batch_size]
        resp = _embed_batch(batch, model)
        out.extend([d.embedding for d in resp.data])
    return out

def suggest_cluster_names(df, title_col, cluster_col="_cluster", max_titles=5) -> Dict[int, str]:
    names = {}
    try:
        client = get_client()
    except Exception:
        return names
    if title_col not in df.columns:
        return names
    for cl, sub in df.groupby(cluster_col):
        titles = sub[title_col].dropna().astype(str).head(max_titles).tolist()
        if not titles:
            continue
        prompt = (
            "Voici quelques titres d’éléments appartenant au même cluster : "
            + "; ".join(titles)
            + ". Donne un nom de thème très court en français."
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu aides à nommer des clusters."},
                    {"role": "user", "content": prompt},
                ],
            )
            names[int(cl)] = resp.choices[0].message.content.strip()
        except Exception:
            continue
    return names
