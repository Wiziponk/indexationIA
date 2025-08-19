from typing import List, Tuple
import numpy as np
from openai import OpenAI
from ..config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def build_texts(rows, fields: List[str]) -> List[str]:
    texts = []
    for r in rows:
        parts = []
        for f in fields:
            v = r.get(f)
            if v is None:
                continue
            if not isinstance(v, str):
                v = str(v)
            v = v.strip()
            if v:
                parts.append(f"{f}: {v}")
        texts.append(" | ".join(parts) if parts else "")
    return texts

def embed_texts(texts: List[str], batch_size: int = 96) -> np.ndarray:
    vectors = []
    model = settings.OPENAI_EMBED_MODEL
    for b in chunk(texts, batch_size):
        resp = client.embeddings.create(model=model, input=b)
        emb = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        vectors.extend(emb)
    return np.stack(vectors, axis=0)
