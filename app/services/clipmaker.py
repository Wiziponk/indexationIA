from __future__ import annotations

import io
import json
import math
import re
import zipfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.signal import find_peaks

import nltk
from nltk.tokenize import sent_tokenize

from .embeddings import batch_embed, build_text_from_fields
from .utils import get_nested_value
from ..config import SEG_EMBED_MODEL, SEG_GEN_MODEL, get_openai_client, DATA_DIR

# Ensure punkt (silent)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ---------- Helpers adapted from your segmenter (credited) ----------
# The segmentation logic here is adapted from the script you shared.
# The approach: windowed sentences → embeddings → boundary detection +
# specificity & coherence scoring → keep top segments.
# Titles are optionally generated via chat completions.
# (Source credited in the response.) 

def _hhmmss_to_seconds(ts: str | None) -> int | None:
    if not ts:
        return None
    h, m, s = [int(x) for x in ts.split(":")]
    return h * 3600 + m * 60 + s

def _seconds_to_hhmmss(sec: int | None) -> str | None:
    if sec is None:
        return None
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _sentence_windows(text: str, lang: str = "fr", win: int = 4) -> Tuple[List[str], List[Dict[str, Any]]]:
    sents = sent_tokenize(text, language="french" if lang.startswith("fr") else "english")
    chunks, metas = [], []
    for i in range(0, len(sents), win):
        chunk = " ".join(sents[i:i+win]).strip()
        if chunk:
            chunks.append(chunk)
            metas.append({"start": None, "end": None})
    return chunks, metas

def _embed_norm(texts: List[str]) -> np.ndarray:
    # normalize to unit length for cosine ops
    embs = batch_embed(texts, model=SEG_EMBED_MODEL)
    arr = np.vstack([np.array(v, dtype=np.float32) for v in embs])
    return normalize(arr)

def _detect_boundaries(emb: np.ndarray, prominence=0.15, distance=2, smooth=3) -> List[int]:
    sim = np.sum(emb[:-1] * emb[1:], axis=1)
    change = 1 - sim
    if smooth > 1:
        k = np.ones(smooth) / smooth
        change = np.convolve(change, k, mode="same")
    peaks, _ = find_peaks(change, prominence=prominence, distance=distance)
    bounds = [0] + [int(p + 1) for p in peaks] + [emb.shape[0]]
    return sorted(set(bounds))

def _tfidf_specificity(texts: List[str], lang="french") -> np.ndarray:
    vec = TfidfVectorizer(max_features=6000, stop_words=lang)
    X = vec.fit_transform(texts)
    idf = dict(zip(vec.get_feature_names_out(), vec.idf_))
    scores = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        idx = row.argsort()[::-1][:12]
        feats = vec.get_feature_names_out()[idx]
        spec = float(np.mean([idf[f] for f in feats])) if len(feats) else 0.0
        scores.append(spec)
    return np.array(scores)

def _intra_coherence(texts: List[str]) -> np.ndarray:
    vals = []
    for t in texts:
        sents = sent_tokenize(t, language="french")
        if len(sents) < 2:
            vals.append(0.0)
            continue
        E = _embed_norm(sents)
        sim = E @ E.T
        m = (np.sum(sim) - len(sents)) / (len(sents) * (len(sents) - 1))
        vals.append(float(m))
    return np.array(vals)

def _robust_unit(x) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    zx = (x - np.median(x)) / (np.std(x) + 1e-9)
    return 1 / (1 + np.exp(-zx))

def _summarize_fr(text: str) -> Tuple[str, str]:
    client = get_openai_client()
    prompt = f"""Tu es éditeur vidéo. Donne un titre bref et un résumé clair du passage ci-dessous.

Texte (français) :
{text}

Réponds en JSON compact avec exactement ces clés: title, summary.
- title : 3–8 mots, informatif, en français, sans guillemets.
- summary : 2–3 phrases claires, en français, sans redire l'intro du documentaire.
"""
    resp = client.chat.completions.create(
        model=SEG_GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return data.get("title", "Segment"), data.get("summary", "")
    except Exception:
        return "Segment", ""


# ---------- Public functions ----------

def segment_text(
    text: str,
    lang: str = "fr",
    keep_ratio: float = 0.6,
    with_titles: bool = False,
    brief: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Return kept segments: [{start,end,text,score,title,summary}]
    """
    chunks, metas = _sentence_windows(text, lang=lang, win=4)
    if len(chunks) < 2:
        base = {"start": None, "end": None, "text": text.strip(), "score": 1.0, "title": "Segment", "summary": ""}
        if with_titles:
            t, s = _summarize_fr(base["text"])
            base["title"], base["summary"] = t, s
        return [base]

    E = _embed_norm(chunks)
    bounds = _detect_boundaries(E, prominence=0.15, distance=2, smooth=3)

    sections = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        txt = " ".join(chunks[a:b]).strip()
        start = metas[a].get("start") if a < len(metas) else None
        end = metas[b - 1].get("end") if (b - 1) < len(metas) else None
        sections.append({"a": a, "b": b, "text": txt, "start": start, "end": end})

    texts = [s["text"] for s in sections]
    spec = _tfidf_specificity(texts, lang="french")
    coh = _intra_coherence(texts)

    # Topic entropy (lower is more focused)
    vec = TfidfVectorizer(max_features=6000, stop_words="french")
    X = vec.fit_transform(texts).toarray()
    eps = 1e-12
    p = X / (X.sum(axis=1, keepdims=True) + eps)
    ent = -np.sum(p * np.log(p + eps), axis=1)

    spec_u, coh_u, ent_u = _robust_unit(spec), _robust_unit(coh), _robust_unit(-ent)
    score = 0.45 * spec_u + 0.25 * coh_u + 0.30 * ent_u

    # Optional brief relevance boost
    if brief:
        brief_vec = _embed_norm([brief])[0]
        seg_vecs = _embed_norm(texts)
        sim = (seg_vecs @ brief_vec).ravel()
        sim_u = (sim + 1) / 2
        score = 0.75 * score + 0.25 * sim_u

    k = max(1, int(math.ceil(keep_ratio * len(sections))))
    keep_idx = np.argsort(score)[::-1][:k]

    results = []
    for i in sorted(keep_idx):
        s = sections[i].copy()
        s["score"] = float(score[i])
        if with_titles:
            t, su = _summarize_fr(s["text"])
            s["title"], s["summary"] = t, su
        else:
            s["title"], s["summary"] = "Segment", ""
        results.append(s)
    return results


def embed_clips(segments: List[Dict[str, Any]]) -> np.ndarray:
    texts = [s["text"] for s in segments]
    embs = batch_embed(texts, model=SEG_EMBED_MODEL)
    return np.vstack([np.array(v, dtype=np.float32) for v in embs])


def program_embedding(program_row: Dict[str, Any], primary_key: str, embed_fields: List[str], segments: List[Dict[str, Any]]) -> np.ndarray:
    # base text from API fields
    base = build_text_from_fields(program_row, embed_fields)
    # add short summaries/titles from clips (if present)
    extra = []
    for s in segments[:6]:
        t = s.get("title") or ""
        su = s.get("summary") or ""
        if t or su:
            extra.append(f"{t}. {su}".strip())
    text = base + ("\n\n" + "\n".join(extra) if extra else "")
    vec = batch_embed([text], model=SEG_EMBED_MODEL)[0]
    return np.array(vec, dtype=np.float32)


def make_zip_for_program(
    uid: str,
    pk_value: str,
    program_row: Dict[str, Any],
    primary_key: str,
    embed_fields: List[str],
    segments: List[Dict[str, Any]],
    clip_embs: np.ndarray,
    prog_emb: np.ndarray,
) -> str:
    out_dir = DATA_DIR / "zips" / uid
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{pk_value}.zip"

    # Prepare small files in-memory
    program_doc = {
        "primary_key": primary_key,
        "pk_value": pk_value,
        "fields": {f: get_nested_value(program_row, f) for f in embed_fields},
    }

    readme = f"""IndexationIA — package for emission {pk_value}

Files:
- program.json             : selected API fields for this emission
- clips/clip_XXXX.txt      : text of each kept clip (title/summary header)
- clips_meta.jsonl         : one JSON per line with start, end, score, title, summary, text
- emb_clips.npy            : (num_clips, dim) clip embeddings ({SEG_EMBED_MODEL})
- emb_program.npy          : (dim,) program embedding ({SEG_EMBED_MODEL})
- emb_model.txt            : model identifiers used

"""

    # Write ZIP
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("program.json", json.dumps(program_doc, ensure_ascii=False, indent=2))
        zf.writestr("README.txt", readme)
        zf.writestr("emb_model.txt", json.dumps({"clip_embed_model": SEG_EMBED_MODEL, "program_embed_model": SEG_EMBED_MODEL}, ensure_ascii=False))

        # clips_meta.jsonl and text files
        meta_buf = io.StringIO()
        for i, s in enumerate(segments, start=1):
            # meta line
            meta_buf.write(json.dumps({
                "index": i,
                "start": s.get("start"),
                "end": s.get("end"),
                "score": s.get("score"),
                "title": s.get("title"),
                "summary": s.get("summary"),
                "text": s.get("text")
            }, ensure_ascii=False) + "\n")
            # text file
            header = f"{s.get('title','Segment')}\n{(s.get('summary') or '')}\n\n"
            body = s.get("text", "")
            zf.writestr(f"clips/clip_{i:04d}.txt", header + body)
        zf.writestr("clips_meta.jsonl", meta_buf.getvalue())

        # embeddings
        zf.writestr("emb_clips.npy", io.BytesIO(np.save(io.BytesIO(), clip_embs) or b"").getvalue())  # trick: np.save writes to buffer, but returns None
        # The above trick won't work as-is; do it properly:
    # Properly write npy via buffer
    with zipfile.ZipFile(zip_path, mode="a", compression=zipfile.ZIP_DEFLATED) as zf2:
        buf = io.BytesIO()
        np.save(buf, clip_embs)
        zf2.writestr("emb_clips.npy", buf.getvalue())
        buf2 = io.BytesIO()
        np.save(buf2, prog_emb)
        zf2.writestr("emb_program.npy", buf2.getvalue())

    return str(zip_path)
