from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import threading, asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException, Form, UploadFile, File
from ..services.jobs import new_job, set_status, set_result, get_job

from ..config import DATA_DIR, CLIP_KEEP_RATIO_DEFAULT
from ..services.api_client import fetch_all_programs
from ..services.transcripts import load_transcripts
from ..services.utils import get_nested_value
from ..services.clipmaker import segment_text, embed_clips, program_embedding, make_zip_for_program
from ..services.preview_cache import get_preview, pop_preview

router = APIRouter()


def _read_ids_frame(tmp: Path) -> pd.DataFrame:
    if tmp.suffix.lower() in [".xlsx", ".xls"]:
        df_ids = pd.read_excel(tmp)
    else:
        df_ids = pd.read_csv(tmp)
    df_ids.columns = [str(c).strip() for c in df_ids.columns]
    return df_ids


@router.post("/segment/prepare")
async def prepare_dataset(
    mode: str = Form("api"),
    primary_key: str = Form(...),
    embed_fields: List[str] = Form(...),
    excel_token: Optional[str] = Form(None),
    excel_id_col: Optional[str] = Form(None),
    transcripts: List[UploadFile] = File(default=[]),
):
    """Build candidate dataset and report which items have transcripts."""
    programs = fetch_all_programs()
    if not programs:
        raise HTTPException(400, "The catalog API returned no data.")

    df = pd.DataFrame(programs)
    if "." in primary_key or primary_key not in df.columns:
        df["_pk"] = df.apply(lambda r: get_nested_value(r.to_dict(), primary_key), axis=1)
        pk_col = "_pk"
        if df[pk_col].astype(str).str.strip().eq("").all():
            raise HTTPException(400, f"Primary key '{primary_key}' not found in API data.")
    else:
        pk_col = primary_key

    wanted_ids = None
    if mode == "excel":
        info = get_preview(excel_token or "")
        if not info or not Path(info["path"]).exists():
            raise HTTPException(400, "Uploaded Excel token expired; please re-upload the file.")
        tmp = Path(info["path"])
        df_ids = _read_ids_frame(tmp)
        if excel_id_col and excel_id_col not in df_ids.columns:
            raise HTTPException(400, f"Column '{excel_id_col}' not found in uploaded file")
        id_col = excel_id_col or df_ids.columns[0]
        wanted_ids = set(df_ids[id_col].astype(str).str.strip().tolist())

    if wanted_ids is not None:
        df = df[df[pk_col].astype(str).str.strip().isin(wanted_ids)].copy()

    # Attach transcripts
    t_map, t_name = await load_transcripts(transcripts)
    df["__has_transcript"] = df[pk_col].astype(str).map(lambda x: x in t_map)

    included = df[df["__has_transcript"]].copy()
    excluded = df[~df["__has_transcript"]][pk_col].astype(str).tolist()

    return {
        "primary_key": primary_key,
        "count_included": int(len(included)),
        "count_excluded": int(len(excluded)),
        "excluded_ids": excluded[:200],  # cap in response
        "sample_ids": included[pk_col].astype(str).head(50).tolist(),  # for preview dropdown
    }


@router.post("/segment/preview")
async def preview_one(
    mode: str = Form("api"),
    primary_key: str = Form(...),
    embed_fields: List[str] = Form(...),
    excel_token: Optional[str] = Form(None),
    excel_id_col: Optional[str] = Form(None),
    sample_pk_value: str = Form(...),
    keep_ratio: float = Form(CLIP_KEEP_RATIO_DEFAULT),
    brief: Optional[str] = Form(None),
    with_titles: bool = Form(False),
    transcripts: List[UploadFile] = File(default=[]),
):
    """Run the clipper on one emission (preview)."""
    programs = fetch_all_programs()
    df = pd.DataFrame(programs)
    if "." in primary_key or primary_key not in df.columns:
        df["_pk"] = df.apply(lambda r: get_nested_value(r.to_dict(), primary_key), axis=1)
        pk_col = "_pk"
    else:
        pk_col = primary_key

    wanted_ids = None
    if mode == "excel":
        info = get_preview(excel_token or "")
        if not info or not Path(info["path"]).exists():
            raise HTTPException(400, "Uploaded Excel token expired; please re-upload the file.")
        tmp = Path(info["path"])
        df_ids = _read_ids_frame(tmp)
        id_col = excel_id_col or df_ids.columns[0]
        wanted_ids = set(df_ids[id_col].astype(str).str.strip().tolist())

    if wanted_ids is not None:
        df = df[df[pk_col].astype(str).str.strip().isin(wanted_ids)].copy()

    # Transcript map
    t_map, t_name = await load_transcripts(transcripts)
    df["__tx"] = df[pk_col].astype(str).map(t_map).fillna("")

    row = df[df[pk_col].astype(str) == str(sample_pk_value)].head(1)
    if row.empty:
        raise HTTPException(400, "Sample ID not found among included items or has no transcript.")

    program_row = row.iloc[0].to_dict()
    transcript_text = program_row.get("__tx", "").strip()
    if not transcript_text:
        raise HTTPException(400, "No transcript attached for this ID.")

    segments = await segment_text(transcript_text, keep_ratio=keep_ratio, with_titles=with_titles, brief=brief)
    clip_embs = await embed_clips(segments)
    prog_emb  = await program_embedding(program_row, primary_key, embed_fields, segments)

    # compact preview payload (truncate text)
    preview = []
    for s in segments[:20]:
        preview.append({
            "start": s.get("start"),
            "end": s.get("end"),
            "score": round(float(s.get("score", 0)), 3),
            "title": s.get("title"),
            "summary": s.get("summary"),
            "text": (s.get("text") or "")[:600] + ("…" if len(s.get("text","")) > 600 else "")
        })
    return {
        "pk_value": str(sample_pk_value),
        "segments": preview,
        "n_segments": len(segments),
        "clip_dim": int(clip_embs.shape[1]),
        "has_program_embedding": True,
    }


@router.post("/segment/name")
async def name_segments(
    segments_json: str = Form(...),
):
    """Name/summary for given segments (used after preview if titles were skipped)."""
    try:
        segs = json.loads(segments_json)
        assert isinstance(segs, list)
    except Exception:
        raise HTTPException(400, "segments_json must be a JSON array of {text,...}")

    named = []
    for s in segs:
        t, su = "Segment", ""
        try:
            from ..services.clipmaker import _summarize_fr  # reuse summarizer
            t, su = _summarize_fr(s.get("text", ""))
        except Exception:
            pass
        s2 = dict(s)
        s2["title"] = t
        s2["summary"] = su
        named.append(s2)
    return {"segments": named}


@router.post("/segment/batch")
async def batch_zip(
    mode: str = Form("api"),
    primary_key: str = Form(...),
    embed_fields: List[str] = Form(...),
    excel_token: Optional[str] = Form(None),
    excel_id_col: Optional[str] = Form(None),
    keep_ratio: float = Form(CLIP_KEEP_RATIO_DEFAULT),
    brief: Optional[str] = Form(None),
    with_titles: bool = Form(True),
    limit: int = Form(0),                  
    max_segments_per_item: int = Form(0), 
    transcripts: List[UploadFile] = File(default=[]),
):
    """Run clipper for ALL included items and write one ZIP per emission + a master ZIP."""
    uid = str(uuid.uuid4())[:8]
    out_dir = DATA_DIR / "zips" / uid
    out_dir.mkdir(parents=True, exist_ok=True)

    programs = fetch_all_programs()
    df = pd.DataFrame(programs)
    if "." in primary_key or primary_key not in df.columns:
        df["_pk"] = df.apply(lambda r: get_nested_value(r.to_dict(), primary_key), axis=1)
        pk_col = "_pk"
    else:
        pk_col = primary_key

    wanted_ids = None
    if mode == "excel":
        info = pop_preview(excel_token or "")
        if not info or not Path(info["path"]).exists():
            raise HTTPException(400, "Uploaded Excel token expired; please re-upload the file.")
        tmp = Path(info["path"])
        df_ids = _read_ids_frame(tmp)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        id_col = excel_id_col or df_ids.columns[0]
        wanted_ids = set(df_ids[id_col].astype(str).str.strip().tolist())
    if wanted_ids is not None:
        df = df[df[pk_col].astype(str).str.strip().isin(wanted_ids)].copy()

    t_map, t_name = await load_transcripts(transcripts)
    df["__tx"] = df[pk_col].astype(str).map(t_map).fillna("")
    df = df[df["__tx"].str.strip() != ""].copy()

    if df.empty:
        raise HTTPException(400, "No items with transcripts — nothing to process.")

    if limit and limit > 0:
        df = df.head(limit)

    uid = new_job()
    set_status(uid, "queued")
    # 🔑 Launch in a real background thread
    _start_batch_thread(uid, df, pk_col, primary_key, embed_fields, keep_ratio, brief, with_titles, t_map)
    # respond immediately; front-end will poll /segment/status/{uid}
    return {"uid": uid, "status": "queued"}


async def _run_batch_async(uid, df, pk_col, primary_key, embed_fields, keep_ratio, brief, with_titles, t_map):
    try:
        total = len(df)
        set_status(uuid, "running", f"starting… 0/{total}")
        out_dir = DATA_DIR / "zips" / uid
        out_dir.mkdir(parents=True, exist_ok=True)

        zip_paths = []
        manifest = []
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            row = r.to_dict()
            pk_value = str(row[pk_col])
            tx = t_map.get(pk_value, "").strip()
            if not tx:
                set_status(uid, "running", f"skip (no transcript) {i}/{total}")
                await asyncio.sleep(0)
                continue
            segs = await segment_text(tx, keep_ratio=keep_ratio, with_titles=with_titles, brief=brief)
            if not segs: 
                continue
            clip_embs = await embed_clips(segs)
            prog_emb  = await program_embedding(row, primary_key, embed_fields, segs)
            zpath = make_zip_for_program(uid, pk_value, row, primary_key, embed_fields, segs, clip_embs, prog_emb)
            zip_paths.append(zpath)
            manifest.append({"uid": uid, "pk": pk_value, "num_clips": len(segs), "zip": Path(zpath).name})
            set_status(uid, "running", f"processed {i}/{total}")
            await asyncio.sleep(0)  # tiny yield back to loop

        master_path = DATA_DIR / "zips" / f"{uid}.zip"
        with zipfile.ZipFile(master_path, "w", compression=zipfile.ZIP_DEFLATED) as master:
            for p in zip_paths:
                master.write(p, arcname=Path(p).name)
            if manifest:
                import io
                import pandas as pd
                buf = io.StringIO(); pd.DataFrame(manifest).to_csv(buf, index=False)
                master.writestr("manifest.csv", buf.getvalue())

        set_result(uid, {
            "uid": uid,
            "count": len(zip_paths),
            "master_zip": f"/api/download/zips/{master_path.name}",
            "zips": [f"/api/download/zips/{uid}/{Path(p).name}" for p in zip_paths],
        })
    except Exception as e:
        set_status(uid, "error", str(e))

def _start_batch_thread(uid, df, pk_col, primary_key, embed_fields, keep_ratio, brief, with_titles, t_map):
    def runner():
        # new thread → safe to create and run a private event loop
        asyncio.run(_run_batch_async(uid, df, pk_col, primary_key, embed_fields, keep_ratio, brief, with_titles, t_map))
    t = threading.Thread(target=runner, name=f"batch-{uid}", daemon=True)
    t.start()

@router.get("/segment/status/{uid}")
async def batch_status(uid: str):
    return get_job(uid)
