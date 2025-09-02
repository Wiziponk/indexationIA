from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db, Base, engine
from ..models import Dataset
from ..services.api_client import fetch_all_programs
from ..services.embeddings import batch_embed, build_text_from_fields
from ..services.storage import save_dataset_files
from ..services.utils import get_nested_value

Base.metadata.create_all(bind=engine)

router = APIRouter()
def _ds_to_dict(ds: Dataset) -> dict:
    return {
        "uid": ds.uid,
        "raw_path": ds.raw_path,
        "emb_path": ds.emb_path,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
        "label": ds.label,
        "config": ds.config,
    }


class DatasetUpdate(BaseModel):
    label: Optional[str] = None

@router.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    rows = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    return [_ds_to_dict(d) for d in rows]

@router.put("/datasets/{uid}")
def update_dataset(uid: str, payload: DatasetUpdate, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.uid == uid).first()
    if not ds:
        raise HTTPException(404, "Not found")
    ds.label = payload.label
    db.commit()
    db.refresh(ds)
    return _ds_to_dict(ds)

@router.delete("/datasets/{uid}", status_code=204)
def delete_dataset(uid: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.uid == uid).first()
    if not ds:
        raise HTTPException(404, "Not found")
    db.delete(ds)
    db.commit()
    return

@router.post("/datasets/{uid}/rerun")
async def rerun_dataset(uid: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.uid == uid).first()
    if not ds:
        raise HTTPException(404, "Not found")
    cfg = ds.config or {}
    mode = cfg.get("mode", "api")
    primary_key = cfg.get("primary_key")
    embed_fields = cfg.get("embed_fields", [])
    if mode != "api":
        raise HTTPException(400, "Re-run only supported for API mode")
    try:
        programs = fetch_all_programs()
    except Exception as e:
        raise HTTPException(400, f"API error while fetching programs: {e}")
    if not programs:
        raise HTTPException(400, "The catalog API returned no data.")
    df = pd.DataFrame(programs)
    if "." in primary_key or primary_key not in df.columns:
        df["_pk"] = df.apply(lambda r: get_nested_value(r.to_dict(), primary_key), axis=1)
        pk_col = "_pk"
    else:
        pk_col = primary_key
    texts = [build_text_from_fields(row.to_dict(), embed_fields) for _, row in df.iterrows()]
    df["_text_for_embedding"] = texts
    df = df[df["_text_for_embedding"].astype(str).str.strip() != ""]
    if df.empty:
        raise HTTPException(400, "Nothing to embed after field selection.")
    embs = await batch_embed(df["_text_for_embedding"].tolist())
    X = np.vstack(embs)
    new_uid = str(uuid.uuid4())[:8]
    raw_name, emb_name = save_dataset_files(df, X, new_uid)
    new_ds = Dataset(uid=new_uid, raw_path=raw_name, emb_path=emb_name, config=cfg, label=ds.label)
    db.add(new_ds)
    db.commit()
    db.refresh(new_ds)
    return new_ds
