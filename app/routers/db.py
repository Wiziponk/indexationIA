from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import desc, func
from sqlmodel import Session, delete, select

from ..db import get_session
from ..models import Clip, Program, Project
from ..schemas import (
    ClipUpdateResponse,
    ErrorResponse,
    ProgramDetailResponse,
    ProgramRerunResponse,
    ProgramsResponse,
    ProjectsResponse,
)
from ..services.clipmaker import (
    embed_clips,
    make_zip_for_program,
    program_embedding,
    segment_text,
)

router = APIRouter()

ERROR_RESPONSES = {400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}}


# ---------- Projects ----------
@router.get("/projects", response_model=ProjectsResponse, responses=ERROR_RESPONSES)
def list_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project).order_by(desc(Project.created_at))).all()
    rows = []
    for p in projects:
        n = session.exec(
            select(func.count()).select_from(Program).where(Program.project_id == p.id)
        ).scalar_one()[0]

        rows.append(
            {
                "id": p.id,
                "name": p.name,
                "created_at": p.created_at.isoformat(),
                "primary_key": p.primary_key,
                "embed_fields": p.embed_fields,
                "keep_ratio": p.keep_ratio,
                "with_titles": p.with_titles,
                "brief": p.brief,
                "mode": p.mode,
                "excel_id_col": p.excel_id_col,
                "programs": n,
            }
        )
    return {"projects": rows}


# ---------- Programs ----------
@router.get("/programs", response_model=ProgramsResponse, responses=ERROR_RESPONSES)
def list_programs(project_id: int, session: Session = Depends(get_session)):
    progs = session.exec(
        select(Program)
        .where(Program.project_id == project_id)
        .order_by(Program.pk_value)
    ).all()
    return {
        "programs": [
            {
                "id": pr.id,
                "pk_value": pr.pk_value,
                "num_clips": pr.num_clips,
                "last_zip_path": pr.last_zip_path,
            }
            for pr in progs
        ]
    }


@router.get(
    "/programs/{program_id}",
    response_model=ProgramDetailResponse,
    responses=ERROR_RESPONSES,
)
def get_program(program_id: int, session: Session = Depends(get_session)):
    pr = session.get(Program, program_id)
    if not pr:
        raise HTTPException(404, "Program not found")
    clips = session.exec(
        select(Clip).where(Clip.program_id == program_id).order_by(Clip.idx)
    ).all()
    return {
        "program": {
            "id": pr.id,
            "project_id": pr.project_id,
            "pk_value": pr.pk_value,
            "fields_json": pr.fields_json,
            "transcript_name": pr.transcript_name,
            "num_clips": pr.num_clips,
            "last_zip_path": pr.last_zip_path,
        },
        "clips": [
            {
                "id": c.id,
                "idx": c.idx,
                "start": c.start,
                "end": c.end,
                "score": c.score,
                "title": c.title,
                "summary": c.summary,
                "text": c.text,
            }
            for c in clips
        ],
    }


# ---------- Clips ----------
@router.patch(
    "/clips/{clip_id}", response_model=ClipUpdateResponse, responses=ERROR_RESPONSES
)
def update_clip(
    clip_id: int,
    payload: dict = Body(...),
    session: Session = Depends(get_session),
):
    c = session.get(Clip, clip_id)
    if not c:
        raise HTTPException(404, "Clip not found")
    for key in ["title", "summary", "text", "score"]:
        if key in payload:
            setattr(c, key, payload[key])
    session.add(c)
    session.commit()
    session.refresh(c)
    return {
        "ok": True,
        "clip": {
            "id": c.id,
            "idx": c.idx,
            "title": c.title,
            "summary": c.summary,
            "score": c.score,
        },
    }


# ---------- Rerun segmentation for one program ----------
@router.post(
    "/programs/{program_id}/rerun",
    response_model=ProgramRerunResponse,
    responses=ERROR_RESPONSES,
)
async def rerun_program(
    program_id: int,
    keep_ratio: Optional[float] = Body(None),
    with_titles: Optional[bool] = Body(None),
    brief: Optional[str] = Body(None),
    session: Session = Depends(get_session),
):
    pr = session.get(Program, program_id)
    if not pr:
        raise HTTPException(404, "Program not found")
    project = session.get(Project, pr.project_id)
    if not project:
        raise HTTPException(400, "Owning project missing")
    if not pr.transcript_text:
        raise HTTPException(400, "No transcript stored for this program; cannot rerun")

    # Effective knobs
    eff_keep = float(keep_ratio if keep_ratio is not None else project.keep_ratio)
    eff_titles = bool(project.with_titles if with_titles is None else with_titles)
    eff_brief = brief if (brief is not None and brief.strip()) else project.brief

    # Segment → embed → program embed
    segs = await segment_text(
        pr.transcript_text, keep_ratio=eff_keep, with_titles=eff_titles, brief=eff_brief
    )
    clip_embs = await embed_clips(segs)
    program_row = pr.fields_json or {}
    prog_emb = await program_embedding(
        program_row, project.primary_key, project.embed_fields, segs
    )

    # Persist: replace old clips
    session.exec(delete(Clip).where(Clip.program_id == pr.id))
    for i, s in enumerate(segs, start=1):
        session.add(
            Clip(
                program_id=pr.id,
                idx=i,
                start=s.get("start"),
                end=s.get("end"),
                score=float(s.get("score", 0.0)),
                title=s.get("title"),
                summary=s.get("summary"),
                text=s.get("text"),
            )
        )
    pr.num_clips = len(segs)

    # Export a fresh program ZIP for this rerun
    uid = "rerun-" + uuid.uuid4().hex[:6]
    zpath = make_zip_for_program(
        uid,
        pr.pk_value,
        program_row,
        project.primary_key,
        project.embed_fields,
        segs,
        clip_embs,
        prog_emb,
    )
    pr.last_zip_path = f"/api/download/zips/{uid}/{Path(zpath).name}"

    session.add(pr)
    session.commit()
    return {
        "ok": True,
        "program_id": pr.id,
        "num_clips": pr.num_clips,
        "zip": pr.last_zip_path,
    }
