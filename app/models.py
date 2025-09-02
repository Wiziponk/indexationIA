from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON

# --- SQLModel models (for Projects/Programs/Clips) ---
from sqlmodel import Column, Field, SQLModel


class Dataset(SQLModel, table=True):  # type: ignore[call-arg]
    uid: str = Field(primary_key=True)
    raw_path: str
    emb_path: str
    label: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # store the original config used to create the dataset
    config: dict = Field(default_factory=dict, sa_column=Column(JSON))


class Project(SQLModel, table=True):  # type: ignore[call-arg]
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    primary_key: str
    # Store selected embed fields as JSON
    embed_fields: list[str] = Field(sa_column=Column(JSON))
    # Defaults & knobs
    keep_ratio: float = 0.6
    with_titles: bool = True
    brief: Optional[str] = None
    mode: str = "api"  # "api" | "excel"
    excel_id_col: Optional[str] = None


class Run(SQLModel, table=True):  # type: ignore[call-arg]
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    uid: str  # job uid (from app.services.jobs)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "queued"  # queued|running|done|error
    note: Optional[str] = None
    master_zip_path: Optional[str] = None


class Program(SQLModel, table=True):  # type: ignore[call-arg]
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    pk_value: str
    # Selected fields snapshot for this emission (what we embedded)
    fields_json: dict = Field(sa_column=Column(JSON))
    transcript_name: Optional[str] = None
    transcript_text: Optional[str] = None
    num_clips: int = 0
    last_zip_path: Optional[str] = None


class Clip(SQLModel, table=True):  # type: ignore[call-arg]
    id: Optional[int] = Field(default=None, primary_key=True)
    program_id: int = Field(foreign_key="program.id")
    idx: int
    start: Optional[int] = None
    end: Optional[int] = None
    score: float = 0.0
    title: Optional[str] = None
    summary: Optional[str] = None
    text: Optional[str] = None


class Artifact(SQLModel, table=True):  # type: ignore[call-arg]
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id")
    kind: str  # "zip" | "parquet" | "npy" etc.
    path: str
