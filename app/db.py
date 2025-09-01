from __future__ import annotations

from pathlib import Path
from sqlmodel import SQLModel, Session, create_engine
from .config import DATA_DIR

DB_PATH = (DATA_DIR / "indexation.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

def init_db() -> None:
    # Import models so SQLModel sees the tables
    from . import models  # noqa: F401
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

