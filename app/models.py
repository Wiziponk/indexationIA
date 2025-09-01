from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON
from .database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String, unique=True, index=True, nullable=False)
    label = Column(String, nullable=True)
    raw_path = Column(String, nullable=False)
    emb_path = Column(String, nullable=False)
    config = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
