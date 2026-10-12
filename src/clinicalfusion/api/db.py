"""Persistence: SQLite by default, PostgreSQL via DATABASE_URL (e.g. postgresql+psycopg2://user:pw@db/cf)."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./clinicalfusion.db")
ARTIFACT_DIR = os.environ.get("CF_ARTIFACT_DIR", "./artifacts")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(default=_now)
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    config: Mapped[dict] = mapped_column(JSON)
    results: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bundle_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    experiment: Mapped[str] = mapped_column(String(8), index=True)
    split: Mapped[str] = mapped_column(String(8))
    y_true: Mapped[int] = mapped_column(Integer)
    prob_raw: Mapped[float] = mapped_column(Float)
    prob_cal: Mapped[float] = mapped_column(Float)
    uncertainty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n_modalities: Mapped[int] = mapped_column(Integer)


_kwargs = {"connect_args": {"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, **_kwargs)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
