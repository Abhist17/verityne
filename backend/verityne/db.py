"""SQLAlchemy models + session management.

SQLite by default (zero setup, the demo DB is a single committable file); set
VERITYNE_DB to a postgresql:// URL and the same schema runs unchanged.
"""
from __future__ import annotations

import datetime as dt
import json
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import DATABASE_URL

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    merchant_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    external_ref: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default="PROCESSING", index=True)

    selfie_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    id_doc_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Ground truth, only ever populated for eval/gauntlet fixtures. NULL in production.
    label: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)
    attack_type: Mapped[Optional[str]] = mapped_column(String(48), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(24), default="api", index=True)

    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    verdict = relationship("Verdict", back_populates="submission", uselist=False, cascade="all, delete-orphan")
    embeddings = relationship("FaceEmbedding", back_populates="submission", cascade="all, delete-orphan")
    hashes = relationship("AssetHash", back_populates="submission", cascade="all, delete-orphan")


class Verdict(Base):
    __tablename__ = "verdicts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id", ondelete="CASCADE"), index=True)

    final_score: Mapped[float] = mapped_column(Float, index=True)
    verdict: Mapped[str] = mapped_column(String(16), index=True)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)

    reasons: Mapped[list] = mapped_column(JSON, default=list)
    detector_breakdown: Mapped[dict] = mapped_column(JSON, default=dict)
    heatmaps: Mapped[dict] = mapped_column(JSON, default=dict)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    attack_pattern: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    generator_guess: Mapped[Optional[str]] = mapped_column(String(48), nullable=True, index=True)

    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    fusion_model: Mapped[str] = mapped_column(String(48), default="heuristic")
    policy_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    submission = relationship("Submission", back_populates="verdict")


class FaceEmbedding(Base):
    """512-d face embeddings, kept so cross-submission linkage can find one face wearing many names."""

    __tablename__ = "face_embeddings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id", ondelete="CASCADE"), index=True)
    merchant_id: Mapped[str] = mapped_column(String(64), index=True, default="default")
    kind: Mapped[str] = mapped_column(String(16), default="selfie")  # selfie | id_photo
    claimed_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    vector: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    submission = relationship("Submission", back_populates="embeddings")


class AssetHash(Base):
    """Asset hashes. Catches the same 'KYC kit' image resubmitted under new identities.

    Both hashes are stored: ``content_hash`` is exact and carries the "same file"
    claim, ``phash`` is perceptual and only ever a weak near-duplicate hint.
    """

    __tablename__ = "asset_hashes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="selfie")
    phash: Mapped[str] = mapped_column(String(32), index=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    submission = relationship("Submission", back_populates="hashes")


class BehavioralSession(Base):
    """How the form was filled, as opposed to what was uploaded.

    One row per form-fill session, not per event: the browser posts its raw
    buffer once, `detectors.behavioral.extract_features` reduces it, and only the
    reduction is stored. Keeping thousands of raw keystroke timings per applicant
    would be a biometric database with no retention story, and nothing downstream
    reads them - the features are sufficient for both scoring and audit.

    Keyed by an opaque token the page mints on load, because telemetry is posted
    while the applicant is still choosing files. `/verify` binds the token to the
    submission afterwards.
    """

    __tablename__ = "behavioral_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    submission_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), index=True, nullable=True
    )
    merchant_id: Mapped[str] = mapped_column(String(64), index=True, default="default")

    features: Mapped[dict] = mapped_column(JSON, default=dict)
    event_count: Mapped[int] = mapped_column(Integer, default=0)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submission_id: Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    actor: Mapped[str] = mapped_column(String(64), default="system")
    event: Mapped[str] = mapped_column(String(48), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)


Index("ix_verdict_score_time", Verdict.final_score, Verdict.created_at)


def init_db() -> None:
    Base.metadata.create_all(engine)
    _migrate()


def _migrate() -> None:
    """Add columns that create_all() will not add to an existing table.

    There is no Alembic here on purpose - the schema is small and the only
    migrations so far are additive. Each step is idempotent, so this is safe to
    run on every boot.
    """
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    if "asset_hashes" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("asset_hashes")}
    if "content_hash" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE asset_hashes ADD COLUMN content_hash VARCHAR(64) DEFAULT ''"))


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_db() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def log_event(session: Session, event: str, submission_id: str | None = None, actor: str = "system", **payload: Any) -> None:
    session.add(AuditEvent(submission_id=submission_id, actor=actor, event=event, payload=payload))
