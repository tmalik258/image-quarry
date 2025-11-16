from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
)
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.dialects.postgresql import JSON as PG_JSON
from sqlalchemy.types import LargeBinary
from sqlalchemy.orm import declarative_mixin

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


@declarative_mixin
class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SegmentationJob(Base, TimestampMixin):
    __tablename__ = "segmentation_jobs"

    id = Column(String(36), primary_key=True, default=_uuid)
    status = Column(String(32), default="pending", nullable=False)
    model_type = Column(String(64), nullable=False)
    total_masks = Column(Integer, default=0, nullable=False)
    error_message = Column(String(512))
    batch_job_id = Column(String(36), ForeignKey("batch_jobs.id"), nullable=True)


class SegmentationMask(Base):
    __tablename__ = "segmentation_masks"

    id = Column(String(36), primary_key=True, default=_uuid)
    job_id = Column(String(36), ForeignKey("segmentation_jobs.id"), nullable=False)
    mask_index = Column(Integer, nullable=False)
    mask_data = Column(LargeBinary, nullable=False)
    confidence = Column(Float, default=1.0)
    # Use JSON type; works across SQLite/Postgres via dialect-specific types
    bbox = Column(SQLITE_JSON().with_variant(PG_JSON(), "postgresql"))
    area = Column(Integer)


class BatchJob(Base, TimestampMixin):
    __tablename__ = "batch_jobs"

    id = Column(String(36), primary_key=True, default=_uuid)
    status = Column(String(32), default="pending", nullable=False)
    total_images = Column(Integer, default=0, nullable=False)
    error_message = Column(String(512))
    task_id = Column(String(128))
    # Optional metadata
    model_type = Column(String(64))
    parameters = Column(SQLITE_JSON().with_variant(PG_JSON(), "postgresql"))
    completed_at = Column(DateTime)