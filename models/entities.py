"""
SQLAlchemy 2.0 ORM Entities for database persistence.
"""
from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utc_now():
    return datetime.now(timezone.utc)


class JobEntity(Base):
    __tablename__ = "analysis_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(20), default="pending", nullable=False, index=True)
    total = Column(Integer, default=0, nullable=False)
    completed = Column(Integer, default=0, nullable=False)
    failed = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    reviews = relationship("ReviewEntity", back_populates="job", cascade="all, delete-orphan")


class ReviewEntity(Base):
    __tablename__ = "reviews"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(36), ForeignKey("analysis_jobs.id"), nullable=True, index=True)
    text = Column(Text, nullable=False)
    overall_label = Column(String(20), nullable=False, index=True)
    overall_score = Column(Integer, nullable=False, index=True)
    theme = Column(String(50), nullable=False, index=True)
    urgency = Column(String(20), default="low", nullable=False, index=True)
    root_cause = Column(Text, nullable=True)
    suggested_action = Column(Text, nullable=True)
    source = Column(String(20), default="llm", nullable=False)
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utc_now, nullable=False, index=True)

    job = relationship("JobEntity", back_populates="reviews")
    aspects = relationship("AspectEntity", back_populates="review", cascade="all, delete-orphan")


class AspectEntity(Base):
    __tablename__ = "review_aspects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(36), ForeignKey("reviews.id"), nullable=False, index=True)
    aspect = Column(String(50), nullable=False, index=True)
    sentiment = Column(String(20), nullable=False)
    score = Column(Integer, nullable=False)
    evidence = Column(Text, nullable=True)

    review = relationship("ReviewEntity", back_populates="aspects")

