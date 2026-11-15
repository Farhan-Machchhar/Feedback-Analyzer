"""
Asynchronous Database Service using SQLAlchemy 2.0 with aiosqlite / asyncpg.
"""
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import uuid

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, func, desc, update
from sqlalchemy.orm import selectinload

from core.config import settings, logger
from models.entities import Base, JobEntity, ReviewEntity, AspectEntity
from models.schemas import AnalysisResult, AspectMention, SentimentLabel, UrgencyLevel, JobStatus

# Async Engine and Session
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)


async def init_db():
    """Create all tables asynchronously."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully.")


async def get_db_session() -> AsyncSession:
    """Dependency helper to yield async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


class DatabaseRepository:
    """Encapsulated database operations for jobs, reviews, and analytics."""

    @staticmethod
    async def create_job(total_reviews: int) -> str:
        async with AsyncSessionLocal() as session:
            job = JobEntity(
                id=str(uuid.uuid4()),
                status=JobStatus.PENDING.value,
                total=total_reviews,
                completed=0,
                failed=0,
                created_at=datetime.now(timezone.utc),
            )
            session.add(job)
            await session.commit()
            return job.id

    @staticmethod
    async def update_job_progress(job_id: str, completed_inc: int = 1, failed_inc: int = 0, status: Optional[str] = None):
        async with AsyncSessionLocal() as session:
            stmt = (
                update(JobEntity)
                .where(JobEntity.id == job_id)
                .values(
                    completed=JobEntity.completed + completed_inc,
                    failed=JobEntity.failed + failed_inc,
                )
            )
            if status:
                stmt = stmt.values(status=status)
            await session.execute(stmt)
            await session.commit()

            # Check if job is completed
            query = select(JobEntity).where(JobEntity.id == job_id)
            res = await session.execute(query)
            job = res.scalar_one_or_none()
            if job and (job.completed + job.failed >= job.total) and job.total > 0:
                job.status = JobStatus.COMPLETED.value
                job.completed_at = datetime.now(timezone.utc)
                await session.commit()

    @staticmethod
    async def save_review(result: AnalysisResult, job_id: Optional[str] = None) -> ReviewEntity:
        async with AsyncSessionLocal() as session:
            review_id = result.id or str(uuid.uuid4())
            review = ReviewEntity(
                id=review_id,
                job_id=job_id,
                text=result.text,
                overall_label=result.overall_label.value,
                overall_score=result.overall_score,
                theme=result.theme,
                urgency=result.urgency.value,
                root_cause=result.root_cause,
                suggested_action=result.suggested_action,
                source=result.source,
                latency_ms=result.latency_ms,
                created_at=datetime.now(timezone.utc),
            )
            session.add(review)

            for a in result.aspects:
                aspect_entity = AspectEntity(
                    review_id=review_id,
                    aspect=a.aspect,
                    sentiment=a.sentiment.value,
                    score=a.score,
                    evidence=a.evidence,
                )
                session.add(aspect_entity)

            await session.commit()
            return review

    @staticmethod
    async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            query = (
                select(JobEntity)
                .options(selectinload(JobEntity.reviews).selectinload(ReviewEntity.aspects))
                .where(JobEntity.id == job_id)
            )
            result = await session.execute(query)
            job = result.scalar_one_or_none()
            if not job:
                return None

            results_list = []
            for r in job.reviews:
                aspects = [
                    AspectMention(
                        aspect=a.aspect,
                        sentiment=SentimentLabel(a.sentiment),
                        score=a.score,
                        evidence=a.evidence or "",
                    )
                    for a in r.aspects
                ]
                results_list.append(
                    AnalysisResult(
                        id=r.id,
                        text=r.text,
                        overall_label=SentimentLabel(r.overall_label),
                        overall_score=r.overall_score,
                        theme=r.theme,
                        aspects=aspects,
                        urgency=UrgencyLevel(r.urgency),
                        root_cause=r.root_cause,
                        suggested_action=r.suggested_action,
                        source=r.source,
                        latency_ms=r.latency_ms or 0.0,
                        created_at=r.created_at.isoformat() if r.created_at else None,
                    )
                )

            progress_pct = round((job.completed / job.total * 100), 1) if job.total > 0 else 100.0

            return {
                "job_id": job.id,
                "status": job.status,
                "total": job.total,
                "completed": job.completed,
                "failed": job.failed,
                "progress_pct": progress_pct,
                "results": results_list,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            }

    @staticmethod
    async def get_history(limit: int = 100, label_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        async with AsyncSessionLocal() as session:
            query = (
                select(ReviewEntity)
                .options(selectinload(ReviewEntity.aspects))
                .order_by(desc(ReviewEntity.created_at))
                .limit(limit)
            )
            if label_filter and label_filter.lower() != "all":
                query = query.where(ReviewEntity.overall_label == label_filter.lower())

            res = await session.execute(query)
            reviews = res.scalars().all()

            output = []
            for r in reviews:
                output.append({
                    "id": r.id,
                    "review": r.text,
                    "label": r.overall_label,
                    "score": r.overall_score,
                    "theme": r.theme,
                    "urgency": r.urgency,
                    "root_cause": r.root_cause,
                    "suggested_action": r.suggested_action,
                    "source": r.source,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "aspects": [
                        {"aspect": a.aspect, "sentiment": a.sentiment, "score": a.score}
                        for a in r.aspects
                    ]
                })
            return output

    @staticmethod
    async def get_analytics() -> Dict[str, Any]:
        async with AsyncSessionLocal() as session:
            total_reviews = await session.scalar(select(func.count(ReviewEntity.id))) or 0
            if total_reviews == 0:
                return {
                    "total_reviews": 0,
                    "average_score": 0.0,
                    "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                    "urgency_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
                    "top_themes": [],
                    "aspect_averages": {},
                }

            avg_score = await session.scalar(select(func.avg(ReviewEntity.overall_score))) or 0.0

            # Label counts
            labels_query = select(ReviewEntity.overall_label, func.count(ReviewEntity.id)).group_by(ReviewEntity.overall_label)
            label_rows = (await session.execute(labels_query)).all()
            sentiment_dist = {lbl: cnt for lbl, cnt in label_rows}

            # Urgency counts
            urgency_query = select(ReviewEntity.urgency, func.count(ReviewEntity.id)).group_by(ReviewEntity.urgency)
            urgency_rows = (await session.execute(urgency_query)).all()
            urgency_dist = {urg: cnt for urg, cnt in urgency_rows}

            # Top themes
            themes_query = select(ReviewEntity.theme, func.count(ReviewEntity.id)).group_by(ReviewEntity.theme).order_by(desc(func.count(ReviewEntity.id))).limit(5)
            theme_rows = (await session.execute(themes_query)).all()
            top_themes = [{"theme": t, "count": cnt} for t, cnt in theme_rows]

            # Aspect score averages
            aspects_query = select(AspectEntity.aspect, func.avg(AspectEntity.score)).group_by(AspectEntity.aspect)
            aspect_rows = (await session.execute(aspects_query)).all()
            aspect_avgs = {asp: round(float(avg_s), 2) for asp, avg_s in aspect_rows if avg_s is not None}

            return {
                "total_reviews": total_reviews,
                "average_score": round(float(avg_score), 2),
                "sentiment_distribution": sentiment_dist,
                "urgency_distribution": urgency_dist,
                "top_themes": top_themes,
                "aspect_averages": aspect_avgs,
            }

