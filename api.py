"""
Enterprise FastAPI Service for Customer Feedback Analysis.

Supports:
- Legacy backward-compatible endpoint: POST /analyze
- Modern Async API v1:
  - Single ABSA Analysis: POST /api/v1/analyze
  - Distributed Batch Ingestion: POST /api/v1/jobs
  - Job Progress & Results: GET /api/v1/jobs/{job_id}
  - Server-Sent Events (SSE): GET /api/v1/jobs/{job_id}/stream
  - Analytics & Audit History: GET /api/v1/history, GET /api/v1/metrics
  - Health & Readiness: GET /api/v1/health
"""
from contextlib import asynccontextmanager
import asyncio
import json
from typing import Optional, List

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings, logger
from models.schemas import (
    Review,
    Analysis,
    AnalysisResult,
    BatchSubmissionRequest,
    BatchSubmissionResponse,
    BatchJobStatus,
    JobStatus,
)
from services.database import init_db, DatabaseRepository
from services.analyzer import analyzer
from services.cache import semantic_cache
from services.job_queue import job_manager
from services.clustering import clusterer
from services.executive_brief import brief_generator

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables
    logger.info("Initializing application and database schemas...")
    await init_db()
    yield
    # Shutdown
    logger.info("Application shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise-grade distributed customer feedback intelligence API with ABSA and semantic caching.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware for modern frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Legacy Route (Backward Compatibility)
# ==========================================
@app.post("/analyze", response_model=Analysis, tags=["Legacy"])
async def analyze_legacy(review: Review):
    """Legacy endpoint returning simple label, score, and theme."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, analyzer.analyze_single, review.text)
    return Analysis(
        label=result.overall_label.value,
        score=result.overall_score,
        theme=result.theme,
    )


# ==========================================
# Health & Telemetry Routes
# ==========================================
@app.get("/health", tags=["System"])
@app.get(f"{settings.API_V1_PREFIX}/health", tags=["System"])
async def health_check():
    """Health check endpoint indicating LLM readiness and cache metrics."""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "llm_mode": "Live Gemini" if analyzer.is_live_llm_ready() else "Heuristic Rule Fallback",
        "semantic_cache": semantic_cache.metrics,
    }


@app.get(f"{settings.API_V1_PREFIX}/metrics", tags=["Analytics"])
async def get_system_metrics():
    """System-wide operational metrics, cache hit rate, and aspect aggregates."""
    db_analytics = await DatabaseRepository.get_analytics()
    return {
        "system_metrics": semantic_cache.metrics,
        "feedback_analytics": db_analytics,
    }


# ==========================================
# Single Review Analysis (ABSA)
# ==========================================
@app.post(
    f"{settings.API_V1_PREFIX}/analyze",
    response_model=AnalysisResult,
    tags=["Analysis"],
)
async def analyze_comprehensive(
    review: Review,
    save_to_db: bool = Query(default=True, description="Persist result immediately to database"),
    bypass_cache: bool = Query(default=False, description="Force LLM inference bypassing semantic cache"),
):
    """
    Run Aspect-Based Sentiment Analysis (ABSA), urgency triage,
    and root-cause diagnosis on a single customer review.
    """
    loop = asyncio.get_running_loop()
    result: AnalysisResult = await loop.run_in_executor(
        None, analyzer.analyze_single, review.text, bypass_cache
    )

    if save_to_db:
        await DatabaseRepository.save_review(result)

    return result


# ==========================================
# Asynchronous Batch Jobs
# ==========================================
@app.post(
    f"{settings.API_V1_PREFIX}/jobs",
    response_model=BatchSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Batch Processing"],
)
async def submit_batch_job(payload: BatchSubmissionRequest):
    """
    Submit a bulk batch of customer reviews for asynchronous processing.
    Returns HTTP 202 Accepted with a job_id for status tracking.
    """
    clean_reviews = [r.strip() for r in payload.reviews if r.strip()]
    if not clean_reviews:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload must contain at least one non-empty review.",
        )

    if len(clean_reviews) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum limit of {settings.MAX_BATCH_SIZE}.",
        )

    job_id = await DatabaseRepository.create_job(total_reviews=len(clean_reviews))
    job_manager.enqueue_job(job_id=job_id, reviews=clean_reviews)

    estimated_seconds = round(len(clean_reviews) / settings.MAX_CONCURRENT_WORKERS * 0.2, 1)

    return BatchSubmissionResponse(
        job_id=job_id,
        total=len(clean_reviews),
        status="accepted",
        message="Batch job enqueued for distributed execution.",
        estimated_time_seconds=estimated_seconds,
    )


@app.get(
    f"{settings.API_V1_PREFIX}/jobs/{{job_id}}",
    response_model=BatchJobStatus,
    tags=["Batch Processing"],
)
async def get_job_status(job_id: str):
    """Query current execution progress and results for a batch job."""
    job_data = await DatabaseRepository.get_job(job_id)
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job_data


@app.get(
    f"{settings.API_V1_PREFIX}/jobs/{{job_id}}/stream",
    tags=["Batch Processing"],
)
async def stream_job_status(job_id: str):
    """
    Server-Sent Events (SSE) stream providing real-time progress updates
    until job completion.
    """
    async def event_generator():
        while True:
            job_data = await DatabaseRepository.get_job(job_id)
            if not job_data:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            payload = {
                "job_id": job_id,
                "status": job_data["status"],
                "total": job_data["total"],
                "completed": job_data["completed"],
                "failed": job_data["failed"],
                "progress_pct": job_data["progress_pct"],
            }
            yield f"data: {json.dumps(payload)}\n\n"

            if job_data["status"] in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==========================================
# History & Management Routes
# ==========================================
@app.get(f"{settings.API_V1_PREFIX}/history", tags=["History"])
async def get_feedback_history(
    limit: int = Query(default=100, ge=1, le=1000),
    label: Optional[str] = Query(default=None, description="Filter by positive, negative, or neutral"),
):
    """Retrieve historically analyzed feedback with aspect details."""
    return await DatabaseRepository.get_history(limit=limit, label_filter=label)


@app.post(f"{settings.API_V1_PREFIX}/cache/clear", tags=["System"])
async def clear_cache():
    """Clear in-memory semantic cache and reset metrics."""
    semantic_cache.clear()
    return {"status": "cache_cleared", "metrics": semantic_cache.metrics}


# ==========================================
# Executive Synthesis & Topic Clustering
# ==========================================
@app.post(f"{settings.API_V1_PREFIX}/jobs/{{job_id}}/executive-brief", tags=["Executive Intelligence"])
async def generate_job_executive_brief(job_id: str):
    """
    Synthesize an executive operational briefing and action plan
    across all reviews in a completed batch job.
    """
    job_data = await DatabaseRepository.get_job(job_id)
    if not job_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    results_dicts = [r.model_dump() for r in job_data["results"]]
    brief = brief_generator.generate_brief(results_dicts)
    return brief


@app.post(f"{settings.API_V1_PREFIX}/analytics/clusters", tags=["Analytics"])
async def compute_topic_clusters(
    limit: int = Query(default=100, ge=5, le=500),
    max_clusters: int = Query(default=4, ge=2, le=8),
):
    """
    Group customer reviews into 2D semantic topic clusters using
    vector embeddings, K-Means, and PCA projection.
    """
    history = await DatabaseRepository.get_history(limit=limit)
    if not history:
        return {"clusters": [], "points": []}
    return clusterer.cluster_reviews(history, max_clusters=max_clusters)
