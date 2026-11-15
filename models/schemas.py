"""
Pydantic data schemas for requests, responses, and internal domain models.
"""
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"  # Food safety, severe illness, legal threats, harassment


class AspectMention(BaseModel):
    aspect: str = Field(
        description="Category: food_quality, delivery_speed, service_staff, pricing, ambiance, cleanliness, app_experience"
    )
    sentiment: SentimentLabel
    score: int = Field(ge=1, le=5, description="Aspect score from 1 to 5")
    evidence: str = Field(description="Direct snippet or quote from the review")


class AnalysisResult(BaseModel):
    id: Optional[str] = None
    text: str
    overall_label: SentimentLabel
    overall_score: int = Field(ge=1, le=5)
    theme: str = Field(description="Primary topic, e.g. delivery, food_quality, service")
    aspects: List[AspectMention] = Field(default_factory=list)
    urgency: UrgencyLevel = UrgencyLevel.LOW
    root_cause: Optional[str] = Field(default=None, description="Underlying operational cause if negative")
    suggested_action: Optional[str] = Field(default=None, description="Recommended operational action for management")
    cached: bool = False
    source: str = "llm"  # "cache", "local_fast", "llm", "mock"
    latency_ms: float = 0.0
    created_at: Optional[str] = None

    # Backward compatibility properties with original project
    @property
    def label(self) -> str:
        return self.overall_label.value

    @property
    def score(self) -> int:
        return self.overall_score


# Legacy endpoints support
class Review(BaseModel):
    text: str


class Analysis(BaseModel):
    label: str
    score: int
    theme: str


# Batch Processing
class BatchSubmissionRequest(BaseModel):
    reviews: List[str] = Field(..., min_length=1, max_length=5000)


class BatchSubmissionResponse(BaseModel):
    job_id: str
    total: int
    status: str
    message: str
    estimated_time_seconds: float


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchJobStatus(BaseModel):
    job_id: str
    status: JobStatus
    total: int
    completed: int
    failed: int
    progress_pct: float
    results: List[AnalysisResult] = Field(default_factory=list)
    created_at: str
    completed_at: Optional[str] = None


class SystemMetrics(BaseModel):
    total_reviews_analyzed: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate_pct: float
    avg_latency_ms: float
    estimated_cost_saved_usd: float

