"""
Core configuration and application settings.
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    PROJECT_NAME: str = "Customer Feedback Analyzer (Production AI)"
    VERSION: str = "2.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # AI / LLM Configuration
    GOOGLE_API_KEY: str = Field(default="", validation_alias="GOOGLE_API_KEY")
    GEMINI_API_KEY: str = Field(default="", validation_alias="GEMINI_API_KEY")
    MODEL_NAME: str = "gemini-2.5-flash"
    
    # Semantic Caching
    ENABLE_SEMANTIC_CACHE: bool = True
    CACHE_SIMILARITY_THRESHOLD: float = 0.90
    CACHE_MAX_ENTRIES: int = 2000
    
    # Asynchronous Workers & Rate Limiting
    MAX_CONCURRENT_WORKERS: int = 5
    RATE_LIMIT_RPM: int = 60
    MAX_BATCH_SIZE: int = 5000
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///feedback.db"
    
    # Fallback / Offline Demo Mode (activates automatically if API key is not provided)
    ALLOW_MOCK_FALLBACK: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    @property
    def effective_api_key(self) -> str:
        return self.GOOGLE_API_KEY or self.GEMINI_API_KEY or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")


settings = Settings()


# Centralized structured logger
import logging
import sys

def setup_logger(name: str = "feedback_analyzer") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log

logger = setup_logger()


