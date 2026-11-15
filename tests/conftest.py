"""
Pytest configuration and shared fixtures.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from api import app
from services.database import init_db
from services.cache import semantic_cache


@pytest_asyncio.fixture(autouse=True)
async def setup_test_env():
    """Initialize database and reset cache before each test."""
    semantic_cache.clear()
    await init_db()
    yield
    semantic_cache.clear()


@pytest_asyncio.fixture
async def async_client():
    """Async test HTTP client for FastAPI."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
