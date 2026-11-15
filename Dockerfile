# ==========================================
# Enterprise Multi-Service Dockerfile
# ==========================================
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.5.24 /uv /bin/uv

# Copy dependency manifests
COPY pyproject.toml uv.lock* ./

# Install project dependencies into system environment
RUN uv pip install -r pyproject.toml

# Copy project source code
COPY . .

# Expose ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000 8501

# Default command runs FastAPI backend
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

