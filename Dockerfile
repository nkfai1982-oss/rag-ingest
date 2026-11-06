FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency specifications first for cache optimization
COPY pyproject.toml README.md ./

# Install project dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code
COPY src/ src/

# Sync project package
RUN uv sync --frozen --no-dev

# Ensure data directories exist
RUN mkdir -p /app/data/uploads

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "rag_ingest.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
