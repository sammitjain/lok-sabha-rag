FROM python:3.11-slim

WORKDIR /app

# Install system deps (curl for health checks + snapshot restore)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Copy application source
COPY src/ src/
COPY frontend/ frontend/
COPY main.py ./
COPY scripts/docker-entrypoint.sh scripts/

# Create data directory (will be mounted as volume)
RUN mkdir -p data/snapshots

# Pre-download the embedding model so first search query is instant
RUN uv run python -c "\
from fastembed import TextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5')"

# Make entrypoint executable
RUN chmod +x scripts/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
