"""Centralized configuration — reads from environment variables with sensible defaults.

Override any value by setting the corresponding env var in .env or your shell:

    QDRANT_COLLECTION=lok_sabha_questions_sample uv run python main.py
"""
from __future__ import annotations

import os
from pathlib import Path

# -- HuggingFace --
HF_DATASET_REPO: str = os.getenv("HF_DATASET_REPO", "opensansad/lok-sabha-qa")
HF_RAG_DATA_REPO: str = os.getenv("HF_RAG_DATA_REPO", "opensansad/lok-sabha-rag-data")

# -- Qdrant --
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "lok_sabha_questions")

# -- Embedding --
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# -- Data paths (RAG-specific outputs: chunks, snapshots, metadata.db) --
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
METADATA_DB_PATH: Path = Path(os.getenv("METADATA_DB_PATH", "data/metadata.db"))
