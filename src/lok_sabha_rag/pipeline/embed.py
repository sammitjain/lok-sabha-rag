#!/usr/bin/env python3
"""Embed chunks and index into Qdrant.

Reads all chunks.jsonl files under data/ and indexes them into a Qdrant
collection using FastEmbed (BAAI/bge-small-en-v1.5).

Usage:
    uv run python -m lok_sabha_rag.pipeline.embed
    uv run python -m lok_sabha_rag.pipeline.embed --data-dir data --overwrite
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from lok_sabha_rag.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    EMBEDDING_MODEL, DATA_DIR,
)

app = typer.Typer()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def discover_chunks_files(data_dir: Path) -> List[Path]:
    """Find all chunks.jsonl files under data_dir recursively."""
    return sorted(data_dir.rglob("chunks.jsonl"))


def build_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build Qdrant payload from chunk record, including full text."""
    meta = record.get("meta", {}) or {}
    source = record.get("source", {}) or {}
    pipeline = record.get("pipeline", {}) or {}

    return {
        "chunk_id": record.get("chunk_id"),
        "text": record.get("text", ""),
        "meta": meta,
        "source": source,
        "pipeline": pipeline,
        "lok_no": meta.get("lok_no"),
        "session_no": meta.get("session_no"),
        "ques_no": meta.get("ques_no"),
        "type": meta.get("type"),
        "date": meta.get("date"),
        "ministry": meta.get("ministry"),
        "mp_names": meta.get("mp_names"),
        "subject": meta.get("subject"),
        "pdf_filename": source.get("pdf_filename"),
        "pdf_url": source.get("pdf_url"),
        "chunk_index": source.get("chunk_index"),
    }


def ensure_collection(client: QdrantClient, model_name: str, collection: str) -> None:
    """Create collection if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        return

    size = client.get_embedding_size(model_name)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=size,
            distance=models.Distance.COSINE,
        ),
    )


def _count_points(client: QdrantClient, collection: str, exact: bool = False) -> int:
    """Count total points in collection."""
    return client.count(collection_name=collection, exact=exact).count


def _progress_poller(
    client: QdrantClient,
    collection: str,
    pbar: tqdm,
    stop_event: threading.Event,
    initial_count: int,
    interval_s: float = 0.75,
    exact_every_s: float = 45.0,
) -> None:
    """Poll Qdrant for point count and update progress bar."""
    last = initial_count
    next_exact = time.time() + exact_every_s

    while not stop_event.is_set():
        try:
            now = time.time()
            use_exact = now >= next_exact
            current = _count_points(client, collection, exact=use_exact)
            if use_exact:
                next_exact = now + exact_every_s

            delta = max(0, current - last)
            if delta:
                pbar.update(delta)
                last = current
        except Exception:
            pass

        stop_event.wait(interval_s)


def load_chunks_from_file(chunks_path: Path) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Load chunks from a JSONL file."""
    texts: List[str] = []
    ids: List[str] = []
    payloads: List[Dict[str, Any]] = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record.get("text")
            chunk_id = record.get("chunk_id")
            if not text or not chunk_id:
                continue
            texts.append(text)
            ids.append(chunk_id)
            payloads.append(build_payload(record))

    return texts, ids, payloads


def index_chunks(
    client: QdrantClient,
    collection: str,
    model: str,
    texts: List[str],
    ids: List[str],
    payloads: List[Dict[str, Any]],
    batch_size: int,
    desc: str,
) -> int:
    """Index chunks into Qdrant with progress tracking."""
    expected = len(texts)
    if expected == 0:
        return 0

    initial_count = _count_points(client, collection, exact=True)

    stop_event = threading.Event()
    pbar = tqdm(total=expected, desc=desc, unit="chunks", dynamic_ncols=True)

    poll_thread = threading.Thread(
        target=_progress_poller,
        args=(client, collection, pbar, stop_event, initial_count),
        daemon=True,
    )
    poll_thread.start()

    try:
        vectors = [models.Document(text=t, model=model) for t in texts]
        client.upload_collection(
            collection_name=collection,
            vectors=vectors,
            ids=ids,
            payload=payloads,
            batch_size=batch_size,
            parallel=4,
        )
    finally:
        stop_event.set()
        poll_thread.join(timeout=2.0)
        final_count = _count_points(client, collection, exact=True)
        indexed_this_batch = final_count - initial_count
        if indexed_this_batch > pbar.n:
            pbar.update(indexed_this_batch - pbar.n)
        pbar.close()

    return expected


@app.command()
def run(
    data_dir: str = typer.Option(str(DATA_DIR), help="Base data directory to scan for chunks.jsonl"),
    collection: str = typer.Option(QDRANT_COLLECTION, help="Qdrant collection name"),
    model: str = typer.Option(EMBEDDING_MODEL, help="Embedding model"),
    batch_size: int = typer.Option(512, help="Chunks per batch"),
    host: str = typer.Option(QDRANT_HOST, help="Qdrant host"),
    port: int = typer.Option(QDRANT_PORT, help="Qdrant port"),
    overwrite: bool = typer.Option(False, help="Delete existing collection before indexing"),
) -> None:
    """Index all chunks.jsonl files found under data_dir into Qdrant."""
    data_path = Path(data_dir)
    chunks_files = discover_chunks_files(data_path)

    if not chunks_files:
        logger.error("No chunks.jsonl files found under %s", data_path)
        raise typer.Exit(code=1)

    logger.info("Found %d chunks.jsonl files to index", len(chunks_files))
    logger.info("Target collection: %s", collection)
    for f in chunks_files:
        logger.info("  - %s", f)

    client = QdrantClient(host=host, port=port)

    if overwrite:
        existing = {c.name for c in client.get_collections().collections}
        if collection in existing:
            logger.info("Deleting existing collection: %s", collection)
            client.delete_collection(collection)

    ensure_collection(client, model, collection)

    total_indexed = 0
    start_time = time.time()

    for chunks_path in chunks_files:
        logger.info("Loading %s", chunks_path)
        texts, ids, payloads = load_chunks_from_file(chunks_path)

        if not texts:
            logger.warning("No valid chunks in %s", chunks_path)
            continue

        logger.info("Indexing %d chunks from %s", len(texts), chunks_path.relative_to(data_path))
        indexed = index_chunks(
            client=client,
            collection=collection,
            model=model,
            texts=texts,
            ids=ids,
            payloads=payloads,
            batch_size=batch_size,
            desc=str(chunks_path.relative_to(data_path)),
        )
        total_indexed += indexed

        elapsed = time.time() - start_time
        rate = total_indexed / elapsed if elapsed > 0 else 0
        logger.info("Progress: indexed=%d rate=%.1f chunks/s", total_indexed, rate)

    final_count = _count_points(client, collection, exact=True)
    logger.info("Indexing complete. Total chunks indexed this run: %d, Collection total: %d", total_indexed, final_count)


if __name__ == "__main__":
    app()
