"""Tests for verifying Qdrant indexing completeness."""

import json
import os
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "lok_sabha_questions"
DEFAULT_DATA_DIR = "data"

SERVER_HOST = os.getenv("QDRANT_HOST", "localhost")
SERVER_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def _server_available() -> bool:
    try:
        QdrantClient(host=SERVER_HOST, port=SERVER_PORT).get_collections()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def client():
    if not _server_available():
        pytest.skip(f"Qdrant server not available at {SERVER_HOST}:{SERVER_PORT}")
    return QdrantClient(host=SERVER_HOST, port=SERVER_PORT)


def discover_chunks_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("chunks.jsonl"))


def count_chunks_in_file(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("text") and record.get("chunk_id"):
                count += 1
    return count


def extract_session_info(path: Path) -> tuple[int, int] | None:
    """Extract lok_no and session_no from path like data/18/chunks/session_6/chunks.jsonl"""
    parts = path.parts
    try:
        chunks_idx = parts.index("chunks")
        lok_no = int(parts[chunks_idx - 1])
        session_dir = parts[chunks_idx + 1]
        session_no = int(session_dir.replace("session_", ""))
        return lok_no, session_no
    except (ValueError, IndexError):
        return None


def test_collection_exists(client):
    collections = {c.name for c in client.get_collections().collections}
    assert COLLECTION_NAME in collections, f"Collection {COLLECTION_NAME} not found"


def test_collection_not_empty(client):
    count = client.count(COLLECTION_NAME, exact=True).count
    assert count > 0, "Collection is empty"


def test_all_chunks_files_indexed(client):
    """Verify that chunks from all discovered chunks.jsonl files are present in Qdrant."""
    data_dir = Path(DEFAULT_DATA_DIR)
    chunks_files = discover_chunks_files(data_dir)

    if not chunks_files:
        pytest.skip("No chunks.jsonl files found in data/")

    total_expected = 0
    session_counts = {}

    for chunks_path in chunks_files:
        session_info = extract_session_info(chunks_path)
        if session_info is None:
            continue

        lok_no, session_no = session_info
        chunk_count = count_chunks_in_file(chunks_path)
        total_expected += chunk_count
        session_counts[(lok_no, session_no)] = chunk_count

    total_in_qdrant = client.count(COLLECTION_NAME, exact=True).count

    assert total_in_qdrant >= total_expected, (
        f"Expected at least {total_expected} points, found {total_in_qdrant}"
    )


def test_each_session_has_points(client):
    """Verify that each session in the source data has corresponding points in Qdrant."""
    data_dir = Path(DEFAULT_DATA_DIR)
    chunks_files = discover_chunks_files(data_dir)

    if not chunks_files:
        pytest.skip("No chunks.jsonl files found in data/")

    missing_sessions = []

    for chunks_path in chunks_files:
        session_info = extract_session_info(chunks_path)
        if session_info is None:
            continue

        lok_no, session_no = session_info
        expected_count = count_chunks_in_file(chunks_path)

        if expected_count == 0:
            continue

        session_filter = models.Filter(
            must=[
                models.FieldCondition(key="lok_no", match=models.MatchValue(value=lok_no)),
                models.FieldCondition(key="session_no", match=models.MatchValue(value=session_no)),
            ]
        )

        actual_count = client.count(
            COLLECTION_NAME,
            count_filter=session_filter,
            exact=True,
        ).count

        if actual_count < expected_count:
            missing_sessions.append({
                "lok_no": lok_no,
                "session_no": session_no,
                "expected": expected_count,
                "actual": actual_count,
            })

    assert not missing_sessions, f"Sessions with missing chunks: {missing_sessions}"


def test_indexed_chunks_have_text(client):
    """Verify that indexed chunks include the text field in payload."""
    sample = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
    )[0]

    assert len(sample) > 0, "No points returned from scroll"

    for point in sample:
        payload = point.payload or {}
        assert "text" in payload, f"Point {point.id} missing 'text' field"
        assert payload["text"], f"Point {point.id} has empty 'text' field"


def test_payload_has_required_fields(client):
    """Verify indexed chunks have required metadata fields."""
    required_fields = ["chunk_id", "lok_no", "session_no", "text"]

    sample = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
    )[0]

    assert len(sample) > 0, "No points returned from scroll"

    for point in sample:
        payload = point.payload or {}
        for field in required_fields:
            assert field in payload, f"Point {point.id} missing required field '{field}'"

