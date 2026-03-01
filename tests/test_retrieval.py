"""Tests for Qdrant retrieval functionality."""

import os

import pytest
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "lok_sabha_questions"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

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


def test_collection_exists(client):
    collections = {c.name for c in client.get_collections().collections}
    assert COLLECTION_NAME in collections


def test_collection_not_empty(client):
    count = client.count(COLLECTION_NAME, exact=True).count
    assert count > 0


def test_semantic_query_returns_results(client):
    query = models.Document(
        text="black spots on national highways in Rajasthan",
        model=MODEL_NAME,
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query,
        limit=3,
    ).points

    assert len(results) > 0

    for point in results:
        assert point.id is not None
        assert point.score is not None
        assert isinstance(point.payload, dict)
        assert "subject" in point.payload
        assert "session_no" in point.payload


def test_query_returns_text_in_payload(client):
    query = models.Document(
        text="health ministry hospitals",
        model=MODEL_NAME,
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query,
        limit=3,
        with_payload=True,
    ).points

    assert len(results) > 0

    for point in results:
        payload = point.payload or {}
        assert "text" in payload, "Expected 'text' field in payload"
        assert payload["text"], "Expected non-empty 'text' field"
