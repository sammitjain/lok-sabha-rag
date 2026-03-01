#!/usr/bin/env bash
set -euo pipefail

# Collection name — reads from QDRANT_COLLECTION env var, defaults to lok_sabha_questions
COLLECTION="${QDRANT_COLLECTION:-lok_sabha_questions}"

echo "=== Lok Sabha RAG Setup ==="
echo "Collection: $COLLECTION"
echo ""

# 1. Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Error: docker is required."; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "Error: uv is required (https://docs.astral.sh/uv/)."; exit 1; }

# 2. Copy .env if missing
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
    echo "  -> Please add your OPENAI_API_KEY to .env before using AI synthesis."
    echo ""
fi

# 3. Install Python dependencies
echo "Installing Python dependencies..."
uv sync
echo ""

# 4. Start Qdrant
echo "Starting Qdrant via docker compose..."
docker compose up -d qdrant
echo "Waiting for Qdrant to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/readyz > /dev/null 2>&1; then
        echo "Qdrant is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Error: Qdrant did not become ready within 30 seconds."
        exit 1
    fi
    sleep 1
done
echo ""

# 5. Restore snapshot (or re-embed from sample chunks as fallback)
SNAPSHOT_FILE=$(find data/snapshots -name '*.snapshot' 2>/dev/null | head -1)

if [ -n "$SNAPSHOT_FILE" ]; then
    SNAPSHOT_NAME=$(basename "$SNAPSHOT_FILE")
    echo "Restoring Qdrant snapshot: $SNAPSHOT_NAME"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X PUT \
        "http://localhost:6333/collections/$COLLECTION/snapshots/recover" \
        -H "Content-Type: application/json" \
        -d "{\"location\": \"/snapshots/$SNAPSHOT_NAME\"}")

    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "202" ]; then
        echo "Snapshot restored successfully."
    else
        echo "Snapshot restore failed (HTTP $HTTP_CODE). Falling back to re-embed..."
        echo "Embedding sample chunks into Qdrant (this downloads the model on first run)..."
        uv run python -m lok_sabha_rag.pipeline.embed_index_qdrant2 --data-dir data/sample --collection "$COLLECTION"
    fi
else
    echo "No snapshot found. Embedding sample chunks into Qdrant..."
    echo "(This downloads the embedding model on first run, ~50 MB.)"
    uv run python -m lok_sabha_rag.pipeline.embed_index_qdrant2 --data-dir data/sample --collection "$COLLECTION"
fi
echo ""

echo "=== Setup complete ==="
echo ""
echo "Start the server:"
echo "  uv run python main.py"
echo ""
echo "Then open: http://localhost:8000"
