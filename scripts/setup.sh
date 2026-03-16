#!/usr/bin/env bash
set -euo pipefail

# Collection name — reads from QDRANT_COLLECTION env var, defaults to lok_sabha_questions
COLLECTION="${QDRANT_COLLECTION:-lok_sabha_questions}"
HF_RAG_DATA="${HF_RAG_DATA_REPO:-opensansad/lok-sabha-rag-data}"

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

# 5. Restore or build Qdrant collection
#    Priority: existing collection > local snapshot > HF download > sample build

POINTS_COUNT=0
COLLECTION_EXISTS=$(curl -sf "http://localhost:6333/collections/$COLLECTION" 2>/dev/null || echo "")
if [ -n "$COLLECTION_EXISTS" ]; then
    POINTS_COUNT=$(echo "$COLLECTION_EXISTS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('result',{}).get('points_count',0))" 2>/dev/null || echo "0")
fi

if [ "$POINTS_COUNT" -gt 0 ] 2>/dev/null; then
    echo "Collection '$COLLECTION' already has $POINTS_COUNT points. Skipping snapshot restore."
else
    SNAPSHOT_FILE=$(find data/snapshots -name '*.snapshot' 2>/dev/null | head -1)

    # Try downloading from HuggingFace if no local snapshot
    if [ -z "$SNAPSHOT_FILE" ]; then
        echo "No local snapshot found. Downloading from HuggingFace ($HF_RAG_DATA)..."
        mkdir -p data/snapshots
        if HF_SNAPSHOT_PATH=$(uv run hf download "$HF_RAG_DATA" \
                "snapshots/lok_sabha_questions.snapshot" \
                --repo-type dataset \
                --local-dir data/hf_cache 2>&1); then
            # huggingface-cli download with --local-dir puts the file at the same relative path
            DOWNLOADED="data/hf_cache/snapshots/lok_sabha_questions.snapshot"
            if [ -f "$DOWNLOADED" ]; then
                cp "$DOWNLOADED" data/snapshots/lok_sabha_questions.snapshot
                SNAPSHOT_FILE="data/snapshots/lok_sabha_questions.snapshot"
                echo "Snapshot downloaded successfully."
            fi
        else
            echo "HuggingFace download failed. Will fall back to sample build."
            echo "  (To use the full dataset, run: uv run hf login)"
        fi
    fi

    if [ -n "$SNAPSHOT_FILE" ]; then
        SNAPSHOT_NAME=$(basename "$SNAPSHOT_FILE")
        SNAPSHOT_SIZE=$(du -h "$SNAPSHOT_FILE" | cut -f1)
        echo "Restoring Qdrant snapshot: $SNAPSHOT_NAME ($SNAPSHOT_SIZE)"
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X PUT \
            "http://localhost:6333/collections/$COLLECTION/snapshots/recover" \
            -H "Content-Type: application/json" \
            -d "{\"location\": \"file:///snapshots/$SNAPSHOT_NAME\"}")

        if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "202" ]; then
            echo "Snapshot restored successfully."
            # Verify point count
            sleep 2
            VERIFY=$(curl -sf "http://localhost:6333/collections/$COLLECTION" 2>/dev/null || echo "")
            if [ -n "$VERIFY" ]; then
                RESTORED_POINTS=$(echo "$VERIFY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('result',{}).get('points_count',0))" 2>/dev/null || echo "?")
                echo "Collection now has $RESTORED_POINTS points."
            fi
        else
            echo "Snapshot restore failed (HTTP $HTTP_CODE). Falling back to sample build..."
            echo ""
            echo "Building sample chunks from HuggingFace dataset (50 questions)..."
            uv run python -m lok_sabha_rag.pipeline.build_chunks --max-files 50 --data-dir data/sample
            echo ""
            echo "Embedding sample chunks into Qdrant (downloads model on first run, ~50 MB)..."
            uv run python -m lok_sabha_rag.pipeline.embed --data-dir data/sample --collection "$COLLECTION" --overwrite
        fi
    else
        echo "No snapshot available. Building sample dataset..."
        echo ""
        echo "Building sample chunks from HuggingFace dataset (50 questions)..."
        uv run python -m lok_sabha_rag.pipeline.build_chunks --max-files 50 --data-dir data/sample
        echo ""
        echo "Embedding sample chunks into Qdrant (downloads model on first run, ~50 MB)..."
        uv run python -m lok_sabha_rag.pipeline.embed --data-dir data/sample --collection "$COLLECTION" --overwrite
    fi
fi
echo ""

# 6. Metadata database
if [ -f data/metadata.db ]; then
    echo "Metadata database already exists at data/metadata.db. Skipping rebuild."
else
    echo "Downloading metadata database from HuggingFace..."
    if uv run hf download "$HF_RAG_DATA" \
            "metadata.db" \
            --repo-type dataset \
            --local-dir data/hf_cache 2>/dev/null; then
        DOWNLOADED_DB="data/hf_cache/metadata.db"
        if [ -f "$DOWNLOADED_DB" ]; then
            cp "$DOWNLOADED_DB" data/metadata.db
            echo "Metadata database downloaded."
        else
            echo "Download succeeded but file not found. Building from scratch..."
            uv run python -m lok_sabha_rag.pipeline.build_metadata_db
        fi
    else
        echo "Download failed. Building metadata database from scratch..."
        uv run python -m lok_sabha_rag.pipeline.build_metadata_db
    fi
fi
echo ""

echo "=== Setup complete ==="
echo ""
echo "Start the server:"
echo "  uv run python main.py"
echo ""
echo "Then open: http://localhost:8000"
