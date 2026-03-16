#!/usr/bin/env bash
set -euo pipefail

COLLECTION="${QDRANT_COLLECTION:-lok_sabha_questions}"
QDRANT_URL="http://${QDRANT_HOST:-qdrant}:${QDRANT_PORT:-6333}"
HF_RAG_DATA="${HF_RAG_DATA_REPO:-opensansad/lok-sabha-rag-data}"

echo "=== Lok Sabha RAG (Docker) ==="
echo "Qdrant: $QDRANT_URL"
echo "Collection: $COLLECTION"
echo ""

# 1. Wait for Qdrant
echo "Waiting for Qdrant..."
for i in $(seq 1 60); do
    if curl -sf "$QDRANT_URL/readyz" > /dev/null 2>&1; then
        echo "Qdrant is ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Error: Qdrant did not become ready within 60 seconds."
        exit 1
    fi
    sleep 1
done
echo ""

# 2. Check if collection already has data
POINTS_COUNT=0
COLLECTION_EXISTS=$(curl -sf "$QDRANT_URL/collections/$COLLECTION" 2>/dev/null || echo "")
if [ -n "$COLLECTION_EXISTS" ]; then
    POINTS_COUNT=$(echo "$COLLECTION_EXISTS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('result',{}).get('points_count',0))" 2>/dev/null || echo "0")
fi

if [ "$POINTS_COUNT" -gt 0 ] 2>/dev/null; then
    echo "Collection '$COLLECTION' already has $POINTS_COUNT points. Skipping restore."
else
    # 3. Find or download snapshot
    SNAPSHOT_FILE=$(find data/snapshots -name '*.snapshot' 2>/dev/null | head -1)

    if [ -z "$SNAPSHOT_FILE" ]; then
        echo "No local snapshot found. Downloading from HuggingFace ($HF_RAG_DATA)..."
        echo "  This may take a few minutes on first run (~1.5 GB download)."
        mkdir -p data/snapshots data/hf_cache
        if uv run hf download "$HF_RAG_DATA" \
                "snapshots/lok_sabha_questions.snapshot" \
                --repo-type dataset \
                --local-dir data/hf_cache 2>&1; then
            DOWNLOADED="data/hf_cache/snapshots/lok_sabha_questions.snapshot"
            if [ -f "$DOWNLOADED" ]; then
                cp "$DOWNLOADED" data/snapshots/lok_sabha_questions.snapshot
                SNAPSHOT_FILE="data/snapshots/lok_sabha_questions.snapshot"
                echo "Snapshot downloaded successfully."
            fi
        else
            echo "HuggingFace download failed."
            echo "  The app will start but search will return no results."
            echo "  Upload a snapshot to $HF_RAG_DATA or place one in data/snapshots/"
        fi
    fi

    # 4. Restore snapshot
    if [ -n "$SNAPSHOT_FILE" ]; then
        SNAPSHOT_NAME=$(basename "$SNAPSHOT_FILE")
        SNAPSHOT_SIZE=$(du -h "$SNAPSHOT_FILE" | cut -f1)
        echo "Restoring Qdrant snapshot: $SNAPSHOT_NAME ($SNAPSHOT_SIZE)"
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X PUT \
            "$QDRANT_URL/collections/$COLLECTION/snapshots/recover" \
            -H "Content-Type: application/json" \
            -d "{\"location\": \"file:///snapshots/snapshots/$SNAPSHOT_NAME\"}")

        if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "202" ]; then
            echo "Snapshot restored successfully."
            sleep 2
            VERIFY=$(curl -sf "$QDRANT_URL/collections/$COLLECTION" 2>/dev/null || echo "")
            if [ -n "$VERIFY" ]; then
                RESTORED_POINTS=$(echo "$VERIFY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('result',{}).get('points_count',0))" 2>/dev/null || echo "?")
                echo "Collection now has $RESTORED_POINTS points."
            fi
        else
            echo "Warning: Snapshot restore failed (HTTP $HTTP_CODE)."
        fi
    fi
fi
echo ""

# 5. Metadata database
if [ -f data/metadata.db ]; then
    echo "Metadata database found."
else
    echo "Downloading metadata database from HuggingFace..."
    mkdir -p data/hf_cache
    if uv run hf download "$HF_RAG_DATA" \
            "metadata.db" \
            --repo-type dataset \
            --local-dir data/hf_cache 2>/dev/null; then
        DOWNLOADED_DB="data/hf_cache/metadata.db"
        if [ -f "$DOWNLOADED_DB" ]; then
            cp "$DOWNLOADED_DB" data/metadata.db
            echo "Metadata database downloaded."
        else
            echo "Warning: metadata.db download succeeded but file not found."
        fi
    else
        echo "Warning: metadata.db download failed. Stats features will not work."
    fi
fi
echo ""

echo "=== Starting server on http://0.0.0.0:8000 ==="
exec uv run uvicorn lok_sabha_rag.api.main:app --host 0.0.0.0 --port 8000
