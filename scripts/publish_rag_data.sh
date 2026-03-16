#!/usr/bin/env bash
set -euo pipefail

# Upload Qdrant snapshot + metadata.db to HuggingFace.
#
# Usage:
#   bash scripts/publish_rag_data.sh
#   bash scripts/publish_rag_data.sh data/snapshots/lok_sabha_questions-2026-03-14.snapshot
#
# Prerequisites:
#   hf login

REPO="opensansad/lok-sabha-rag-data"

# Resolve snapshot file
if [ $# -ge 1 ]; then
    SNAPSHOT_FILE="$1"
else
    SNAPSHOT_FILE=$(find data/snapshots -name '*.snapshot' 2>/dev/null | sort | tail -1)
fi

if [ -z "$SNAPSHOT_FILE" ] || [ ! -f "$SNAPSHOT_FILE" ]; then
    echo "Error: No snapshot file found. Pass the path as an argument or place it in data/snapshots/"
    exit 1
fi

SNAPSHOT_SIZE=$(du -h "$SNAPSHOT_FILE" | cut -f1)
echo "=== Publishing RAG data to $REPO ==="
echo ""

# Upload snapshot
echo "Uploading snapshot: $SNAPSHOT_FILE ($SNAPSHOT_SIZE)..."
uv run hf upload "$REPO" \
    "$SNAPSHOT_FILE" "snapshots/lok_sabha_questions.snapshot" \
    --repo-type dataset
echo "Snapshot uploaded."
echo ""

# Upload metadata.db
METADATA_DB="data/metadata.db"
if [ ! -f "$METADATA_DB" ]; then
    echo "Warning: $METADATA_DB not found, skipping."
else
    METADATA_SIZE=$(du -h "$METADATA_DB" | cut -f1)
    echo "Uploading metadata: $METADATA_DB ($METADATA_SIZE)..."
    uv run hf upload "$REPO" \
        "$METADATA_DB" "metadata.db" \
        --repo-type dataset
    echo "Metadata uploaded."
fi

echo ""
echo "=== Done: https://huggingface.co/datasets/$REPO ==="
