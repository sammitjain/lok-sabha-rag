#!/usr/bin/env bash
# Create a Qdrant snapshot, download it, and report stats.
#
# Usage:
#   bash scripts/snapshot.sh              # default: localhost:6333
#   bash scripts/snapshot.sh host:port    # custom Qdrant address

set -euo pipefail

HOST="${1:-localhost:6333}"
COLLECTION="lok_sabha_questions"
BASE="http://${HOST}"
DEST_DIR="data/snapshots"

echo "=== Qdrant Snapshot ==="
echo "Host: ${HOST}"
echo "Collection: ${COLLECTION}"
echo ""

# Report collection stats
POINTS=$(curl -sf "${BASE}/collections/${COLLECTION}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")
echo "Points in collection: $(printf "%'d" "${POINTS}")"
echo ""

# Create snapshot
echo "Creating snapshot..."
SNAPSHOT_NAME=$(curl -sf -X POST "${BASE}/collections/${COLLECTION}/snapshots" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['name'])")
echo "Snapshot created: ${SNAPSHOT_NAME}"

# Download
mkdir -p "${DEST_DIR}"
DEST="${DEST_DIR}/lok_sabha_questions.snapshot"
echo "Downloading to ${DEST}..."
curl -o "${DEST}" "${BASE}/collections/${COLLECTION}/snapshots/${SNAPSHOT_NAME}"

# Report
SIZE=$(du -h "${DEST}" | cut -f1)
echo ""
echo "=== Done ==="
echo "Points:   $(printf "%'d" "${POINTS}")"
echo "Snapshot: ${DEST} (${SIZE})"
echo ""
echo "Next: bash scripts/publish_rag_data.sh"
