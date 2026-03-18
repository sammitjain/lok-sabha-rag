# Recipes

Quick reference for common pipeline operations. All commands assume you're in the project root.

## End-to-End: Add a New Session

Replace `--lok` and `--sessions` with the target Lok Sabha and session number.

### 1. Curate — index questions to download

```bash
cd ../lok-sabha-dataset
uv run python -m lok_sabha_dataset.pipeline.curate --lok 18 --sessions 7 --force
```

### 2. Download — fetch PDFs

```bash
uv run python -m lok_sabha_dataset.pipeline.download --lok 18
```

### 3. Extract — OCR / text extraction (two passes)

```bash
uv run python -m lok_sabha_dataset.pipeline.extract run --lok 18 --sessions 7 --engine docling
uv run python -m lok_sabha_dataset.pipeline.extract run --lok 18 --sessions 7 --engine easyocr --retry-empty
```

### 4. Build & publish dataset to HuggingFace

```bash
uv run python -m lok_sabha_dataset.build
uv run python -m lok_sabha_dataset.publish --push
```

### 5. Build chunks (back in RAG repo)

```bash
cd ../lok-sabha-rag
uv run python -m lok_sabha_rag.pipeline.build_chunks
```

### 6. Embed — index chunks into Qdrant

Embed all new chunks:

```bash
uv run python -m lok_sabha_rag.pipeline.embed --files data/18/chunks/session_7/chunks.jsonl
```

Multiple files:

```bash
uv run python -m lok_sabha_rag.pipeline.embed \
  --files data/17/chunks/session_14/chunks.jsonl \
  --files data/17/chunks/session_15/chunks.jsonl
```

### 7. Rebuild metadata DB

```bash
uv run python -m lok_sabha_rag.pipeline.build_metadata_db
```

### 8. Create and download Qdrant snapshot

```bash
bash scripts/snapshot.sh
```

### 9. Publish snapshot + metadata to HuggingFace

```bash
uv run hf login          # one-time
bash scripts/publish_rag_data.sh
```

## Standalone Recipes

### Run the app locally (without Docker)

```bash
bash scripts/setup.sh
uv run python main.py
# http://localhost:8000
```

### Run the app with Docker

```bash
cp .env.example .env    # add OPENAI_API_KEY
docker compose up
# http://localhost:8000
```
