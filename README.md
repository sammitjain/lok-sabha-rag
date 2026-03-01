# Lok Sabha Q&A RAG

A Retrieval-Augmented Generation system for searching and synthesizing answers from Indian Lok Sabha (parliamentary) question-and-answer records.

## Features

- **Semantic search** across 86,000+ parliamentary questions (Lok Sabha 17 & 18)
- **AI-powered answer synthesis** with `[Q#]` citations back to source PDFs (GPT-4o-mini)
- **MP activity statistics** from metadata database (question counts, ministry breakdowns)
- **Filter by** Lok Sabha number, session, ministry, or MP name
- **Lazy-loaded question text** — card content fetched on demand from Qdrant

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [Docker](https://docs.docker.com/get-docker/) (for Qdrant vector database)
- OpenAI API key (for AI synthesis mode)

### Setup

```bash
git clone <repo-url> && cd lok-sabha-rag
./scripts/setup.sh
# Edit .env to add your OPENAI_API_KEY
uv run python main.py
# Open http://localhost:8000
```

### Manual Setup

```bash
cp .env.example .env              # add your OPENAI_API_KEY
uv sync                           # install Python dependencies
docker compose up -d               # start Qdrant on localhost:6333

# Load sample data into Qdrant (100 questions from Lok Sabha 18 Session 7)
uv run python -m lok_sabha_rag.pipeline.embed_index_qdrant2 --data-dir data/sample

# Start server
uv run python main.py             # http://localhost:8000
```

## Architecture

```
src/lok_sabha_rag/
├── api/           # FastAPI routes (search, synthesize, stats, members, question-text)
├── core/          # Retriever (Qdrant), Synthesizer (OpenAI), Stats (SQLite)
├── prompts/       # LLM system + user prompt templates
└── pipeline/      # 5-stage data ingestion (curate -> download -> extract -> chunk -> embed)
```

- **Vector DB**: Qdrant (384-dim BAAI/bge-small-en-v1.5 embeddings, cosine distance)
- **LLM**: GPT-4o-mini via OpenAI API
- **Metadata**: SQLite database with question + MP data
- **Frontend**: Vanilla JS/CSS single-page app

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full reference.

## Sample Data

The repo includes a starter dataset of 100 questions (~1,074 chunks) from Lok Sabha 18 Session 7 at `data/sample/`. This lets you run the app immediately after setup.

To index the full dataset, run the [data pipeline](docs/DATA_PIPELINE.md).

## Data Pipeline

The ingestion pipeline has 5 stages, all in `src/lok_sabha_rag/pipeline/`:

1. **Curate metadata** — fetch question index from sansad.in API
2. **Download PDFs** — download Q&A PDF documents
3. **Extract text** — PDF to text (Docling + PyMuPDF fallback)
4. **Chunk** — tokenizer-aware splitting (BGE tokenizer, 500 token max)
5. **Embed & index** — embed with FastEmbed, upsert to Qdrant

See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for details.

## Documentation

The `docs/` directory contains a complete project knowledge base:

| File | Contents |
|------|----------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Tech stack, directory tree, data flow, M/N/C/Q strategy |
| [BACKEND.md](docs/BACKEND.md) | API routes, core modules, schemas, function signatures |
| [FRONTEND.md](docs/FRONTEND.md) | JS functions, CSS variables, UI behavior, feature matrix |
| [DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | Pipeline stages, data layout, SQLite + Qdrant schemas |
