# Architecture

## What This Is

A Retrieval-Augmented Generation (RAG) system for Indian Lok Sabha parliamentary Q&A records. Users search questions via semantic vector search (Qdrant) and optionally get LLM-synthesized answers (GPT-4o-mini) with `[Q#]` citations back to source documents. A SQLite metadata DB provides MP activity statistics.

## Tech Stack

| Tool | Version | Role |
|------|---------|------|
| Python | >=3.11 | Runtime |
| FastAPI | >=0.129 | API server |
| Qdrant | (external) | Vector DB, localhost:6333 |
| BAAI/bge-small-en-v1.5 | 384-dim | Embedding model (via FastEmbed) |
| GPT-4o-mini | (OpenAI API) | LLM synthesis |
| SQLite | (stdlib) | Metadata DB (`data/metadata.db`) |
| Typer | >=0.21 | CLI tools |
| Docling + PyMuPDF | | PDF text extraction |
| marked.js | CDN | Frontend markdown rendering |
| Vanilla JS/CSS | | Frontend (no framework) |

## Directory Tree

```
lok-sabha-rag/
├── main.py                          # Entry point: uvicorn on port 8000
├── pyproject.toml                   # Dependencies (hatchling build)
├── docker-compose.yml               # Qdrant service
├── .env.example                     # Template for OPENAI_API_KEY
├── scripts/
│   ├── setup.sh                     # One-command setup (deps, Qdrant, snapshot)
│   └── create_sample_data.py        # Generate starter dataset from full data
├── src/lok_sabha_rag/
│   ├── api/
│   │   ├── main.py                  # FastAPI app, CORS, static mount, route registration
│   │   ├── schemas.py               # All Pydantic request/response models
│   │   └── routes/
│   │       ├── search.py            # POST /api/search
│   │       ├── synthesize.py        # POST /api/synthesize
│   │       ├── members.py           # GET  /api/members/{lok}
│   │       ├── stats.py             # POST /api/mp-stats
│   │       └── question_text.py     # POST /api/question-text (lazy leading chunk fetch)
│   ├── core/
│   │   ├── retriever.py             # Qdrant search, evidence grouping, context building
│   │   ├── synthesizer.py           # OpenAI calls, prompt loading, citation extraction
│   │   └── stats.py                 # SQLite MP stats queries + LLM formatter
│   ├── prompts/                     # LLM prompt templates
│   │   ├── system.txt               # System prompt (citation rules, MP stats usage)
│   │   └── user.txt                 # User prompt template ({query}, {context})
│   ├── pipeline/                    # Data ingestion (5-stage + metadata DB)
│   │   ├── curate_ls_metadata.py    # Stage 1: fetch metadata from sansad.in API
│   │   ├── download_pdfs.py         # Stage 2: download Q&A PDFs
│   │   ├── extract_text.py          # Stage 3: PDF → text (Docling/PyMuPDF)
│   │   ├── build_chunks_bge.py      # Stage 4: tokenizer-aware chunking (500 tok max)
│   │   ├── embed_index_qdrant2.py   # Stage 5: embed + upsert to Qdrant
│   │   └── build_metadata_db.py     # Build SQLite from index JSONL files
│   └── dev/                         # Local-only exploration scripts (gitignored)
├── frontend/
│   ├── index.html                   # SPA shell (cache busters: ?v=16)
│   ├── app.js                       # Client logic (~870 lines)
│   └── styles.css                   # Styles (~1100 lines)
├── data/
│   ├── metadata.db                  # SQLite: 86,273 questions, 143,471 MP rows
│   ├── sample/                      # Starter dataset (100 questions from session 7)
│   ├── snapshots/                   # Qdrant snapshots
│   ├── 17/                          # Lok Sabha 17 (metadata only, 15 sessions)
│   └── 18/                          # Lok Sabha 18 (full pipeline, sessions 6-7 indexed)
└── docs/                            # Project knowledge base
```

## Data Flow

```
User query
    │
    ├──→ Frontend (app.js)
    │       ├── POST /api/synthesize      ←── (slow: Qdrant search + LLM)
    │       ├── POST /api/mp-stats        ←── (fast: SQLite, fires in parallel)
    │       └── POST /api/question-text   ←── (lazy: Qdrant scroll on card expand)
    │
    ├──→ FastAPI (api/main.py)
    │       ├── routes/synthesize.py
    │       │     ├── Retriever.search()        → Qdrant query_points (M chunks)
    │       │     ├── Retriever.group_evidence() → top N questions, C leading chunks each
    │       │     ├── get_mp_stats()             → SQLite (if single MP filter)
    │       │     ├── build_context_grouped()    → "[Q#] header\nchunk text" format
    │       │     ├── Synthesizer.generate()     → OpenAI GPT-4o-mini
    │       │     └── extract_citations()        → regex [Q#] from answer
    │       └── routes/stats.py
    │             └── get_mp_stats()             → SQLite aggregation
    │
    └──→ Response
            ├── answer (markdown with [Q#] citations)
            ├── evidence_groups (grouped chunks for source cards)
            └── mp_stats (optional: totals, breakdowns, recent questions)
```

## M/N/C/Q Retrieval Strategy

| Param | Name | Default | Range | Description |
|-------|------|---------|-------|-------------|
| M | `top_k` | 50 | 5-200 | Total chunks retrieved from Qdrant vector search |
| N | `top_n` | 10 | 1-50 | Max unique questions kept (ranked by best chunk score) |
| C | `chunks_per_question` | 1 | 1-10 | Leading chunks fetched per question (chunk_index 0..C-1, direct Qdrant scroll) |
| Q | `top_q` | 15 | 1-50 | Recent questions in MP stats summary |

Flow: Search M chunks → group by question → sort by best score → keep top N → for each, fetch first C chunks from Qdrant → build `[Q#]` labeled context for LLM.

## How to Run

```bash
# Quick start (requires Docker + uv)
./scripts/setup.sh             # installs deps, starts Qdrant, loads sample data
uv run python main.py          # → http://localhost:8000

# Manual setup
cp .env.example .env           # add your OPENAI_API_KEY
docker compose up -d            # start Qdrant on localhost:6333
uv sync                        # install Python dependencies
uv run python main.py          # start server

# Pipeline commands (for full data ingestion)
uv run python -m lok_sabha_rag.pipeline.curate_ls_metadata --lok 18 --sessions 6-7
uv run python -m lok_sabha_rag.pipeline.download_pdfs --lok 18 --sessions 6-7
uv run python -m lok_sabha_rag.pipeline.extract_text --lok 18 --sessions 6-7
uv run python -m lok_sabha_rag.pipeline.build_chunks_bge --lok 18 --sessions 6-7
uv run python -m lok_sabha_rag.pipeline.embed_index_qdrant2
uv run python -m lok_sabha_rag.pipeline.build_metadata_db

# Debug logging
RAG_LOG_CHUNKS=1 uv run python main.py
```

## Current Scope

- **Indexed in Qdrant**: Lok Sabha 18, sessions 6-7
- **Metadata DB**: Lok Sabha 17 (all sessions) + Lok Sabha 18 (sessions 2-7) = 86,273 questions
- **Question identity**: `(lok_no, session_no, type, ques_no)` — `type` (Starred/Unstarred) is part of the composite key because ques_no ranges overlap within a session
- **MP autocomplete**: Hardcoded to Lok Sabha 18 (`LOK_NO = 18` in app.js)
- **MP stats**: Triggered when MP filter is active (regardless of AI mode)

## Deferred / Future

- ~~Config centralization~~ → done (`config.py`, env-var driven)
- Ingestion pipeline automation
- Multi-MP stats support
- Streaming/SSE for LLM responses
- Indexing more Lok Sabha sessions
