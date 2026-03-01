# Lok Sabha RAG -- Project Overview

A Retrieval-Augmented Generation system for searching and answering questions about Indian Lok Sabha (parliamentary) Q&A documents. Users query the system with natural language; it retrieves relevant chunks from a Qdrant vector database and optionally synthesizes an answer with GPT-4o-mini, citing the source evidence.

## Architecture at a Glance

```
                          +----------------+
   User (browser)  --->   |   Frontend     |   index.html / app.js / styles.css
                          +-------+--------+
                                  |
                                  v
                          +-------+--------+
                          |   FastAPI       |   /api/search   (retrieval only)
                          |   (Uvicorn)     |   /api/synthesize (retrieval + LLM)
                          +--+-----------+--+
                             |           |
                    +--------+--+    +---+----------+
                    | Retriever |    | Synthesizer   |
                    | (Qdrant)  |    | (GPT-4o-mini) |
                    +-----------+    +--------------+
```

## Ingestion Pipeline

Five sequential stages turn raw parliamentary data into searchable vectors:

1. **Metadata curation** (`curate_ls_metadata.py`) -- Calls the sansad.in API to fetch Lok Sabha member lists, ministry names, and per-session question indices. Outputs a JSONL index file per session (`data/<lok>/index_session_<n>.jsonl`) containing question number, type, date, ministry, MP names, subject, and PDF URL.

2. **PDF download** (`download_pdfs.py`) -- Downloads question-answer PDFs from URLs in the index files. Atomic writes (`.part` then rename), polite crawling with random sleep, idempotent (skips existing). Stores files at `data/<lok>/pdfs/session_<n>/`.

3. **Text extraction** (`extract_text.py`) -- Parses PDFs with Docling for structure-aware markdown output. Falls back to PyMuPDF for plain text when Docling fails. Outputs JSON per document at `data/<lok>/parsed/session_<n>/`.

4. **Chunking** (`build_chunks_bge.py`) -- Splits parsed text into chunks using the `BAAI/bge-small-en-v1.5` tokenizer (max 500 tokens). Each chunk is prepended with a structured metadata header (Lok Sabha, session, question number, ministry, MPs, subject). Outputs JSONL at `data/<lok>/chunks/session_<n>/chunks.jsonl`.

5. **Embedding and indexing** (`embed_index_qdrant2.py`) -- Reads all `chunks.jsonl` files, embeds them with BGE-small via Qdrant's FastEmbed, and upserts into the `lok_sabha_questions` collection (cosine distance, 384-dim vectors). Batch size 512, 4 parallel workers.

### Current data scope

Only **Lok Sabha 18, sessions 6 and 7** have been fully ingested (parsed, chunked, and indexed).

## Query Pipeline

Two API endpoints, both under `/api`:

### `POST /api/search` -- retrieval only
- Takes: `query`, `top_k` (1--50), optional filters (`lok`, `session`, `ministry`).
- Runs semantic search against Qdrant using `query_points()` with the same BGE-small model.
- Reconstructs live PDF URLs from the session index files.
- Returns ranked evidence items with metadata and full chunk text.

### `POST /api/synthesize` -- retrieval + LLM
- Same retrieval step as above.
- Builds a context string from evidence chunks (formatted as `[E1] header | text`).
- Sends context + query to GPT-4o-mini with a system prompt enforcing citation-first, evidence-only answering.
- Extracts `[E#]` citations from the response and validates them.
- Returns the synthesized answer, list of citations used, and the full evidence set.

### Prompts

- **System prompt** (`prompts/system.txt`): Instructs the LLM to use only provided evidence, cite every claim as `[E#]`, never hallucinate, and describe conflicts between sources.
- **User prompt** (`prompts/user.txt`): Template injecting `{query}` and `{context}`, asking for a cited answer or clarifying questions if the query is ambiguous.

## Frontend

Single-page application served by FastAPI at `/`:

- **Search bar** with example placeholder queries.
- **AI Mode toggle** -- switches between `/api/synthesize` (LLM answer + evidence) and `/api/search` (evidence only).
- **Chunk count selector** (5--50, default 15).
- **Results display** -- When AI mode is on, shows the synthesized answer with clickable `[E#]` citations that scroll to and highlight the corresponding evidence card. When AI mode is off, shows evidence cards directly.
- **Evidence cards** -- Grouped by parliamentary question. Each card shows subject, ministry, MP names, match percentage, and expandable full text. Links to the original PDF on sansad.in.
- **Detail modal** -- Click a card to see full text and all metadata.

## Tech Stack

| Component       | Technology                              |
|-----------------|-----------------------------------------|
| Language        | Python 3.11+                            |
| API framework   | FastAPI + Uvicorn                       |
| Vector DB       | Qdrant (Docker, localhost:6333)         |
| Embedding model | BAAI/bge-small-en-v1.5 (384-dim)       |
| LLM             | OpenAI GPT-4o-mini                      |
| PDF parsing     | Docling (primary), PyMuPDF (fallback)   |
| CLI tooling     | Typer                                   |
| Frontend        | Vanilla HTML/CSS/JS                     |
| Package manager | uv + hatchling                          |

## Project Structure

```
lok-sabha-rag/
  main.py                          # Entry point (uvicorn server)
  pyproject.toml                   # Dependencies and build config
  src/lok_sabha_rag/
    api/
      main.py                     # FastAPI app, CORS, static mount
      schemas.py                  # Pydantic request/response models
      routes/
        search.py                 # /api/search endpoint
        synthesize.py             # /api/synthesize endpoint
    core/
      retriever.py                # Qdrant search, filtering, URL reconstruction
      synthesizer.py              # Prompt loading, OpenAI call, citation extraction
    prompts/
      system.txt                  # LLM system prompt
      user.txt                    # LLM user prompt template
    curate_ls_metadata.py         # Stage 1: metadata from sansad.in API
    download_pdfs.py              # Stage 2: PDF download
    extract_text.py               # Stage 3: Docling/PyMuPDF parsing
    build_chunks_bge.py           # Stage 4: tokenizer-aware chunking
    embed_index_qdrant2.py        # Stage 5: embed + index to Qdrant
  frontend/
    index.html                    # SPA shell
    app.js                        # Client logic
    styles.css                    # Styling
  data/
    18/
      index_session_*.jsonl       # Question metadata indices
      pdfs/session_*/             # Downloaded PDFs
      parsed/session_*/           # Parsed JSON outputs
      chunks/session_*/chunks.jsonl  # Chunked JSONL
```
