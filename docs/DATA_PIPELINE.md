# Data Pipeline & Storage Reference

---

## Five-Stage Ingestion Pipeline

### Stage 1: Metadata Curation (`pipeline/curate_ls_metadata.py`)

```bash
uv run python -m lok_sabha_rag.pipeline.curate_ls_metadata --lok 18 --sessions 6-7
```

- **Input**: sansad.in API (`/api_ls/question/...`)
- **Output**:
  - `data/<lok>/members.json` — MP name list
  - `data/<lok>/ministries.json` — ministry list
  - `data/<lok>/loksabha_sessions.json` — session dates
  - `data/<lok>/index_session_<N>.jsonl` — per-session question index
  - `data/<lok>/progress.json` — curation status
- **Behavior**: Fetches question feed pages, extracts metadata. Idempotent (skips existing entries). Polite crawling with sleep.

### Stage 2: PDF Download (`pipeline/download_pdfs.py`)

```bash
uv run python -m lok_sabha_rag.pipeline.download_pdfs --lok 18 --sessions 6-7
```

- **Input**: `data/<lok>/index_session_<N>.jsonl` (reads `questionsFilePath` URLs)
- **Output**: `data/<lok>/pdfs/session_<N>/<filename>.pdf`
- **Behavior**: Atomic writes (`.part` → rename). Skips existing. Random sleep between downloads. Logs failures to `failed_downloads.txt`.

### Stage 3: Text Extraction (`pipeline/extract_text.py`)

```bash
uv run python -m lok_sabha_rag.pipeline.extract_text --lok 18 --sessions 6-7
```

- **Input**: `data/<lok>/pdfs/session_<N>/*.pdf`
- **Output**: `data/<lok>/parsed/session_<N>/<filename>.json`
- **Behavior**: Docling (primary) with PyMuPDF fallback. Idempotent + resumable. Output includes per-page text + `full_text` concatenation.

### Stage 4: Chunking (`pipeline/build_chunks_bge.py`)

```bash
uv run python -m lok_sabha_rag.pipeline.build_chunks_bge --lok 18 --sessions 6-7
```

- **Input**: `data/<lok>/parsed/session_<N>/*.json` + `data/<lok>/index_session_<N>.jsonl` (for metadata)
- **Output**: `data/<lok>/chunks/session_<N>/chunks.jsonl`
- **Behavior**: Tokenizer-aware chunking (BAAI/bge-small-en-v1.5 tokenizer, max 500 tokens). Prepends structured metadata header to each chunk. UUID5 chunk IDs for determinism.

### Stage 5: Embed & Index (`pipeline/embed_index_qdrant2.py`)

```bash
uv run python -m lok_sabha_rag.pipeline.embed_index_qdrant2
```

- **Input**: `data/**/chunks.jsonl` (discovers recursively)
- **Output**: Qdrant collection `lok_sabha_questions`
- **Behavior**: Embeds with FastEmbed (BAAI/bge-small-en-v1.5). Batch 512, 4 workers. Upserts (idempotent). Creates collection if missing (cosine distance).

### Metadata DB Build (`pipeline/build_metadata_db.py`)

```bash
uv run python -m lok_sabha_rag.pipeline.build_metadata_db
uv run python -m lok_sabha_rag.pipeline.build_metadata_db --db-path data/metadata.db
```

- **Input**: `data/*/index_session_*.jsonl`
- **Output**: `data/metadata.db` (SQLite)
- **Behavior**: Scans all index JSONL across all Lok Sabha numbers. INSERT OR REPLACE for idempotency. Converts `DD.MM.YYYY` → `YYYY-MM-DD` dates.

---

## Data Directory Layout

```
data/
├── metadata.db                          # SQLite (86,273 questions, 143,471 MP rows)
├── snapshots/                           # Qdrant snapshots (for starter dataset)
├── sample/                              # Starter dataset (100 questions, committed to git)
│   └── 18/
│       ├── chunks/session_7/chunks.jsonl
│       └── index_session_7.jsonl
├── 17/                                  # Lok Sabha 17 (~67MB, metadata only)
│   ├── members.json                     # [{"mpName": "...", ...}, ...]
│   ├── ministries.json                  # [{"ministryName": "..."}, ...]
│   ├── index_session_1.jsonl            # sessions 1-15 (skip 13)
│   ├── index_session_2.jsonl
│   ├── ...
│   └── index_session_15.jsonl
└── 18/                                  # Lok Sabha 18 (~6.3GB full pipeline)
    ├── members.json
    ├── ministries.json
    ├── loksabha_sessions.json
    ├── index_session_2.jsonl            # sessions 2-7
    ├── ...
    ├── index_session_7.jsonl
    ├── pdfs/                            # NOT in git (5.8 GB)
    ├── parsed/                          # NOT in git (289 MB)
    ├── chunks/                          # NOT in git (390 MB)
    └── failed_downloads.txt             # NOT in git
```

**Git-tracked data** (~110 MB): `metadata.db`, `sample/`, all `index_session_*.jsonl`, `members.json`, `ministries.json`, `loksabha_sessions.json`.
**Git-ignored**: `pdfs/`, `parsed/`, `chunks/`, `progress.json`, `failed_downloads.txt`.

---

## Index JSONL Record Schema

Each line in `data/<lok>/index_session_<N>.jsonl`:

```json
{
  "lok_no": 18,
  "session_no": 7,
  "ques_no": 1975,
  "type": "Unstarred",
  "date": "04.02.2025",
  "subjects": "Welfare Measures for Senior Citizens",
  "members": ["Shri Rahul Gandhi", "Smt. Priyanka Gandhi Vadra"],
  "ministry": "Ministry of Social Justice and Empowerment",
  "questionsFilePath": "https://sansad.in/getFile/...pdf?source=pqals"
}
```

Key fields: `lok_no`, `session_no`, `type`, `ques_no` (composite key — `type` is required because Starred/Unstarred ques_no ranges overlap within the same session), `date` (DD.MM.YYYY format), `members` (array of MP names), `questionsFilePath` (PDF download URL).

---

## Chunk JSONL Record Schema

Each line in `data/<lok>/chunks/session_<N>/chunks.jsonl`:

```json
{
  "chunk_id": "uuid5-deterministic-hash",
  "text": "QUESTION: ...\nAsked by: Shri X\nMinistry: ...\n\n[chunk text content]",
  "lok_no": 18,
  "session_no": 7,
  "ques_no": 1975,
  "type": "Unstarred",
  "date": "04.02.2025",
  "ministry": "Ministry of Social Justice and Empowerment",
  "mp_names": ["Shri Rahul Gandhi", "Smt. Priyanka Gandhi Vadra"],
  "subject": "Welfare Measures for Senior Citizens",
  "source": {
    "pdf_filename": "UnsQA_18_07_1975_abc123.pdf",
    "pdf_relpath": "session_7/UnsQA_18_07_1975_abc123.pdf",
    "chunk_index": 0
  },
  "pipeline": {
    "parser": "docling",
    "chunker": "bge-small-en-v1.5",
    "max_tokens": 500
  }
}
```

The `text` field includes a metadata header prepended by the chunker (question info, first 3 MP names).

---

## SQLite Schema (`data/metadata.db`)

```sql
CREATE TABLE questions (
    lok_no       INTEGER,
    session_no   INTEGER,
    ques_no      INTEGER,
    type         TEXT,           -- "STARRED" | "UNSTARRED"
    date         TEXT,           -- YYYY-MM-DD (converted from DD.MM.YYYY)
    ministry     TEXT,
    subject      TEXT,
    pdf_filename TEXT,
    PRIMARY KEY (lok_no, session_no, type, ques_no)
    -- NOTE: type is in the PK because starred/unstarred ques_no ranges overlap
    --       within the same session (both start at 1).
);

CREATE TABLE question_mps (
    lok_no     INTEGER,
    session_no INTEGER,
    ques_no    INTEGER,
    type       TEXT,             -- must match questions.type for correct join
    mp_name    TEXT              -- one row per MP per question
);

CREATE INDEX idx_qmp_question ON question_mps (lok_no, session_no, type, ques_no);
CREATE INDEX idx_qmp_mp ON question_mps (mp_name);
```

**Row counts:** 86,273 questions, 143,471 MP rows (Lok 17 + 18 combined).
Previously only 79,561 questions because starred/unstarred ques_no ranges overlap (both start at 1 per session), so starred rows were silently overwritten by INSERT OR REPLACE.

**Stats query pattern** (used by `core/stats.py`):
```sql
SELECT q.lok_no, q.session_no, q.ques_no, q.type, q.date, q.ministry, q.subject
FROM questions q
JOIN question_mps m USING (lok_no, session_no, type, ques_no)
WHERE m.mp_name = ?
ORDER BY q.date DESC, q.ques_no DESC
```

---

## Qdrant Collection Schema

**Collection:** `lok_sabha_questions`
**Vector:** 384 dimensions (BAAI/bge-small-en-v1.5), cosine distance

**Payload fields** (per point):

```
chunk_id: str         # UUID5 hash
text: str             # chunk text (with metadata header)
lok_no: int
session_no: int
ques_no: int
type: str             # "Starred" | "Unstarred"
date: str             # YYYY-MM-DD
ministry: str
mp_names: list[str]   # array — enables MatchAny filter
subject: str
source.pdf_filename: str
source.pdf_relpath: str
source.chunk_index: int     # 0-based position within the question
```

**Filter examples:**
- MP filter: `FieldCondition(key="mp_names", match=MatchAny(any=["Shri X"]))`
- Leading chunks: `FieldCondition(key="source.chunk_index", range=Range(gte=0, lt=C))`
- Type disambiguation: `FieldCondition(key="type", match=MatchValue(value="Starred"))` — used by `_fetch_leading_chunks` and `_count_total_chunks` to disambiguate overlapping ques_no ranges

---

## Data Relationships

```
index_session_*.jsonl
    ├── questionsFilePath → download_pdfs → pdfs/session_N/file.pdf
    ├── (all fields) → build_metadata_db → metadata.db (questions + question_mps)
    └── (metadata) → build_chunks_bge → chunks.jsonl (header prepended)
                                            └── embed_index_qdrant2 → Qdrant collection

At query time:
    Qdrant search → EvidenceItem (with source.pdf_filename)
    _load_url_index() reads index JSONL → maps pdf_filename → live_url
    metadata.db → MP statistics (get_mp_stats)
```
