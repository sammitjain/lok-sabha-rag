# Backend Reference

All Python source lives under `src/lok_sabha_rag/`.

---

## API Layer (`api/`)

### `api/main.py` — FastAPI App Setup

- `app = FastAPI(title="Lok Sabha RAG", version="0.1.0")`
- CORS: allow all origins/methods/headers
- Routes registered with `/api` prefix:
  - `search.router` (tags: search)
  - `synthesize.router` (tags: synthesize)
  - `members.router` (tags: members)
  - `stats.router` (tags: stats)
  - `question_text.router` (tags: question-text)
- `GET /` → serves `frontend/index.html`
- Static mount: `/static` → `frontend/` directory
- `FRONTEND_DIR = Path(__file__).parent.parent.parent.parent / "frontend"`

### `api/schemas.py` — Pydantic Models

**Request Models:**

```
SearchRequest:
    query: str (required, min_length=1)
    top_k: int = 10 (1-200)
    lok: Optional[int]
    session: Optional[int]
    ministry: Optional[str]
    mp_names: Optional[List[str]]    # OR logic in Qdrant filter

SynthesizeRequest:
    query: str (required, min_length=1)
    top_k: int = 15 (1-200)          # M
    top_n: Optional[int] = 10 (1-50) # N
    chunks_per_question: Optional[int] = 2 (1-10)  # C
    top_q: Optional[int] = 10 (1-50) # Q
    lok, session, ministry, mp_names  # same as SearchRequest
```

**Response Models:**

```
EvidenceItemResponse:
    index, score, chunk_id, lok_no, session_no, ques_no,
    type: Optional[str],   # "Starred" | "Unstarred"
    asked_by, ministry, subject, pdf_relpath, pdf_filename,
    chunk_index, live_url, text, text_preview

SearchResponse:
    query, results: List[EvidenceItemResponse], total

ChunkDetail:
    chunk_index, score, chunk_id, text, text_preview

EvidenceGroupResponse:
    group_index, lok_no, session_no, ques_no,
    type: Optional[str],   # "Starred" | "Unstarred"
    ministry, subject, asked_by, live_url, best_score,
    chunks: List[ChunkDetail], total_chunks_available: int = 0

MpStatsResponse:
    mp_name, total_questions, by_lok: dict, by_session: dict,
    by_type: dict, top_ministries: List[dict], recent_questions: List[dict]

SynthesizeResponse:
    query, answer, citations_used: List[int],
    evidence_groups: List[EvidenceGroupResponse],
    total_chunks, mp_stats: Optional[MpStatsResponse]
```

### `api/routes/search.py` — POST /api/search

**Global:** `retriever = Retriever()` (shared Qdrant client)

**Functions:**
- `_truncate(text: str, max_len: int = 200) -> str` — truncate with ellipsis
- `_log_retrieved_chunks(query, items, mp_names) -> None` — terminal debug output (if `RAG_LOG_CHUNKS=1`)
- `search(req: SearchRequest) -> SearchResponse` — calls `retriever.search()`, maps to `EvidenceItemResponse` list

**Flow:** query → Qdrant vector search (M chunks) → flat ranked list → return as-is

### `api/routes/synthesize.py` — POST /api/synthesize

**Globals:** `retriever = Retriever()`, `synthesizer = Synthesizer()`

**Functions:**
- `_truncate(text, max_len=200) -> str`
- `_log_retrieval(query, items, groups, req) -> None` — logs raw chunks + grouped questions
- `synthesize(req: SynthesizeRequest) -> SynthesizeResponse` — main endpoint

**Flow:**
1. `retriever.search(query, top_k=M, filters...)` → flat items
2. If single MP filter: `get_mp_stats(mp_name, top_q=Q)` → MpStats
3. `retriever.group_evidence(items, top_n=N, chunks_per_question=C)` → groups
4. `retriever.build_context_grouped(groups)` → evidence text with `[Q#]` labels
5. If stats exist: `context = stats_text + "\n\n" + evidence_text`
6. `synthesizer.generate(query, context)` → LLM answer
7. `extract_citations(answer, max_n=len(groups))` → citation indices
8. Build `EvidenceGroupResponse` list + `MpStatsResponse`
9. Return `SynthesizeResponse`

### `api/routes/members.py` — GET /api/members/{lok}

**Global:** `_cache: Dict[int, List[str]] = {}` (in-memory)

- `_load_members(lok: int) -> List[str]` — reads `data/<lok>/members.json`, extracts `mpName` field, sorts case-insensitive, caches
- `get_members(lok: int) -> List[str]` — returns cached MP names

### `api/routes/stats.py` — POST /api/mp-stats

**Local model:** `MpStatsRequest(mp_name: str, top_q: int = 10)`

- `mp_stats(req: MpStatsRequest) -> MpStatsResponse` — calls `get_mp_stats()`, returns 404 if MP not found, otherwise builds `MpStatsResponse` with ministry tuples → `[{"ministry": m, "count": c}]`

This endpoint exists separately from synthesize so the frontend can fire it **in parallel** (stats is fast SQLite; synthesize is slow LLM).

### `api/routes/question_text.py` — POST /api/question-text

**Global:** `_retriever = Retriever()`

**Local models:**
- `QuestionTextRequest(lok_no: int, session_no: int, ques_no: int, type: str|None, c: int = 1)`
- `QuestionTextResponse(text: str)`

**Function:**
- `question_text(req: QuestionTextRequest) -> QuestionTextResponse` — calls `_retriever._fetch_leading_chunks(lok_no, session_no, ques_no, c, qtype=type)`, strips header from chunks after the first, joins with `\n\n`. Returns 404 if no chunks found.

This endpoint is called **lazily** by the frontend when a user expands a card or opens the modal, to fetch the absolute leading C chunks (chunk_index 0..C-1) rather than showing whatever was returned by vector search.

---

## Core Modules (`core/`)

### `config.py` — Centralized Configuration

All settings read from env vars with sensible defaults. See `.env.example`.

```python
QDRANT_HOST: str       # default "localhost"
QDRANT_PORT: int       # default 6333
QDRANT_COLLECTION: str # default "lok_sabha_questions"
EMBEDDING_MODEL: str   # default "BAAI/bge-small-en-v1.5"
DATA_DIR: Path         # default "data"
METADATA_DB_PATH: Path # default "data/metadata.db"
```

### `core/retriever.py` — Qdrant Search & Context Building

**Imports from `config`:** `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`, `EMBEDDING_MODEL`, `DATA_DIR`

**Dataclasses:**

```python
@dataclass(frozen=True)
class EvidenceItem:
    score: float
    chunk_id: str
    lok_no: Optional[int]
    session_no: Optional[int]
    ques_no: Optional[int]
    type: Optional[str]           # "Starred" | "Unstarred" — part of unique question identity
    asked_by: Optional[str]       # raw mp_names from Qdrant payload
    ministry: Optional[str]
    subject: Optional[str]
    pdf_relpath: Optional[str]
    pdf_filename: Optional[str]
    chunk_index: Optional[int]
    live_url: Optional[str]
    text: str

@dataclass
class EvidenceGroup:
    group_index: int
    lok_no, session_no, ques_no: Optional[int]
    type: Optional[str]           # "Starred" | "Unstarred"
    ministry, subject, asked_by, live_url: Optional[str]
    best_score: float
    chunks: List[EvidenceItem] = field(default_factory=list)
    total_chunks_available: int = 0
```

**Module-level functions:**
- `_build_filter(lok, session, ministry, mp_names) -> Optional[models.Filter]` — builds Qdrant filter. mp_names uses `MatchAny` (OR logic)
- `_safe_str(x) -> Optional[str]` — None if empty/whitespace
- `_load_url_index(base_dir, lok, session) -> Dict[str, str]` — loads `index_session_*.jsonl`, maps pdf_filename → live URL
- `to_dict_list(items) -> List[Dict]` — convert to dicts via `asdict()`

**Class `Retriever`:**

```python
Retriever(host="localhost", port=6333, data_dir="data")
```

Methods:
- `_get_live_url(lok, session, pdf_filename) -> Optional[str]` — cached URL index lookup
- `search(query, top_k=10, lok=None, session=None, ministry=None, mp_names=None) -> List[EvidenceItem]` — `client.query_points()` with Document query + optional filter
- `build_context(items) -> str` — flat `[E#]` labeled context (used by search mode, currently unused in synthesis)
- `_fetch_leading_chunks(lok_no, session_no, ques_no, c, qtype=None) -> List[EvidenceItem]` — scroll Qdrant for chunk_index 0..C-1, sorted by chunk_index. Filters by `type` when `qtype` is provided. Score = 0.0 (not from vector search)
- `_count_total_chunks(lok_no, session_no, ques_no, qtype=None) -> int` — exact count via `client.count()`. Filters by `type` when `qtype` is provided
- `group_evidence(items, top_n=None, chunks_per_question=None) -> List[EvidenceGroup]` — group by **(lok, session, type, ques)**, sort by best_score desc, trim to N, optionally replace chunks with leading C from Qdrant (passes `qtype=g.type`), re-index 1..N
- `build_context_grouped(groups) -> str` — `[Q#] header | header\nchunk_text` format with trimming notes

### `core/synthesizer.py` — LLM Synthesis

**Constants:**
- `PROMPTS_DIR = Path(__file__).parent.parent / "prompts"`
- `_CITATION_RE = re.compile(r"\[Q(\d+)\]")`

**Functions:**
- `load_prompt(name: str) -> str` — reads from `prompts/` dir
- `get_system_prompt() -> str` — loads `system.txt`
- `get_user_prompt(query, context) -> str` — loads `user.txt`, formats `{query}` and `{context}`
- `extract_citations(answer: str, max_n: int) -> List[int]` — finds all `[Q#]` in answer, validates 1..max_n, returns sorted
- `validate_citations(answer: str, evidence_count: int) -> List[str]` — returns error strings for out-of-range citations

**Class `Synthesizer`:**

```python
Synthesizer()  # requires OPENAI_API_KEY env var
    .model = "gpt-4o-mini"
    .client = OpenAI()
```

Methods:
- `generate(query: str, context: str) -> str` — calls `client.responses.create()` with system + user prompts, parses `output_text` or falls back to `output[].content[].text`

### `core/stats.py` — MP Statistics from SQLite

**Imports from `config`:** `METADATA_DB_PATH`

**Dataclasses:**

```python
@dataclass
class QuestionRecord:
    lok_no: int, session_no: int, ques_no: int,
    type: str, date: Optional[str], ministry: str, subject: str

@dataclass
class MpStats:
    mp_name: str
    total_questions: int
    by_lok: dict[int, int]
    by_session: dict[str, int]      # keys like "Lok 18 Session 6"
    by_type: dict[str, int]         # "Starred", "Unstarred"
    by_ministry: List[Tuple[str, int]]  # sorted desc, top 15
    recent_questions: List[QuestionRecord]
```

**Functions:**
- `get_mp_stats(mp_name: str, top_q: int = 10, db_path = DEFAULT_DB_PATH) -> Optional[MpStats]` — single SQL query joining `questions` + `question_mps`, aggregates in Python, returns None if no rows
- `format_stats_for_llm(stats: MpStats) -> str` — produces `=== MP STATISTICS ===` delimited text block with all breakdowns + recent questions list + NOTE about coverage

---

## Prompts (`prompts/`)

### `system.txt` — LLM System Prompt

Key rules:
1. Use ONLY provided evidence, no outside knowledge
2. Stay on Lok Sabha Q&A topics
3. Never invent names/numbers/dates/schemes
4. Cite every claim as `[Q1]`, `[Q2][Q3]` etc.
5. If sources conflict, cite both
6. Begin with confidence note
7. Structure: Summary → Detailed Findings → Limitations
8. Truncated questions: point to "View PDF" for annexures/tables
9. MP Statistics section: reference freely without `[Q#]`, suggest searching for relevant uncovered questions, start with high-level stats summary when useful (rule 6, user-added)

### `user.txt` — LLM User Prompt Template

```
USER QUESTION:
{query}

CONTEXT:
{context}

TASK:
Answer using ONLY CONTEXT. Cite as [Q#]. Follow structured format.
If ambiguous, ask 1-3 clarifying questions in Limitations.
```

---

## Pipeline Scripts (`pipeline/`)

All ingestion scripts live in `src/lok_sabha_rag/pipeline/` and are invoked via `python -m lok_sabha_rag.pipeline.<name>`.

| Script | Purpose | Command |
|--------|---------|---------|
| `curate_ls_metadata.py` | Stage 1: fetch metadata from sansad.in | `uv run python -m lok_sabha_rag.pipeline.curate_ls_metadata --lok 18` |
| `download_pdfs.py` | Stage 2: download Q&A PDFs | `uv run python -m lok_sabha_rag.pipeline.download_pdfs --lok 18` |
| `extract_text.py` | Stage 3: PDF → text | `uv run python -m lok_sabha_rag.pipeline.extract_text --lok 18` |
| `build_chunks_bge.py` | Stage 4: tokenizer-aware chunking | `uv run python -m lok_sabha_rag.pipeline.build_chunks_bge --lok 18` |
| `embed_index_qdrant2.py` | Stage 5: embed + upsert to Qdrant | `uv run python -m lok_sabha_rag.pipeline.embed_index_qdrant2` |
| `build_metadata_db.py` | Build SQLite from index JSONL | `uv run python -m lok_sabha_rag.pipeline.build_metadata_db` |

## Dev Scripts (`dev/` — gitignored)

Local-only exploration and debugging tools in `src/lok_sabha_rag/dev/`. Not committed to git.

| Script | Purpose |
|--------|---------|
| `query_stats.py` | Query MP stats from CLI |
| `explore_chunks.py` | Inspect chunk content |
| `explore_qdrant.py` | Inspect Qdrant collection |
| `answer_query.py` | Standalone query answering |
| `retrieve_chunks.py` | Direct chunk retrieval |
| `retrieve_results.py` | Results retrieval |
| `synthesize_answer.py` | Standalone synthesis (imports `build_evidence`) |
| `build_evidence.py` | Evidence data builder |
| `collect_questions.py` | Question collection from parsed PDFs |
| `audit_unmatched_ministries.py` | Ministry name auditing |
| `build_chunks_old.py` | Legacy chunker (superseded) |
| `trial_docling.py` | Docling parser testing |

---

## Environment Variables

All config vars are defined in `config.py` and read from the environment. See `.env.example`.

| Var | Required | Default | Description |
|-----|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes (for synthesis) | — | OpenAI API key, loaded via `python-dotenv` from `.env` |
| `QDRANT_HOST` | No | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | No | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | No | `lok_sabha_questions` | Qdrant collection name |
| `EMBEDDING_MODEL` | No | `BAAI/bge-small-en-v1.5` | Embedding model (384-dim) |
| `DATA_DIR` | No | `data` | Root data directory |
| `METADATA_DB_PATH` | No | `data/metadata.db` | SQLite metadata database path |
| `RAG_LOG_CHUNKS` | No | — | Set to `1` to enable terminal debug output of retrieved chunks |
