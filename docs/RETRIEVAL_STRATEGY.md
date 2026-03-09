# Retrieval & Synthesis Strategy

> Living document tracking how chunks are built, retrieved, grouped, and presented to the LLM.

## Data model

Each **row** in the HuggingFace dataset is one parliamentary Q&A (one PDF).
Each Q&A is split into **chunks** (max 500 tokens, bge-small-en-v1.5 tokenizer).
Every chunk carries structured **metadata** as Qdrant payload fields:

| Field | Example | Use |
|-------|---------|-----|
| `question_id` | `"17_1_42_Starred"` | Stable composite key for grouping + scroll |
| `lok_no` | `17` | Filter |
| `session_no` | `1` | Filter |
| `ques_no` | `42` | Grouping (legacy 4-tuple) |
| `type` | `"Starred"` | Grouping (legacy 4-tuple) |
| `ministry` | `"HEALTH AND FAMILY WELFARE"` | Filter |
| `mp_names` | `["Shri X", "Smt Y"]` | Filter (OR match) |
| `subject` | `"Status of AIIMS"` | Display only |
| `chunk_index` | `0` | C-chunk fetch ordering |

---

## Pipeline overview

```
[HF Dataset] ──> build_chunks ──> chunks.jsonl ──> embed ──> Qdrant collection
                                                                    │
User query ─────────────────────────────────────────────> retriever.search()
                                                                    │
                                                          top-M raw chunks
                                                                    │
                                                     retriever.group_evidence()
                                                                    │
                                                   ┌────────────────┴──────────────┐
                                                   │  Group by question_id          │
                                                   │  Sort groups by best_score     │
                                                   │  Trim to top-N groups          │
                                                   │  Fetch leading C chunks/group  │
                                                   └────────────────┬──────────────┘
                                                                    │
                                                   build_context_grouped()
                                                                    │
                                                        [Q#] labeled context
                                                                    │
                                               ┌────────────────────┤
                                               │ (optional)         │
                                          MP stats text        evidence text
                                               │                    │
                                               └────────┬───────────┘
                                                        │
                                              system_prompt + user_prompt
                                                        │
                                                   LLM (gpt-4o-mini)
                                                        │
                                                  synthesized answer
```

### Parameters (tunable per request)

| Param | Default | Meaning |
|-------|---------|---------|
| **M** (`top_k`) | 15 | Total chunks retrieved from Qdrant via vector search |
| **N** (`top_n`) | 10 | Max unique questions kept after grouping |
| **C** (`chunks_per_question`) | 2 | Leading chunks fetched per question (via scroll, not vector search) |
| **Q** (`top_q`) | 10 | Recent questions shown in MP stats summary |

---

## Chunking strategy

Chunk text is **body only** — pure answer content, no metadata header.

The old approach prepended a structured header (Lok/Session/Q#/Ministry/MPs/Subject) to every chunk's text before embedding. This was removed because:
- All those fields already exist as structured metadata in the Qdrant payload
- The header inflated cosine similarity for metadata keyword matches (e.g. "health" in "Ministry: HEALTH") rather than actual semantic content matches
- `build_context_grouped()` reconstructs a clean header from payload fields for the LLM anyway
- The header ate into the 500-token budget on every chunk

Now: full token budget goes to actual content. Metadata is only in payload fields.

The chunker (`build_chunks.py`) is model-agnostic — accepts `--model` to specify which HF model tokenizer to use (defaults to `EMBEDDING_MODEL` from config).

---

## Grouping & C-chunk fetch

Groups by `question_id` — a stable composite key: `f"{lok_no}_{session_no}_{ques_no}_{type}"`.

Scroll and count filters use a single `FieldCondition` on `question_id` (keyword-indexed in Qdrant) instead of the old 4-field composite filter. Falls back to 4-tuple if `question_id` is missing (legacy data).

---

## Context building for LLM

`build_context_grouped()` constructs the context string with `[Q#]` citation labels.
Each group gets a metadata header line reconstructed from payload fields (not from chunk text).
Chunk text is appended verbatim after the header.

This means the LLM sees structured metadata context regardless of what's in the embedded text — so removing the header from the embedding doesn't affect what the LLM receives.

---

## Filters

| Filter | Qdrant field | How applied |
|--------|-------------|-------------|
| Lok Sabha | `lok_no` | `MatchValue` in `_build_filter()` |
| Session | `session_no` | `MatchValue` |
| Ministry | `ministry` | `MatchValue` |
| MP name(s) | `mp_names` | `MatchAny` (OR logic) |

Future: multiple ministry filters, staggered retrieval across filter combinations.

---

## Testing strategy

Changes require a **full re-index** (different embedding input = incompatible vectors).

To avoid touching the production collection:
1. Build chunks with `--max-files 50 --data-dir data/test --overwrite`
2. Embed into a test collection: `--collection lok_sabha_test --data-dir data/test --overwrite`
3. Point the app at the test collection: `QDRANT_COLLECTION=lok_sabha_test`
4. Use the `/api/debug/trace` endpoint to compare retrieval quality

The `QDRANT_COLLECTION` env var (in `config.py`) controls which collection the entire app reads from — retriever, search, synthesize, and debug endpoints all use it.

---

## Change log

| Date | Change |
|------|--------|
| 2026-03-08 | Initial doc. Implemented: drop header from embedded text, add `question_id`, model-agnostic chunker, `self.collection` on retriever. |
