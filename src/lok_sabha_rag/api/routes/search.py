"""Search endpoint - non-LLM retrieval."""

import os

from fastapi import APIRouter

from lok_sabha_rag.core.retriever import Retriever, EvidenceItem
from lok_sabha_rag.api.schemas import SearchRequest, SearchResponse, EvidenceItemResponse

_LOG_CHUNKS = os.environ.get("RAG_LOG_CHUNKS", "").strip() == "1"

router = APIRouter()

retriever = Retriever()


def _truncate(text: str, max_len: int = 200) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _log_retrieved_chunks(query: str, items: list[EvidenceItem], mp_names: list[str] | None = None) -> None:
    """Pretty-print retrieved chunks to terminal."""
    filter_str = f"  mp_names={mp_names}" if mp_names else ""
    print(f"\n{'─'*70}")
    print(f"  SEARCH | query={query!r}  top_k={len(items)}{filter_str}")
    print(f"{'─'*70}")
    seen_questions: set[tuple] = set()
    for i, item in enumerate(items, 1):
        qkey = (item.lok_no, item.session_no, item.type, item.ques_no)
        new_q = qkey not in seen_questions
        seen_questions.add(qkey)
        marker = "★" if new_q else " "
        asked = item.asked_by or "?"
        subject = (item.subject or "?")[:60]
        print(f"  {marker} [{i:>2}] Q{item.ques_no or '?'} chunk={item.chunk_index}  score={item.score:.3f}")
        print(f"         Subject: {subject}")
        print(f"         Asked by: {asked}")
    print(f"{'─'*70}")
    print(f"  {len(items)} chunks from {len(seen_questions)} unique questions\n")


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    items = retriever.search(
        query=req.query,
        top_k=req.top_k,
        lok=req.lok,
        session=req.session,
        ministry=req.ministry,
        mp_names=req.mp_names,
    )

    if _LOG_CHUNKS:
        _log_retrieved_chunks(req.query, items, req.mp_names)

    results = [
        EvidenceItemResponse(
            index=i,
            score=item.score,
            chunk_id=item.chunk_id,
            lok_no=item.lok_no,
            session_no=item.session_no,
            ques_no=item.ques_no,
            type=item.type,
            asked_by=item.asked_by,
            ministry=item.ministry,
            subject=item.subject,
            pdf_filename=item.pdf_filename,
            pdf_url=item.pdf_url,
            chunk_index=item.chunk_index,
            text=item.text,
            text_preview=_truncate(item.text),
        )
        for i, item in enumerate(items, start=1)
    ]

    return SearchResponse(
        query=req.query,
        results=results,
        total=len(results),
    )

