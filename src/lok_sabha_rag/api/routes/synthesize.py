"""Synthesize endpoint - LLM-powered answer generation."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from lok_sabha_rag.core.retriever import Retriever, EvidenceItem, EvidenceGroup
from lok_sabha_rag.core.stats import get_mp_stats, format_stats_for_llm

_LOG_CHUNKS = os.environ.get("RAG_LOG_CHUNKS", "").strip() == "1"
from lok_sabha_rag.core.synthesizer import Synthesizer, extract_citations
from lok_sabha_rag.api.schemas import (
    SynthesizeRequest,
    SynthesizeResponse,
    EvidenceGroupResponse,
    MpStatsResponse,
    ChunkDetail,
)

router = APIRouter()

retriever = Retriever()
synthesizer = Synthesizer()


def _truncate(text: str, max_len: int = 200) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _log_retrieval(
    query: str,
    items: list[EvidenceItem],
    groups: list[EvidenceGroup],
    req: SynthesizeRequest,
) -> None:
    """Pretty-print retrieved chunks and final grouped questions to terminal."""
    filter_str = f"  mp_names={req.mp_names}" if req.mp_names else ""
    print(f"\n{'━'*70}")
    print(f"  SYNTHESIZE | query={query!r}")
    print(f"  M={req.top_k}  N={req.top_n}  C={req.chunks_per_question}{filter_str}")
    print(f"{'━'*70}")

    # Log raw retrieved chunks
    seen: set[tuple] = set()
    for i, item in enumerate(items, 1):
        qkey = (item.lok_no, item.session_no, item.type, item.ques_no)
        new_q = qkey not in seen
        seen.add(qkey)
        marker = "★" if new_q else " "
        subject = (item.subject or "?")[:55]
        asked = item.asked_by or "?"
        print(f"  {marker} [{i:>2}] Q{item.ques_no or '?'} chunk={item.chunk_index}  score={item.score:.3f}  {subject}")
        if new_q:
            print(f"         Asked by: {asked}")
    print(f"{'─'*70}")
    print(f"  Retrieved {len(items)} chunks from {len(seen)} unique questions")

    # Log grouped + trimmed questions sent to LLM
    print(f"\n  → Grouped into {len(groups)} questions for LLM:")
    for g in groups:
        subject = (g.subject or "?")[:55]
        asked = g.asked_by or "?"
        trimmed = g.total_chunks_available - len(g.chunks)
        trail = f"  (+{trimmed} trailing)" if trimmed > 0 else ""
        print(f"    [Q{g.group_index}] Q{g.ques_no} | {len(g.chunks)} chunks{trail} | score={g.best_score:.3f}")
        print(f"         Subject: {subject}")
        print(f"         Asked by: {asked}")
    print(f"{'━'*70}\n")


@router.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest) -> SynthesizeResponse:
    items = retriever.search(
        query=req.query,
        top_k=req.top_k,
        lok=req.lok,
        session=req.session,
        ministry=req.ministry,
        mp_names=req.mp_names,
    )

    if not items:
        raise HTTPException(
            status_code=404,
            detail="No relevant evidence found. Try broadening your search or removing filters.",
        )

    # Fetch MP stats from metadata DB when a single MP filter is active
    mp_stats = None
    mp_stats_text = ""
    if req.mp_names and len(req.mp_names) == 1:
        mp_stats = get_mp_stats(req.mp_names[0], top_q=req.top_q or 10)
        if mp_stats:
            mp_stats_text = format_stats_for_llm(mp_stats)

    # Group by parliamentary question, trim to top N questions with C chunks each
    groups = retriever.group_evidence(
        items,
        top_n=req.top_n,
        chunks_per_question=req.chunks_per_question,
    )

    if _LOG_CHUNKS:
        _log_retrieval(req.query, items, groups, req)
        if mp_stats:
            print(f"  MP Stats: {mp_stats.mp_name} — {mp_stats.total_questions} total questions, showing {len(mp_stats.recent_questions)} recent")

    # Build grouped context for LLM (no scores, [Q#] labels)
    evidence_context = retriever.build_context_grouped(groups)

    # Compose full context: stats (if available) + evidence
    if mp_stats_text:
        context = mp_stats_text + "\n\n" + evidence_context
    else:
        context = evidence_context

    answer = synthesizer.generate(query=req.query, context=context)
    citations = extract_citations(answer, max_n=len(groups))

    evidence_groups = [
        EvidenceGroupResponse(
            group_index=g.group_index,
            question_id=g.question_id,
            lok_no=g.lok_no,
            session_no=g.session_no,
            ques_no=g.ques_no,
            type=g.type,
            ministry=g.ministry,
            subject=g.subject,
            asked_by=g.asked_by,
            pdf_url=g.pdf_url,
            best_score=g.best_score,
            total_chunks_available=g.total_chunks_available,
            chunks=[
                ChunkDetail(
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    text_preview=_truncate(chunk.text),
                )
                for chunk in g.chunks
            ],
        )
        for g in groups
    ]

    # Build stats response for frontend display
    mp_stats_response = None
    if mp_stats:
        mp_stats_response = MpStatsResponse(
            mp_name=mp_stats.mp_name,
            total_questions=mp_stats.total_questions,
            by_lok=mp_stats.by_lok,
            by_session=mp_stats.by_session,
            by_type=mp_stats.by_type,
            top_ministries=[
                {"ministry": m, "count": c} for m, c in mp_stats.by_ministry
            ],
            recent_questions=[
                {
                    "lok_no": q.lok_no,
                    "session_no": q.session_no,
                    "ques_no": q.ques_no,
                    "type": q.type,
                    "date": q.date,
                    "ministry": q.ministry,
                    "subject": q.subject,
                }
                for q in mp_stats.recent_questions
            ],
        )

    return SynthesizeResponse(
        query=req.query,
        answer=answer,
        citations_used=citations,
        evidence_groups=evidence_groups,
        total_chunks=len(items),
        mp_stats=mp_stats_response,
    )
