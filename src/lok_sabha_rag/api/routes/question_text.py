"""Question text endpoint — fetch absolute leading C chunks for a specific question."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lok_sabha_rag.core.retriever import Retriever

router = APIRouter()

# Shared retriever instance (reuses same Qdrant connection + URL cache as other routes)
_retriever = Retriever()


class QuestionTextRequest(BaseModel):
    lok_no: int
    session_no: int
    ques_no: int
    type: str | None = Field(None, description="'Starred' or 'Unstarred' — disambiguates overlapping ques_no ranges")
    c: int = Field(1, ge=1, le=10, description="Number of leading chunks to fetch (chunk_index 0..c-1)")


class QuestionTextResponse(BaseModel):
    text: str


@router.post("/question-text", response_model=QuestionTextResponse)
def question_text(req: QuestionTextRequest) -> QuestionTextResponse:
    """Fetch the absolute first C chunks (chunk_index 0..c-1) for a specific question.

    Uses Qdrant metadata scroll — not vector search — so the returned chunks are
    always the true document-order leading chunks, regardless of what was retrieved
    during the original search.

    Header (metadata prefix prepended by the chunker) is shown once (from the first
    chunk); subsequent chunks have their header stripped so body text flows cleanly.
    """
    chunks = _retriever._fetch_leading_chunks(
        req.lok_no, req.session_no, req.ques_no, req.c, qtype=req.type,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for Lok {req.lok_no} Session {req.session_no} Q{req.ques_no}",
        )

    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        text = chunk.text or ""
        if i == 0:
            # First chunk: keep the metadata header intact
            parts.append(text)
        else:
            # Subsequent chunks: strip the header (everything up to first blank line)
            sep = text.find("\n\n")
            parts.append(text[sep + 2:] if sep != -1 else text)

    return QuestionTextResponse(text="\n\n".join(parts))
