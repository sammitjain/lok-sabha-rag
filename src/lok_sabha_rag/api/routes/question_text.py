"""Question text endpoint — fetch absolute leading C chunks for a specific question."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lok_sabha_rag.core.retriever import Retriever

router = APIRouter()

# Shared retriever instance (reuses same Qdrant connection + URL cache as other routes)
_retriever = Retriever()


class QuestionTextRequest(BaseModel):
    question_id: str | None = Field(None, description="Stable composite key (preferred)")
    lok_no: int | None = None
    session_no: int | None = None
    ques_no: int | None = None
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

    Accepts either question_id (preferred) or the legacy lok_no/session_no/ques_no fields.
    """
    chunks = _retriever._fetch_leading_chunks(
        c=req.c,
        question_id=req.question_id,
        lok_no=req.lok_no,
        session_no=req.session_no,
        ques_no=req.ques_no,
        qtype=req.type,
    )

    if not chunks:
        detail = f"No chunks found for question_id={req.question_id}" if req.question_id else (
            f"No chunks found for Lok {req.lok_no} Session {req.session_no} Q{req.ques_no}"
        )
        raise HTTPException(status_code=404, detail=detail)

    # Join chunk texts — body only (no header to strip since chunks are pure body now)
    parts = [chunk.text or "" for chunk in chunks]

    return QuestionTextResponse(text="\n\n".join(parts))
