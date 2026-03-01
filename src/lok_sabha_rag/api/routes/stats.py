"""MP statistics endpoint — fast SQLite lookup for parallel fetching."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lok_sabha_rag.core.stats import get_mp_stats
from lok_sabha_rag.api.schemas import MpStatsResponse

router = APIRouter()


class MpStatsRequest(BaseModel):
    mp_name: str = Field(..., min_length=1)
    top_q: int = Field(10, ge=1, le=50)


@router.post("/mp-stats", response_model=MpStatsResponse)
def mp_stats(req: MpStatsRequest) -> MpStatsResponse:
    stats = get_mp_stats(req.mp_name, top_q=req.top_q)
    if not stats:
        raise HTTPException(status_code=404, detail=f"No questions found for '{req.mp_name}'")

    return MpStatsResponse(
        mp_name=stats.mp_name,
        total_questions=stats.total_questions,
        by_lok=stats.by_lok,
        by_session=stats.by_session,
        by_type=stats.by_type,
        top_ministries=[
            {"ministry": m, "count": c} for m, c in stats.by_ministry
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
            for q in stats.recent_questions
        ],
    )
