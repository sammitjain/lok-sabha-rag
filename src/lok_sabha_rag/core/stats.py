"""Query MP statistics from the SQLite metadata database."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from lok_sabha_rag.config import METADATA_DB_PATH


@dataclass
class QuestionRecord:
    lok_no: int
    session_no: int
    ques_no: int
    type: str
    date: Optional[str]
    ministry: str
    subject: str


@dataclass
class MpStats:
    mp_name: str
    total_questions: int
    by_lok: dict[int, int]
    by_session: dict[str, int]
    by_type: dict[str, int]
    by_ministry: List[Tuple[str, int]]  # sorted desc, top 15
    recent_questions: List[QuestionRecord]


def get_mp_stats(
    mp_name: str,
    top_q: int = 10,
    db_path: str | Path = METADATA_DB_PATH,
) -> Optional[MpStats]:
    """Query metadata DB for an MP's parliamentary question statistics.

    Returns None if the MP has no questions in the database.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            """
            SELECT q.lok_no, q.session_no, q.ques_no, q.type, q.date,
                   q.ministry, q.subject
            FROM questions q
            JOIN question_mps m USING (lok_no, session_no, type, ques_no)
            WHERE m.mp_name = ?
            ORDER BY q.date DESC, q.ques_no DESC
            """,
            (mp_name,),
        ).fetchall()

        if not rows:
            return None

        # Aggregations
        by_lok: dict[int, int] = {}
        by_session: dict[str, int] = {}
        by_ministry: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for r in rows:
            by_lok[r["lok_no"]] = by_lok.get(r["lok_no"], 0) + 1
            key = f"Lok {r['lok_no']} Session {r['session_no']}"
            by_session[key] = by_session.get(key, 0) + 1
            by_ministry[r["ministry"]] = by_ministry.get(r["ministry"], 0) + 1
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1

        # Top 15 ministries, sorted by count descending
        ministry_sorted = sorted(by_ministry.items(), key=lambda x: -x[1])[:15]

        # Top Q most recent questions
        recent = [
            QuestionRecord(
                lok_no=r["lok_no"],
                session_no=r["session_no"],
                ques_no=r["ques_no"],
                type=r["type"],
                date=r["date"],
                ministry=r["ministry"],
                subject=(r["subject"] or "").strip(),
            )
            for r in rows[:top_q]
        ]

        return MpStats(
            mp_name=mp_name,
            total_questions=len(rows),
            by_lok=dict(sorted(by_lok.items())),
            by_session=dict(sorted(by_session.items())),
            by_type=dict(sorted(by_type.items())),
            by_ministry=ministry_sorted,
            recent_questions=recent,
        )
    finally:
        conn.close()


def format_stats_for_llm(stats: MpStats) -> str:
    """Format MP stats into a delimited text block for inclusion in LLM context."""
    lines: list[str] = []

    lines.append("=== MP STATISTICS (from parliamentary metadata) ===")
    lines.append(f"MP Name: {stats.mp_name}")
    lines.append(f"Total questions asked: {stats.total_questions}")
    lines.append("")

    lines.append("By Lok Sabha:")
    for lok, count in stats.by_lok.items():
        lines.append(f"  Lok {lok}: {count}")
    lines.append("")

    lines.append("By Session:")
    for key, count in stats.by_session.items():
        lines.append(f"  {key}: {count}")
    lines.append("")

    lines.append("By Type:")
    for qtype, count in stats.by_type.items():
        lines.append(f"  {qtype}: {count}")
    lines.append("")

    lines.append("Top Ministries:")
    for ministry, count in stats.by_ministry:
        lines.append(f"  {count:4d}  {ministry}")
    lines.append("")

    if stats.recent_questions:
        lines.append(f"Recent Questions (most recent {len(stats.recent_questions)}):")
        for i, q in enumerate(stats.recent_questions, 1):
            lines.append(
                f"  {i}. [Lok {q.lok_no}, Session {q.session_no}, Q{q.ques_no}] "
                f"{q.ministry} — {q.subject} ({q.date or 'N/A'})"
            )
        lines.append("")

    lines.append(
        "NOTE: These statistics cover ALL questions asked by this MP, "
        "including those whose full text may not be in the evidence chunks below."
    )
    lines.append("=== END MP STATISTICS ===")

    return "\n".join(lines)
