"""Retrieve evidence chunks from Qdrant."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

from qdrant_client import QdrantClient, models

from lok_sabha_rag.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    EMBEDDING_MODEL,
)


@dataclass(frozen=True)
class EvidenceItem:
    score: float
    chunk_id: str
    question_id: Optional[str]
    lok_no: Optional[int]
    session_no: Optional[int]
    ques_no: Optional[int]
    type: Optional[str]  # "Starred" | "Unstarred"
    asked_by: Optional[str]
    ministry: Optional[str]
    subject: Optional[str]
    pdf_filename: Optional[str]
    pdf_url: Optional[str]
    chunk_index: Optional[int]
    text: str


@dataclass
class EvidenceGroup:
    """A group of chunks from the same parliamentary question."""
    group_index: int
    question_id: Optional[str]
    lok_no: Optional[int]
    session_no: Optional[int]
    ques_no: Optional[int]
    type: Optional[str]  # "Starred" | "Unstarred"
    ministry: Optional[str]
    subject: Optional[str]
    asked_by: Optional[str]
    pdf_url: Optional[str]
    best_score: float
    chunks: List[EvidenceItem] = field(default_factory=list)
    total_chunks_available: int = 0   # how many chunks existed before trimming


def _build_filter(
    lok: Optional[int] = None,
    session: Optional[int] = None,
    ministry: Optional[str] = None,
    mp_names: Optional[List[str]] = None,
) -> Optional[models.Filter]:
    must = []
    if lok is not None:
        must.append(models.FieldCondition(key="lok_no", match=models.MatchValue(value=lok)))
    if session is not None:
        must.append(models.FieldCondition(key="session_no", match=models.MatchValue(value=session)))
    if ministry:
        must.append(models.FieldCondition(key="ministry", match=models.MatchValue(value=ministry)))
    if mp_names:
        # OR filter: match chunks where mp_names contains ANY of the provided names
        must.append(models.FieldCondition(
            key="mp_names",
            match=models.MatchAny(any=mp_names),
        ))
    return models.Filter(must=must) if must else None


def _safe_str(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _extract_evidence(p, *, score: float = 0.0) -> EvidenceItem:
    """Convert a Qdrant point to an EvidenceItem."""
    pay = p.payload or {}
    src = pay.get("source", {}) or {}

    return EvidenceItem(
        score=score,
        chunk_id=str(pay.get("chunk_id") or p.id),
        question_id=_safe_str(pay.get("question_id")),
        lok_no=int(pay["lok_no"]) if pay.get("lok_no") is not None else None,
        session_no=int(pay["session_no"]) if pay.get("session_no") is not None else None,
        ques_no=pay.get("ques_no"),
        type=_safe_str(pay.get("type")),
        asked_by=_safe_str(pay.get("mp_names")),
        ministry=_safe_str(pay.get("ministry")),
        subject=_safe_str(pay.get("subject")),
        pdf_filename=_safe_str(src.get("pdf_filename")),
        pdf_url=_safe_str(src.get("pdf_url")),
        chunk_index=int(src["chunk_index"]) if src.get("chunk_index") is not None else None,
        text=pay.get("text", ""),
    )


class Retriever:
    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection: str = QDRANT_COLLECTION,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.model = EMBEDDING_MODEL
        self.collection = collection

    def search(
        self,
        query: str,
        top_k: int = 10,
        lok: Optional[int] = None,
        session: Optional[int] = None,
        ministry: Optional[str] = None,
        mp_names: Optional[List[str]] = None,
    ) -> List[EvidenceItem]:
        query_filter = _build_filter(lok=lok, session=session, ministry=ministry, mp_names=mp_names)

        points = self.client.query_points(
            collection_name=self.collection,
            query=models.Document(text=query, model=self.model),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            search_params=models.SearchParams(hnsw_ef=256),
        ).points

        return [_extract_evidence(p, score=float(p.score)) for p in points]

    def build_context(self, items: List[EvidenceItem]) -> str:
        parts = ["EVIDENCE (verbatim chunks; cite as [E#]):"]
        for i, ev in enumerate(items, start=1):
            header = []
            if ev.lok_no is not None and ev.session_no is not None:
                header.append(f"Lok {ev.lok_no}, Session {ev.session_no}")
            if ev.ques_no is not None:
                header.append(f"Q{ev.ques_no}")
            if ev.ministry:
                header.append(f"Ministry: {ev.ministry}")
            if ev.asked_by:
                header.append(f"Asked by: {ev.asked_by}")
            if ev.subject:
                header.append(f"Subject: {ev.subject}")
            header.append(f"Score: {ev.score:.4f}")

            parts.append(f"\n[E{i}] " + " | ".join(header))
            parts.append(ev.text.strip() if ev.text else "")

        return "\n".join(parts).strip()

    def _fetch_leading_chunks(
        self,
        c: int,
        question_id: Optional[str] = None,
        lok_no: Optional[int] = None,
        session_no: Optional[int] = None,
        ques_no: Optional[int] = None,
        qtype: Optional[str] = None,
    ) -> List[EvidenceItem]:
        """Fetch the first C chunks (by chunk_index) for a specific question
        directly from Qdrant using metadata scroll, regardless of vector search.

        Prefers single-field question_id filter; falls back to 4-field composite.
        """
        if question_id:
            must = [
                models.FieldCondition(key="question_id", match=models.MatchValue(value=question_id)),
                models.FieldCondition(
                    key="source.chunk_index",
                    range=models.Range(gte=0, lt=c),
                ),
            ]
        else:
            must = [
                models.FieldCondition(key="lok_no", match=models.MatchValue(value=lok_no)),
                models.FieldCondition(key="session_no", match=models.MatchValue(value=session_no)),
                models.FieldCondition(key="ques_no", match=models.MatchValue(value=ques_no)),
                models.FieldCondition(
                    key="source.chunk_index",
                    range=models.Range(gte=0, lt=c),
                ),
            ]
            if qtype:
                must.append(models.FieldCondition(key="type", match=models.MatchValue(value=qtype)))

        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(must=must),
            limit=c,
            with_payload=True,
        )

        items = [_extract_evidence(p) for p in points]
        items.sort(key=lambda c: c.chunk_index or 0)
        return items

    def _count_total_chunks(
        self,
        question_id: Optional[str] = None,
        lok_no: Optional[int] = None,
        session_no: Optional[int] = None,
        ques_no: Optional[int] = None,
        qtype: Optional[str] = None,
    ) -> int:
        """Count total chunks for a question in Qdrant."""
        if question_id:
            must = [
                models.FieldCondition(key="question_id", match=models.MatchValue(value=question_id)),
            ]
        else:
            must = [
                models.FieldCondition(key="lok_no", match=models.MatchValue(value=lok_no)),
                models.FieldCondition(key="session_no", match=models.MatchValue(value=session_no)),
                models.FieldCondition(key="ques_no", match=models.MatchValue(value=ques_no)),
            ]
            if qtype:
                must.append(models.FieldCondition(key="type", match=models.MatchValue(value=qtype)))

        result = self.client.count(
            collection_name=self.collection,
            count_filter=models.Filter(must=must),
            exact=True,
        )
        return result.count

    def group_evidence(
        self,
        items: List[EvidenceItem],
        top_n: Optional[int] = None,
        chunks_per_question: Optional[int] = None,
    ) -> List[EvidenceGroup]:
        """Group flat evidence items by question_id (or 4-tuple fallback).

        Args:
            items: Flat list of evidence chunks from search.
            top_n: Keep only the top N questions (ranked by best chunk score).
                   None means keep all.
            chunks_per_question: For each question, fetch the first C chunks
                   (chunk_index 0..C-1) directly from Qdrant. None means use
                   whatever chunks were retrieved by vector search.
        """
        groups: OrderedDict[str, List[EvidenceItem]] = OrderedDict()
        for item in items:
            key = item.question_id or f"{item.lok_no}_{item.session_no}_{item.ques_no}_{item.type}"
            groups.setdefault(key, []).append(item)

        result: List[EvidenceGroup] = []
        for idx, (key, chunk_list) in enumerate(groups.items(), start=1):
            best = max(chunk_list, key=lambda c: c.score)

            result.append(EvidenceGroup(
                group_index=idx,
                question_id=best.question_id,
                lok_no=best.lok_no,
                session_no=best.session_no,
                ques_no=best.ques_no,
                type=best.type,
                ministry=best.ministry,
                subject=best.subject,
                asked_by=best.asked_by,
                pdf_url=best.pdf_url,
                best_score=best.score,
                chunks=sorted(chunk_list, key=lambda c: c.chunk_index or 0),
                total_chunks_available=0,  # filled below if C is set
            ))

        # Sort by best_score descending, then trim to top N questions
        result.sort(key=lambda g: g.best_score, reverse=True)
        if top_n is not None:
            result = result[:top_n]

        # For each kept question, fetch the guaranteed first C chunks from Qdrant
        if chunks_per_question is not None:
            for g in result:
                total = self._count_total_chunks(
                    question_id=g.question_id,
                    lok_no=g.lok_no,
                    session_no=g.session_no,
                    ques_no=g.ques_no,
                    qtype=g.type,
                )
                g.total_chunks_available = total
                leading = self._fetch_leading_chunks(
                    c=chunks_per_question,
                    question_id=g.question_id,
                    lok_no=g.lok_no,
                    session_no=g.session_no,
                    ques_no=g.ques_no,
                    qtype=g.type,
                )
                if leading:
                    g.chunks = leading
                # If fetch failed, keep original chunks as fallback

        # Re-assign group_index after trimming so citations are 1..N
        for i, g in enumerate(result, start=1):
            g.group_index = i

        return result

    def build_context_grouped(self, groups: List[EvidenceGroup]) -> str:
        """Build LLM context string from grouped evidence. Uses [Q#] citation labels."""
        parts = ["EVIDENCE (grouped by parliamentary question; cite as [Q#]):"]

        for g in groups:
            header = []
            if g.lok_no is not None and g.session_no is not None:
                header.append(f"Lok {g.lok_no}, Session {g.session_no}")
            if g.ques_no is not None:
                header.append(f"Q{g.ques_no}")
            if g.ministry:
                header.append(f"Ministry: {g.ministry}")
            if g.asked_by:
                header.append(f"Asked by: {g.asked_by}")
            if g.subject:
                header.append(f"Subject: {g.subject}")

            parts.append(f"\n[Q{g.group_index}] " + " | ".join(header))

            for chunk in g.chunks:
                text = chunk.text.strip() if chunk.text else ""
                if text:
                    parts.append(text)

            # Note about trailing chunks that were trimmed
            trimmed = g.total_chunks_available - len(g.chunks)
            if trimmed > 0:
                parts.append(
                    f"\n[NOTE: This question has {trimmed} more chunk(s) with "
                    f"additional details (annexures, tables, etc.) not shown here. "
                    f"Refer to the source PDF for complete information.]"
                )

        return "\n".join(parts).strip()


def to_dict_list(items: List[EvidenceItem]) -> List[Dict[str, Any]]:
    return [asdict(e) for e in items]
