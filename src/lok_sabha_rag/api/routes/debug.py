"""Debug trace endpoint — replays the synthesis pipeline without calling the LLM."""

from __future__ import annotations

import html as html_mod
import json
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from lok_sabha_rag.core.retriever import Retriever, EvidenceItem, EvidenceGroup
from lok_sabha_rag.core.stats import get_mp_stats, format_stats_for_llm
from lok_sabha_rag.core.synthesizer import get_system_prompt, get_user_prompt
from lok_sabha_rag.api.schemas import TraceChunk, TraceGroup

router = APIRouter()

retriever = Retriever()


# ── helpers ────────────────────────────────────────────────────────────


def _item_to_trace_chunk(item: EvidenceItem) -> TraceChunk:
    return TraceChunk(
        chunk_id=item.chunk_id,
        question_id=item.question_id,
        chunk_index=item.chunk_index,
        score=item.score,
        lok_no=item.lok_no,
        session_no=item.session_no,
        ques_no=item.ques_no,
        type=item.type,
        ministry=item.ministry,
        asked_by=item.asked_by,
        subject=item.subject,
        pdf_url=item.pdf_url,
        text=item.text,
    )


def _group_key(g: EvidenceGroup) -> str:
    return g.question_id or f"({g.lok_no}, {g.session_no}, {g.type}, {g.ques_no})"


def _group_to_trace(g: EvidenceGroup) -> TraceGroup:
    return TraceGroup(
        group_key=_group_key(g),
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
        chunk_count=len(g.chunks),
        total_chunks_available=g.total_chunks_available,
        chunks=[_item_to_trace_chunk(c) for c in g.chunks],
    )


def _run_trace(
    q: str,
    top_k: int,
    top_n: int,
    chunks_per_question: int,
    top_q: int,
    lok: Optional[int],
    session: Optional[int],
    ministry: Optional[str],
    mp: Optional[List[str]],
) -> dict:
    """Execute the full pipeline and return all stages as a plain dict."""

    trace_input = {
        "query": q,
        "filters": {
            "lok": lok,
            "session": session,
            "ministry": ministry,
            "mp_names": mp,
        },
        "params": {
            "top_k_M": top_k,
            "top_n_N": top_n,
            "chunks_per_question_C": chunks_per_question,
            "top_q_Q": top_q,
        },
    }

    items = retriever.search(
        query=q, top_k=top_k, lok=lok, session=session,
        ministry=ministry, mp_names=mp,
    )

    if not items:
        raise HTTPException(
            status_code=404,
            detail="No relevant evidence found. Try broadening your search or removing filters.",
        )

    trace_vector_search = {
        "total_chunks": len(items),
        "chunks": [_item_to_trace_chunk(it).model_dump() for it in items],
    }

    raw_groups = retriever.group_evidence(items, top_n=None, chunks_per_question=None)
    trace_grouping = {
        "total_groups": len(raw_groups),
        "groups": [_group_to_trace(g).model_dump() for g in raw_groups],
    }

    final_groups = retriever.group_evidence(
        items, top_n=top_n, chunks_per_question=chunks_per_question,
    )
    trace_c_chunk_fetch = {
        "total_groups": len(final_groups),
        "groups": [_group_to_trace(g).model_dump() for g in final_groups],
    }

    mp_stats_dict = None
    mp_stats_text = ""
    if mp and len(mp) == 1:
        stats = get_mp_stats(mp[0], top_q=top_q)
        if stats:
            mp_stats_text = format_stats_for_llm(stats)
            mp_stats_dict = {
                "mp_name": stats.mp_name,
                "total_questions": stats.total_questions,
                "by_lok": stats.by_lok,
                "by_session": stats.by_session,
                "by_type": stats.by_type,
                "top_ministries": [
                    {"ministry": m, "count": c} for m, c in stats.by_ministry
                ],
                "recent_questions_count": len(stats.recent_questions),
                "formatted_text": mp_stats_text,
            }

    evidence_context = retriever.build_context_grouped(final_groups)
    full_context = (mp_stats_text + "\n\n" + evidence_context) if mp_stats_text else evidence_context

    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(query=q, context=full_context)

    return {
        "input": trace_input,
        "vector_search": trace_vector_search,
        "grouping": trace_grouping,
        "c_chunk_fetch": trace_c_chunk_fetch,
        "mp_stats": mp_stats_dict,
        "evidence_context": full_context,
        "prompt": {"system_prompt": system_prompt, "user_prompt": user_prompt},
    }


# ── HTML renderer ──────────────────────────────────────────────────────

_E = html_mod.escape


def _render_chunk_table(chunks: list[dict], show_score: bool = True) -> str:
    """Render a list of chunk dicts as an HTML table."""
    rows = []
    for c in chunks:
        score_td = f'<td class="num">{c["score"]:.4f}</td>' if show_score else ""
        subj = _E(str(c.get("subject") or "—"))
        ministry = _E(str(c.get("ministry") or "—"))
        asked = _E(str(c.get("asked_by") or "—"))
        text = _E(c.get("text", "")[:500])
        if len(c.get("text", "")) > 500:
            text += "…"
        pdf = c.get("pdf_url") or ""
        pdf_link = f'<a href="{_E(pdf)}" target="_blank">PDF</a>' if pdf else "—"

        rows.append(f"""<tr>
            <td class="num">Q{c.get('ques_no', '?')}</td>
            <td class="num">{c.get('chunk_index', '?')}</td>
            {score_td}
            <td>{subj}</td>
            <td>{ministry}</td>
            <td>{asked}</td>
            <td>{pdf_link}</td>
            <td class="chunk-text">{text}</td>
        </tr>""")

    score_th = '<th>Score</th>' if show_score else ""
    return f"""<table>
        <thead><tr>
            <th>Ques</th><th>Chunk#</th>{score_th}
            <th>Subject</th><th>Ministry</th><th>Asked by</th><th>PDF</th><th>Text (first 500 chars)</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>"""


def _render_group_cards(groups: list[dict], title_prefix: str = "Q") -> str:
    """Render evidence groups as collapsible cards."""
    cards = []
    for g in groups:
        key = _E(g.get("group_key", ""))
        subj = _E(str(g.get("subject") or "—"))
        ministry = _E(str(g.get("ministry") or "—"))
        asked = _E(str(g.get("asked_by") or "—"))
        score = g.get("best_score", 0)
        n_chunks = g.get("chunk_count", len(g.get("chunks", [])))
        total = g.get("total_chunks_available", 0)
        total_str = f" / {total} total in DB" if total else ""

        chunk_table = _render_chunk_table(g.get("chunks", []))

        cards.append(f"""<details class="group-card">
            <summary>
                <strong>{key}</strong> &mdash; {subj}
                <span class="badge">score {score:.4f}</span>
                <span class="badge">{n_chunks} chunks{total_str}</span>
            </summary>
            <div class="group-meta">
                Ministry: {ministry} &bull; Asked by: {asked}
            </div>
            {chunk_table}
        </details>""")

    return "\n".join(cards)


def _render_html(data: dict, q: str, params: dict) -> str:
    """Build the full HTML page from trace data."""
    inp = data["input"]
    vs = data["vector_search"]
    grp = data["grouping"]
    cfetch = data["c_chunk_fetch"]
    mp = data.get("mp_stats")
    ctx = data["evidence_context"]
    prompt = data["prompt"]

    filters = inp["filters"]
    active_filters = {k: v for k, v in filters.items() if v is not None}
    filter_str = ", ".join(f"{k}={v}" for k, v in active_filters.items()) if active_filters else "none"

    params_d = inp["params"]

    # MP stats section
    mp_section = ""
    if mp:
        mp_section = f"""<section>
            <h2>5. MP Stats</h2>
            <p><strong>{_E(mp.get('mp_name', ''))}</strong> &mdash;
               {mp.get('total_questions', 0)} total questions</p>
            <pre class="context-block">{_E(mp.get('formatted_text', ''))}</pre>
        </section>"""
    else:
        mp_section = """<section>
            <h2>5. MP Stats</h2>
            <p class="muted">No single-MP filter active — skipped.</p>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trace: {_E(q)}</title>
<style>
    :root {{
        --bg: #0d1117; --surface: #161b22; --border: #30363d;
        --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
        --green: #3fb950; --orange: #d29922; --red: #f85149;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        background: var(--bg); color: var(--text);
        line-height: 1.5; padding: 1.5rem; max-width: 1400px; margin: 0 auto;
    }}
    h1 {{ font-size: 1.4rem; margin-bottom: 0.3rem; }}
    h2 {{
        font-size: 1.1rem; color: var(--accent); margin: 1.5rem 0 0.5rem;
        border-bottom: 1px solid var(--border); padding-bottom: 0.3rem;
    }}
    .query-bar {{
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 1rem;
    }}
    .query-bar code {{ color: var(--accent); font-size: 1.05rem; }}
    .params {{ color: var(--muted); font-size: 0.85rem; margin-top: 0.3rem; }}
    .params span {{ margin-right: 1rem; }}
    .badge {{
        display: inline-block; font-size: 0.75rem; padding: 0.1rem 0.5rem;
        border-radius: 10px; background: var(--surface); border: 1px solid var(--border);
        color: var(--muted); margin-left: 0.4rem;
    }}
    section {{ margin-bottom: 1rem; }}
    table {{
        width: 100%; border-collapse: collapse; font-size: 0.82rem;
        margin-top: 0.4rem;
    }}
    th {{
        background: var(--surface); color: var(--muted); text-align: left;
        padding: 0.4rem 0.5rem; border-bottom: 2px solid var(--border);
        position: sticky; top: 0;
    }}
    td {{
        padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--border);
        vertical-align: top;
    }}
    tr:hover td {{ background: rgba(88,166,255,0.04); }}
    td.num {{ font-family: 'SF Mono', Consolas, monospace; text-align: right; white-space: nowrap; }}
    td.chunk-text {{
        max-width: 450px; font-size: 0.78rem; color: var(--muted);
        word-break: break-word;
    }}
    details {{ margin: 0.3rem 0; }}
    details summary {{
        cursor: pointer; padding: 0.4rem 0.6rem; border-radius: 4px;
        background: var(--surface); border: 1px solid var(--border);
        font-size: 0.88rem; list-style: none;
    }}
    details summary::-webkit-details-marker {{ display: none; }}
    details summary::before {{ content: "▸ "; color: var(--muted); }}
    details[open] summary::before {{ content: "▾ "; }}
    details[open] summary {{ border-radius: 4px 4px 0 0; border-bottom: none; }}
    .group-card > :not(summary) {{ padding: 0.5rem 0.6rem; }}
    .group-meta {{ font-size: 0.82rem; color: var(--muted); margin-bottom: 0.3rem; }}
    .context-block {{
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 4px; padding: 0.8rem; font-size: 0.8rem;
        font-family: 'SF Mono', Consolas, monospace;
        white-space: pre-wrap; word-break: break-word;
        max-height: 500px; overflow-y: auto; color: var(--muted);
    }}
    .muted {{ color: var(--muted); font-style: italic; }}
    .stage-summary {{
        font-size: 0.9rem; color: var(--muted); margin-bottom: 0.4rem;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    /* search form */
    .search-form {{
        display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: end;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 6px; padding: 0.8rem 1rem; margin-bottom: 1.5rem;
    }}
    .search-form label {{ font-size: 0.78rem; color: var(--muted); display: block; }}
    .search-form input, .search-form select {{
        background: var(--bg); border: 1px solid var(--border); color: var(--text);
        border-radius: 4px; padding: 0.35rem 0.5rem; font-size: 0.85rem;
    }}
    .search-form input[type="text"] {{ width: 280px; }}
    .search-form input[type="number"] {{ width: 55px; }}
    .search-form button {{
        background: var(--accent); color: #000; border: none; border-radius: 4px;
        padding: 0.4rem 1rem; font-size: 0.85rem; cursor: pointer; font-weight: 600;
    }}
    .search-form button:hover {{ opacity: 0.9; }}
    .field {{ display: flex; flex-direction: column; gap: 0.15rem; }}
</style>
</head>
<body>

<h1>Pipeline Trace</h1>

<form class="search-form" method="get" action="/api/debug/trace">
    <div class="field">
        <label>Query</label>
        <input type="text" name="q" value="{_E(q)}" required>
    </div>
    <div class="field">
        <label>M (top_k)</label>
        <input type="number" name="M" value="{params_d['top_k_M']}" min="1" max="200">
    </div>
    <div class="field">
        <label>N (top_n)</label>
        <input type="number" name="N" value="{params_d['top_n_N']}" min="1" max="50">
    </div>
    <div class="field">
        <label>C (chunks/q)</label>
        <input type="number" name="C" value="{params_d['chunks_per_question_C']}" min="1" max="10">
    </div>
    <div class="field">
        <label>MP filter</label>
        <input type="text" name="mp" value="{_E((inp['filters'].get('mp_names') or [''])[0] if inp['filters'].get('mp_names') else '')}" placeholder="e.g. Shri Rahul Gandhi" style="width:180px">
    </div>
    <div class="field">
        <label>Ministry</label>
        <input type="text" name="ministry" value="{_E(inp['filters'].get('ministry') or '')}" placeholder="optional" style="width:160px">
    </div>
    <div class="field">
        <label>&nbsp;</label>
        <button type="submit">Trace</button>
    </div>
</form>

<div class="query-bar">
    <code>{_E(q)}</code>
    <div class="params">
        <span>M={params_d['top_k_M']}</span>
        <span>N={params_d['top_n_N']}</span>
        <span>C={params_d['chunks_per_question_C']}</span>
        <span>Filters: {_E(filter_str)}</span>
    </div>
</div>

<section>
    <h2>2. Vector Search <span class="badge">{vs['total_chunks']} chunks</span></h2>
    <p class="stage-summary">Raw top-M chunks from Qdrant, ranked by cosine similarity.</p>
    <details>
        <summary>Show {vs['total_chunks']} chunks</summary>
        {_render_chunk_table(vs['chunks'])}
    </details>
</section>

<section>
    <h2>3. Grouping (before trim) <span class="badge">{grp['total_groups']} groups</span></h2>
    <p class="stage-summary">Natural clustering by (lok, session, type, ques_no) — all groups, no C-fetch.</p>
    {_render_group_cards(grp['groups'])}
</section>

<section>
    <h2>4. C-Chunk Fetch (final) <span class="badge">{cfetch['total_groups']} groups</span></h2>
    <p class="stage-summary">Trimmed to top-N groups, leading C chunks fetched via Qdrant scroll.</p>
    {_render_group_cards(cfetch['groups'])}
</section>

{mp_section}

<section>
    <h2>6. Evidence Context</h2>
    <p class="stage-summary">The full context string sent to the LLM.</p>
    <details>
        <summary>Show context ({len(ctx)} chars)</summary>
        <pre class="context-block">{_E(ctx)}</pre>
    </details>
</section>

<section>
    <h2>7. Prompt</h2>
    <details>
        <summary>System prompt ({len(prompt['system_prompt'])} chars)</summary>
        <pre class="context-block">{_E(prompt['system_prompt'])}</pre>
    </details>
    <details style="margin-top:0.3rem">
        <summary>User prompt ({len(prompt['user_prompt'])} chars)</summary>
        <pre class="context-block">{_E(prompt['user_prompt'])}</pre>
    </details>
</section>

</body>
</html>"""


# ── trace endpoint ─────────────────────────────────────────────────────


@router.get("/trace", response_class=HTMLResponse)
def trace(
    q: str = Query(..., min_length=1, description="Question to trace"),
    top_k: int = Query(15, ge=1, le=200, alias="M", description="Total chunks to retrieve"),
    top_n: int = Query(10, ge=1, le=50, alias="N", description="Max questions to keep"),
    chunks_per_question: int = Query(2, ge=1, le=10, alias="C", description="Leading chunks per question"),
    top_q: int = Query(10, ge=1, le=50, alias="Q", description="Recent questions in MP stats"),
    lok: Optional[int] = Query(None, description="Filter by Lok Sabha number"),
    session: Optional[int] = Query(None, description="Filter by session number"),
    ministry: Optional[str] = Query(None, description="Filter by ministry"),
    mp: Optional[List[str]] = Query(None, description="Filter by MP name(s)"),
    fmt: Optional[str] = Query(None, description="Response format: 'json' for raw JSON"),
):
    """Run the full retrieval pipeline and return every intermediate stage.

    No LLM call is made — the trace stops after building the prompt.

    Example URLs:
        /api/debug/trace?q=air+pollution
        /api/debug/trace?q=health+policy&M=20&N=5&C=3
        /api/debug/trace?q=questions+asked&mp=Shri+Rahul+Gandhi
        /api/debug/trace?q=air+pollution&fmt=json   (raw JSON)
    """
    # Sanitise empty form values → None so they don't become bogus filters
    ministry = ministry.strip() if ministry else None
    mp = [name for name in (mp or []) if name.strip()] or None

    data = _run_trace(q, top_k, top_n, chunks_per_question, top_q, lok, session, ministry, mp)

    if fmt == "json":
        return HTMLResponse(
            content=json.dumps(data, indent=2, ensure_ascii=False),
            media_type="application/json",
        )

    return HTMLResponse(_render_html(data, q, data["input"]["params"]))
