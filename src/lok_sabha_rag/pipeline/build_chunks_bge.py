"""Build retrieval chunks (doc-level) with PRECISE tokenization.

Inputs:
- Parsed JSON: data/<lok>/parsed/session_<n>/*.json
- Curated metadata: data/<lok>/index_session_<n>.jsonl

Output:
- Chunks JSONL: data/<lok>/chunks/session_<n>/chunks.jsonl

Features:
- Uses 'BAAI/bge-small-en-v1.5' tokenizer for exact counts.
- Prepends Context Header to EVERY chunk.
- Restores robust error logging and configurable overlap.
"""

from __future__ import annotations

import gc
import uuid
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import typer
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

app = typer.Typer(no_args_is_help=True)

# 2026 Standard for lightweight RAG
MODEL_NAME = "BAAI/bge-small-en-v1.5"

def _parse_sessions(sessions: str) -> List[int]:
    """Parses session string (e.g., '1-7' or '1,3,5') into a list of integers."""
    out: List[int] = []
    parts = [p.strip() for p in sessions.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a.strip())
            end = int(b.strip())
            if end < start:
                start, end = end, start
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))

    seen = set()
    uniq: List[int] = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def _iter_parsed_files(parsed_dir: Path) -> Iterable[Path]:
    yield from sorted(parsed_dir.glob("*.json"))

def _load_metadata_index(index_path: Path) -> Dict[str, dict]:
    m: Dict[str, dict] = {}
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            url = obj.get("questionsFilePath")
            if not url:
                continue
            fname = url.split("/")[-1].split("?")[0]
            m[fname] = obj
    return m

def _clean_markdown(md: str) -> str:
    md = (md or "").replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _chunk_id(*parts: str) -> str:
    key = "\x1f".join(parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

def _make_header(meta: dict) -> str:
    MAX_NAMES = 3

    members = meta.get("members") or []
    members = [m.strip() for m in members if m.strip()]

    if members:
        if len(members) > MAX_NAMES:
            shown = members[:MAX_NAMES]
            remaining = len(members) - MAX_NAMES
            members_str = f"{', '.join(shown)} + {remaining} more"
        else:
            members_str = ", ".join(members)

        asked_by_line = f"Asked by: {members_str}"
    else:
        asked_by_line = None

    lines = [
        f"Lok Sabha: {meta.get('lok_no')} | Session: {meta.get('session_no')} | Q: {meta.get('ques_no')} | {meta.get('type')} | Date: {meta.get('date')}",
        f"Ministry: {meta.get('ministry')}",
    ]

    if asked_by_line:
        lines.append(asked_by_line)

    lines.append(f"Subject: {meta.get('subjects')}")
    lines.append("---")

    header = "\n".join(lines)
    return header
    # lines = [
    #     f"Lok Sabha: {meta.get('lok_no')} | Session: {meta.get('session_no')} | Q: {meta.get('ques_no')} | {meta.get('type')} | Date: {meta.get('date')}",
    #     f"Ministry: {meta.get('ministry')}",
    #     f"Members: {members_str}",
    #     f"Subject: {meta.get('subjects')}",
    #     "---",
    # ]
    # return "\n".join([x for x in lines if x is not None])

def _soft_split(text: str, max_chars: int, overlap: int) -> List[str]:
    """Fallback: Splits text strictly by character count."""
    if len(text) <= max_chars:
        return [text]
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out

def _split_with_tokenizer(
    text: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    header_tokens: int,
    overlap_chars: int
) -> List[str]:
    """Splits text ensuring strict token limits, falling back to char split if needed."""

    # Calculate available space for content
    available_tokens = max_tokens - header_tokens - 5 # 5 token buffer

    if available_tokens <= 50:
        available_tokens = 200 # Emergency fallback if header is huge

    # Strategy: Accumulate paragraphs until we hit the limit
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current_chunk = []
    current_len = 0

    # Char threshold beyond which we skip tokenization and go straight
    # to soft split (1 token ≈ 4 chars, so available_tokens * 4 is the
    # max chars that could possibly fit).  Tokenizing 100k+ char
    # paragraphs is extremely slow and pointless — they will always
    # exceed the token limit.
    skip_tokenize_chars = available_tokens * 5  # generous upper bound

    for p in paras:
        if len(p) > skip_tokenize_chars:
            # Oversized paragraph — commit buffer, soft-split, move on
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            max_chars = available_tokens * 4
            parts = _soft_split(p, max_chars=max_chars, overlap=overlap_chars)
            chunks.extend(parts)
            continue

        count = len(tokenizer.encode(p, add_special_tokens=False).ids)

        if current_len + count > available_tokens:
            # Commit current buffer
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            
            # If the single paragraph is STILL too big, soft split it
            if count > available_tokens:
                # Convert tokens to approx chars (1 token ~= 4 chars) for the split
                max_chars = available_tokens * 4
                parts = _soft_split(p, max_chars=max_chars, overlap=overlap_chars)
                chunks.extend(parts)
                continue

        current_chunk.append(p)
        current_len += count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks

@app.command()
def run(
    lok: int = typer.Option(..., help="Lok Sabha number"),
    sessions: str = typer.Option(..., help="Sessions like '7' or '1-7'"),
    base_dir: str = typer.Option("data", help="Base data directory"),
    max_tokens: int = typer.Option(500, help="Strict max tokens per chunk (Model limit is 512)."),
    overlap_chars: int = typer.Option(300, help="Overlap for soft splitting massive paragraphs."),
    overwrite: bool = typer.Option(False, help="Overwrite chunks.jsonl"),
    max_files: Optional[int] = typer.Option(None, help="Testing limit"),
) -> None:
    
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer_path = hf_hub_download(repo_id=MODEL_NAME, filename="tokenizer.json")
    hf_tokenizer = Tokenizer.from_file(tokenizer_path)
    
    base = Path(base_dir) / str(lok)
    failed_log = base / "failed_chunk.txt"
    sess_list = _parse_sessions(sessions)

    for sess in sess_list:
        parsed_dir = base / "parsed" / f"session_{sess}"
        index_path = base / f"index_session_{sess}.jsonl"
        out_dir = base / "chunks" / f"session_{sess}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "chunks.jsonl"
        tmp_path = out_dir / "chunks.jsonl.tmp"

        if not parsed_dir.exists() or not index_path.exists():
            print(f"[warn] Missing parsed dir or index for Session {sess}")
            continue

        if out_path.exists() and not overwrite:
            print(f"Session {sess}: chunks exist, skipping.")
            continue

        meta_by_pdf = _load_metadata_index(index_path)
        print(f"Session {sess}: Building chunks -> {out_path}")
        tmp_path.unlink(missing_ok=True)

        processed_files = 0
        chunk_count = 0

        with tmp_path.open("a", encoding="utf-8") as out:
            for parsed_path in _iter_parsed_files(parsed_dir):
                processed_files += 1
                # if processed_files == 580:
                #     # print(f"Processing: {processed_files}: {parsed_path.name}")
                #     continue
                try:
                    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
                    pdf_filename = parsed.get("pdf_filename") or (parsed_path.stem + ".pdf")
                    meta = meta_by_pdf.get(pdf_filename)
                    if not meta: continue

                    body = parsed.get("full_markdown") or parsed.get("full_text") or ""
                    body = _clean_markdown(body)
                    if not body: continue

                    header = _make_header(meta)
                    
                    # Calculate header overhead
                    header_token_count = len(hf_tokenizer.encode(header, add_special_tokens=False).ids)

                    # Intelligent Split
                    body_chunks = _split_with_tokenizer(
                        text=body,
                        tokenizer=hf_tokenizer,
                        max_tokens=max_tokens,
                        header_tokens=header_token_count,
                        overlap_chars=overlap_chars
                    )

                    for i, b in enumerate(body_chunks):
                        text = f"{header}\n\n{b}".strip()
                        cid = _chunk_id(str(lok), str(sess), pdf_filename, str(i), text[:100])
                        
                        rec = {
                            "chunk_id": cid,
                            "text": text,
                            "source": {
                                "pdf_filename": pdf_filename,
                                "chunk_index": i,
                                "pdf_relpath": parsed.get("pdf_relpath"),
                            },
                            "meta": {
                                "lok_no": meta.get("lok_no"),
                                "session_no": meta.get("session_no"),
                                "ques_no": meta.get("ques_no"),
                                "type": meta.get("type"),
                                "date": meta.get("date"),
                                "ministry": meta.get("ministry"),
                                "mp_names": meta.get("members"),
                                "subject": meta.get("subjects")
                            },
                            "pipeline": {
                                "model": MODEL_NAME,
                                "max_tokens": max_tokens,
                            }
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        chunk_count += 1

                except Exception as e:
                    print(f"[error] {parsed_path.name} -> {e}")
                    # Restored explicit file logging
                    with failed_log.open("a", encoding="utf-8") as f:
                         f.write(f"{sess}\t{parsed_path.name}\t{repr(e)}\n")

                # Free large strings between files to keep memory stable
                if processed_files % 500 == 0:
                    gc.collect()
                    print(f"  ... {processed_files} files, {chunk_count} chunks so far")

                if max_files and processed_files >= max_files: break

        tmp_path.replace(out_path)
        print(f"Session {sess}: Wrote {chunk_count} chunks.")
        del meta_by_pdf
        gc.collect()

if __name__ == "__main__":
    app()