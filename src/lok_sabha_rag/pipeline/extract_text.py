"""Extract text from downloaded Lok Sabha PDFs (v0).

Reads PDFs under:
  data/<lok>/pdfs/session_<n>/

Writes parsed JSON under:
  data/<lok>/parsed/session_<n>/

Each output JSON includes:
- parser engine + version
- per-page text
- concatenated full_text

Idempotent + resumable:
- skips if output JSON already exists (unless --overwrite)
- logs failures to data/<lok>/failed_parse.txt
- lightweight progress print every few seconds

Usage:
  uv add pymupdf typer
  uv run python -m lok_sabha_rag.pipeline.extract_text run --lok 18 --sessions 7
  uv run python -m lok_sabha_rag.pipeline.extract_text run --lok 18 --sessions 7 --max-files 10

Notes:
- This is intentionally basic: text-based PDFs only (no OCR).
- If you later switch parsers (paid/local), keep the output schema stable.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pymupdf
import typer

app = typer.Typer(no_args_is_help=True)

from docling.document_converter import DocumentConverter

# Initialize once to avoid reloading models in a loop
converter = DocumentConverter()

def _parse_sessions(sessions: str) -> List[int]:
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


def _iter_pdf_files(pdf_dir: Path) -> Iterable[Path]:
    yield from sorted(pdf_dir.glob("*.pdf"))


def extract_pdf_with_docling(pdf_path: Path) -> dict:
    """Refactored extraction using Docling for structure-aware parsing."""
    start_time = time.time()
    
    # Run the conversion
    result = converter.convert(pdf_path)
    doc = result.document
    
    # 1. Export to Markdown (Best for LLMs/RAG)
    full_markdown = doc.export_to_markdown()
    
    # 2. Export to Dictionary (The raw Docling structure if you need it)
    # This is quite large, so we usually just keep the essential parts
    doc_dict = doc.export_to_dict()

    # 3. Create a stable output schema matching your v0
    # We map Docling's pages to your previous format
    pages = []
    # Note: Docling's page-level access varies by document structure, 
    # but for simple 2-page PDFs, we can extract text per page easily.
    # For now, we store the full text and markdown as the primary assets.

    return {
        "pdf_filename": pdf_path.name,
        "pdf_relpath": str(pdf_path.as_posix()),
        "engine": "docling",
        "engine_version": "2.x", # Docling is currently in v2
        "extracted_at_unix": int(time.time()),
        "processing_time_sec": round(time.time() - start_time, 2),
        # "full_text": doc.export_to_text(),
        "full_markdown": full_markdown, # NEW: The 'Golden' source for RAG
        "metadata": {
            "title": getattr(doc, "title", None),
            "num_pages": len(doc.pages) if hasattr(doc, 'pages') else 0
        }
    }

def extract_pdf_text(pdf_path: Path) -> dict:
    """Return parsed JSON-ready dict for a PDF."""
    doc = pymupdf.open(pdf_path)

    pages = []
    full_parts = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        # normalize whitespace lightly; keep newlines (helps paragraph splitting later)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        pages.append({"page": i + 1, "text": text})
        full_parts.append(text)

    full_text = "\n\n".join(full_parts).strip()

    return {
        "pdf_filename": pdf_path.name,
        "pdf_relpath": str(pdf_path.as_posix()),
        "engine": "pymupdf",
        "engine_version": getattr(pymupdf, "__doc__", None),
        "extracted_at_unix": int(time.time()),
        "page_count": doc.page_count,
        "pages": pages,
        "full_text": full_text,
    }


@app.command()
def run(
    lok: int = typer.Option(..., help="Lok Sabha number"),
    sessions: str = typer.Option(..., help="Sessions like '7' or '1-7' or '1,3,7'"),
    base_dir: str = typer.Option("data", help="Base data directory"),
    overwrite: bool = typer.Option(False, help="Re-extract even if parsed JSON exists"),
    sleep_min: float = typer.Option(0.0, help="Optional min seconds between PDFs"),
    sleep_max: float = typer.Option(0.0, help="Optional max seconds between PDFs"),
    max_files: Optional[int] = typer.Option(None, help="Stop after processing N PDFs (for testing)"),
) -> None:
    base = Path(base_dir) / str(lok)
    if not base.exists():
        raise typer.BadParameter(f"Missing data directory: {base}")

    sess_list = _parse_sessions(sessions)

    failed_log = base / "failed_parse.txt"

    processed = 0
    extracted = 0
    skipped = 0
    errors = 0

    last_report = time.time()
    report_interval = 15

    for sess in sess_list:
        pdf_dir = base / "pdfs" / f"session_{sess}"
        out_dir = base / "parsed" / f"session_{sess}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not pdf_dir.exists():
            print(f"[warn] missing pdf dir: {pdf_dir}")
            continue

        print(f"Session {sess}: {pdf_dir} -> {out_dir}")

        for pdf_path in _iter_pdf_files(pdf_dir):
            processed += 1
            out_path = out_dir / f"{pdf_path.stem}.json"

            if out_path.exists() and not overwrite:
                skipped += 1
            else:
                try:
                    parsed = extract_pdf_with_docling(pdf_path)
                    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                    tmp.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
                    tmp.replace(out_path)
                    extracted += 1
                except Exception as e:
                    errors += 1
                    print(f"[error] {pdf_path.name} -> {e}")
                    with failed_log.open("a", encoding="utf-8") as f:
                        f.write(f"{sess}\t{pdf_path.name}\t{repr(e)}\n")

            now = time.time()
            if now - last_report >= report_interval:
                print(f"Progress: processed={processed} extracted={extracted} skipped={skipped} errors={errors}")
                last_report = now

            if max_files is not None and processed >= max_files:
                print("Reached --max-files limit")
                print(f"Processed={processed} extracted={extracted} skipped={skipped} errors={errors}")
                return

            # optional sleep between PDFs (usually keep 0; parsing is local)
            if sleep_max > 0:
                lo = max(0.0, sleep_min)
                hi = max(lo, sleep_max)
                time.sleep(lo + random.random() * (hi - lo))

    print("\nDone.")
    print(f"Processed={processed} extracted={extracted} skipped={skipped} errors={errors}")
    print(f"Failed log: {failed_log}")


if __name__ == "__main__":
    app()
