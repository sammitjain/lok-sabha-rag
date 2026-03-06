"""Build retrieval chunks from the HuggingFace dataset.

Reads the published dataset (opensansad/lok-sabha-qa) and produces chunked
JSONL files suitable for embedding and indexing into Qdrant.

Output:
- Chunks JSONL: data/<lok>/<chunks/session_<n>/chunks.jsonl

Features:
- Uses 'BAAI/bge-small-en-v1.5' tokenizer for exact token counts.
- Prepends a structured context header to every chunk.
- Reads from HF dataset by default; supports a local parquet override.

Usage:
  uv run python -m lok_sabha_rag.pipeline.build_chunks_bge
  uv run python -m lok_sabha_rag.pipeline.build_chunks_bge --parquet /path/to/local.parquet
  uv run python -m lok_sabha_rag.pipeline.build_chunks_bge --max-files 10
"""

from __future__ import annotations

import gc
import uuid
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import typer
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

from lok_sabha_rag.config import DATA_DIR, HF_DATASET_REPO

app = typer.Typer(no_args_is_help=True)

# 2026 Standard for lightweight RAG
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _pdf_filename_from_url(url: str | None) -> str | None:
    """Extract filename from a sansad.in download URL."""
    if not url:
        return None
    fname = url.split("/")[-1].split("?")[0]
    return fname if fname else None


def _clean_markdown(md: str) -> str:
    md = (md or "").replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _chunk_id(*parts: str) -> str:
    key = "\x1f".join(parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _make_header(row: dict) -> str:
    MAX_NAMES = 3

    members = row.get("members") or []
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
        f"Lok Sabha: {row.get('lok_no')} | Session: {row.get('session_no')} | Q: {row.get('ques_no')} | {row.get('type')} | Date: {row.get('date')}",
        f"Ministry: {row.get('ministry')}",
    ]

    if asked_by_line:
        lines.append(asked_by_line)

    lines.append(f"Subject: {row.get('subject')}")
    lines.append("---")

    return "\n".join(lines)


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
    overlap_chars: int,
) -> List[str]:
    """Splits text ensuring strict token limits, falling back to char split if needed."""
    available_tokens = max_tokens - header_tokens - 5  # 5 token buffer

    if available_tokens <= 50:
        available_tokens = 200  # Emergency fallback if header is huge

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current_chunk = []
    current_len = 0

    skip_tokenize_chars = available_tokens * 5

    for p in paras:
        if len(p) > skip_tokenize_chars:
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
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0

            if count > available_tokens:
                max_chars = available_tokens * 4
                parts = _soft_split(p, max_chars=max_chars, overlap=overlap_chars)
                chunks.extend(parts)
                continue

        current_chunk.append(p)
        current_len += count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def _load_dataset(dataset: str, parquet: str | None) -> list[dict]:
    """Load dataset rows from HF or a local parquet file."""
    if parquet:
        from datasets import Dataset as HFDataset

        print(f"Loading from local parquet: {parquet}")
        ds = HFDataset.from_parquet(parquet)
    else:
        from datasets import load_dataset

        print(f"Loading from HuggingFace: {dataset}")
        ds = load_dataset(dataset, split="train")

    return list(ds)


@app.command()
def run(
    dataset: str = typer.Option(HF_DATASET_REPO, help="HuggingFace dataset repo ID"),
    parquet: Optional[str] = typer.Option(None, help="Local parquet file (overrides --dataset)"),
    data_dir: str = typer.Option(str(DATA_DIR), "--data-dir", help="Output directory for chunks"),
    max_tokens: int = typer.Option(500, help="Strict max tokens per chunk (model limit is 512)."),
    overlap_chars: int = typer.Option(300, help="Overlap for soft splitting massive paragraphs."),
    overwrite: bool = typer.Option(False, help="Overwrite existing chunks.jsonl"),
    max_files: Optional[int] = typer.Option(None, help="Testing limit (max questions to process)"),
) -> None:
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer_path = hf_hub_download(repo_id=MODEL_NAME, filename="tokenizer.json")
    hf_tokenizer = Tokenizer.from_file(tokenizer_path)

    rows = _load_dataset(dataset, parquet)
    print(f"Loaded {len(rows)} questions from dataset")

    # Group rows by (lok_no, session_no) for output organization
    by_lok_session: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        lok = row.get("lok_no")
        sess = row.get("session_no")
        if lok is not None and sess is not None:
            by_lok_session[(lok, sess)].append(row)

    base = Path(data_dir)
    failed_log = base / "failed_chunk.txt"
    total_processed = 0
    total_chunks = 0

    for lok, sess in sorted(by_lok_session):
        session_rows = by_lok_session[(lok, sess)]
        out_dir = base / str(lok) / "chunks" / f"session_{sess}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "chunks.jsonl"
        tmp_path = out_dir / "chunks.jsonl.tmp"

        if out_path.exists() and not overwrite:
            print(f"Lok {lok} Session {sess}: chunks exist, skipping.")
            continue

        print(f"Lok {lok} Session {sess}: Building chunks for {len(session_rows)} questions -> {out_path}")
        tmp_path.unlink(missing_ok=True)

        chunk_count = 0

        with tmp_path.open("a", encoding="utf-8") as out:
            for row in session_rows:
                total_processed += 1

                try:
                    body = row.get("full_text") or ""
                    body = _clean_markdown(body)
                    if not body:
                        continue

                    pdf_filename = _pdf_filename_from_url(row.get("pdf_url"))
                    lok_no = row.get("lok_no")

                    header = _make_header(row)
                    header_token_count = len(hf_tokenizer.encode(header, add_special_tokens=False).ids)

                    body_chunks = _split_with_tokenizer(
                        text=body,
                        tokenizer=hf_tokenizer,
                        max_tokens=max_tokens,
                        header_tokens=header_token_count,
                        overlap_chars=overlap_chars,
                    )

                    for i, b in enumerate(body_chunks):
                        text = f"{header}\n\n{b}".strip()
                        cid = _chunk_id(str(lok_no), str(sess), pdf_filename or "", str(i), text[:100])

                        rec = {
                            "chunk_id": cid,
                            "text": text,
                            "source": {
                                "pdf_filename": pdf_filename,
                                "pdf_url": row.get("pdf_url"),
                                "chunk_index": i,
                            },
                            "meta": {
                                "lok_no": lok_no,
                                "session_no": sess,
                                "ques_no": row.get("ques_no"),
                                "type": row.get("type"),
                                "date": row.get("date"),
                                "ministry": row.get("ministry"),
                                "mp_names": row.get("members"),
                                "subject": row.get("subject"),
                            },
                            "pipeline": {
                                "model": MODEL_NAME,
                                "max_tokens": max_tokens,
                            },
                        }
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        chunk_count += 1

                except Exception as e:
                    qid = row.get("id", "unknown")
                    print(f"[error] {qid} -> {e}")
                    with failed_log.open("a", encoding="utf-8") as f:
                        f.write(f"{sess}\t{qid}\t{repr(e)}\n")

                if total_processed % 500 == 0:
                    gc.collect()
                    print(f"  ... {total_processed} questions, {total_chunks + chunk_count} chunks so far")

                if max_files and total_processed >= max_files:
                    break

        tmp_path.replace(out_path)
        total_chunks += chunk_count
        print(f"Lok {lok} Session {sess}: Wrote {chunk_count} chunks.")
        gc.collect()

        if max_files and total_processed >= max_files:
            print("Reached --max-files limit")
            break

    print(f"\nDone. Processed {total_processed} questions, wrote {total_chunks} chunks.")


if __name__ == "__main__":
    app()
