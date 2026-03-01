#!/usr/bin/env python3
"""Create a sample data subset for the Lok Sabha RAG starter dataset.

Extracts the first N unique questions from a session's chunks JSONL,
along with matching index entries, for inclusion in the git repo.

Usage:
    uv run python scripts/create_sample_data.py
    uv run python scripts/create_sample_data.py --n-questions 50
    uv run python scripts/create_sample_data.py --lok 18 --session 7 --n-questions 100
"""
from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def create_sample(
    n_questions: int = typer.Option(100, help="Number of unique questions to include"),
    data_dir: str = typer.Option("data", help="Root data directory"),
    lok: int = typer.Option(18, help="Lok Sabha number"),
    session: int = typer.Option(7, help="Session number"),
    output_dir: str = typer.Option("data/sample", help="Output directory for sample data"),
) -> None:
    """Extract a small sample of chunks + index entries for the starter dataset."""
    root = Path(data_dir)
    out = Path(output_dir)

    # -- Paths --
    chunks_path = root / str(lok) / "chunks" / f"session_{session}" / "chunks.jsonl"
    index_path = root / str(lok) / f"index_session_{session}.jsonl"

    if not chunks_path.exists():
        typer.echo(f"Error: chunks file not found: {chunks_path}")
        raise typer.Exit(1)
    if not index_path.exists():
        typer.echo(f"Error: index file not found: {index_path}")
        raise typer.Exit(1)

    # -- Pass 1: scan chunks, collect first N unique questions --
    seen_questions: OrderedDict[tuple[int, str], bool] = OrderedDict()
    chunk_count = 0
    sample_chunks: list[dict] = []

    typer.echo(f"Scanning {chunks_path} for first {n_questions} questions...")

    with open(chunks_path) as f:
        for line in f:
            record = json.loads(line)
            meta = record.get("meta", {})
            qkey = (meta.get("ques_no"), meta.get("type", ""))

            if qkey not in seen_questions:
                if len(seen_questions) >= n_questions:
                    break
                seen_questions[qkey] = True

            sample_chunks.append(record)
            chunk_count += 1

    typer.echo(f"  Found {len(seen_questions)} questions, {chunk_count} chunks")

    # -- Write sample chunks --
    out_chunks = out / str(lok) / "chunks" / f"session_{session}" / "chunks.jsonl"
    out_chunks.parent.mkdir(parents=True, exist_ok=True)

    with open(out_chunks, "w") as f:
        for record in sample_chunks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    typer.echo(f"  Wrote {out_chunks} ({out_chunks.stat().st_size / 1024:.0f} KB)")

    # -- Write matching index entries --
    out_index = out / str(lok) / f"index_session_{session}.jsonl"
    out_index.parent.mkdir(parents=True, exist_ok=True)

    index_count = 0
    with open(index_path) as fin, open(out_index, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            qkey = (entry.get("ques_no"), entry.get("type", ""))
            if qkey in seen_questions:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                index_count += 1

    typer.echo(f"  Wrote {out_index} ({index_count} entries)")

    # -- Create snapshots dir (for Qdrant snapshot placement) --
    snapshots_dir = root / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    (snapshots_dir / ".gitkeep").touch()

    typer.echo("")
    typer.echo("Sample data created. Next steps:")
    typer.echo("  1. docker compose up -d")
    typer.echo(f"  2. uv run python -m lok_sabha_rag.pipeline.embed_index_qdrant2 --data-dir {output_dir}")
    typer.echo("  3. curl -X POST http://localhost:6333/collections/lok_sabha_questions/snapshots")
    typer.echo("  4. Copy the snapshot file to data/snapshots/")


if __name__ == "__main__":
    app()
