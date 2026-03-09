#!/usr/bin/env python3
"""Create a sample data subset for quick Qdrant testing.

Extracts the first N unique questions from a session's chunks JSONL
for lightweight embedding and indexing.

Prerequisites:
    First build chunks from the HF dataset:
    uv run python -m lok_sabha_rag.pipeline.build_chunks

Usage:
    uv run python scripts/create_sample_data.py
    uv run python scripts/create_sample_data.py --n-questions 50
    uv run python scripts/create_sample_data.py --lok 18 --session 7 --n-questions 100
"""
from __future__ import annotations

import json
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
    """Extract a small sample of chunks for quick embedding/indexing."""
    root = Path(data_dir)
    out = Path(output_dir)

    chunks_path = root / str(lok) / "chunks" / f"session_{session}" / "chunks.jsonl"

    if not chunks_path.exists():
        typer.echo(f"Error: chunks file not found: {chunks_path}")
        typer.echo("Run 'uv run python -m lok_sabha_rag.pipeline.build_chunks' first.")
        raise typer.Exit(1)

    # Scan chunks, collect first N unique questions
    seen_questions: OrderedDict[tuple[int, str], bool] = OrderedDict()
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

    typer.echo(f"  Found {len(seen_questions)} questions, {len(sample_chunks)} chunks")

    # Write sample chunks
    out_chunks = out / str(lok) / "chunks" / f"session_{session}" / "chunks.jsonl"
    out_chunks.parent.mkdir(parents=True, exist_ok=True)

    with open(out_chunks, "w") as f:
        for record in sample_chunks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    typer.echo(f"  Wrote {out_chunks} ({out_chunks.stat().st_size / 1024:.0f} KB)")

    # Create snapshots dir (for Qdrant snapshot placement)
    snapshots_dir = root / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("")
    typer.echo("Sample data created. Next steps:")
    typer.echo("  1. docker compose up -d")
    typer.echo(f"  2. uv run python -m lok_sabha_rag.pipeline.embed --data-dir {output_dir}")
    typer.echo("  3. uv run python -m lok_sabha_rag.pipeline.build_metadata_db")
    typer.echo("  4. uv run python main.py")


if __name__ == "__main__":
    app()
