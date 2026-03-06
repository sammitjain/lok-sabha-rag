"""Build a SQLite metadata database from the HuggingFace dataset.

Usage:
    uv run python -m lok_sabha_rag.pipeline.build_metadata_db
    uv run python -m lok_sabha_rag.pipeline.build_metadata_db --db-path data/metadata.db
    uv run python -m lok_sabha_rag.pipeline.build_metadata_db --parquet /path/to/local.parquet
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import typer

from lok_sabha_rag.config import DATA_DIR, HF_DATASET_REPO, METADATA_DB_PATH

app = typer.Typer(add_completion=False)

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS questions (
    lok_no       INTEGER,
    session_no   INTEGER,
    ques_no      INTEGER,
    type         TEXT,
    date         TEXT,
    ministry     TEXT,
    subject      TEXT,
    pdf_filename TEXT,
    PRIMARY KEY (lok_no, session_no, type, ques_no)
);

CREATE TABLE IF NOT EXISTS question_mps (
    lok_no     INTEGER,
    session_no INTEGER,
    ques_no    INTEGER,
    type       TEXT,
    mp_name    TEXT
);

CREATE INDEX IF NOT EXISTS idx_qmp_question
    ON question_mps (lok_no, session_no, type, ques_no);

CREATE INDEX IF NOT EXISTS idx_qmp_mp
    ON question_mps (mp_name);
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pdf_filename_from_url(url: str | None) -> str | None:
    """Extract filename from a sansad.in download URL."""
    if not url:
        return None
    fname = url.split("/")[-1].split("?")[0]
    return fname if fname else None


def _load_dataset(dataset: str, parquet: str | None) -> list[dict]:
    """Load dataset rows from HF or a local parquet file."""
    if parquet:
        from datasets import Dataset as HFDataset

        typer.echo(f"Loading from local parquet: {parquet}")
        ds = HFDataset.from_parquet(parquet)
    else:
        from datasets import load_dataset

        typer.echo(f"Loading from HuggingFace: {dataset}")
        ds = load_dataset(dataset, split="train")

    return list(ds)


# ── Core ──────────────────────────────────────────────────────────────────────


def _build(rows: list[dict], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    q_count = 0
    mp_count = 0

    for row in rows:
        lok = row.get("lok_no")
        sess = row.get("session_no")
        qno = row.get("ques_no")
        if lok is None or sess is None or qno is None:
            continue

        qtype = row.get("type")

        conn.execute(
            "INSERT OR REPLACE INTO questions "
            "(lok_no, session_no, ques_no, type, date, ministry, subject, pdf_filename) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                lok,
                sess,
                qno,
                qtype,
                row.get("date"),  # already YYYY-MM-DD in the parquet
                row.get("ministry"),
                row.get("subject") or None,
                _pdf_filename_from_url(row.get("pdf_url")),
            ),
        )
        q_count += 1

        # Delete existing MP rows for this question (idempotent on re-run)
        conn.execute(
            "DELETE FROM question_mps WHERE lok_no=? AND session_no=? AND type=? AND ques_no=?",
            (lok, sess, qtype, qno),
        )
        for mp in row.get("members") or []:
            mp = mp.strip()
            if mp:
                conn.execute(
                    "INSERT INTO question_mps (lok_no, session_no, ques_no, type, mp_name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (lok, sess, qno, qtype, mp),
                )
                mp_count += 1

    conn.commit()
    conn.close()

    typer.echo(f"Done — {q_count} questions, {mp_count} MP rows → {db_path.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def build(
    dataset: str = typer.Option(HF_DATASET_REPO, help="HuggingFace dataset repo ID"),
    parquet: Optional[str] = typer.Option(None, help="Local parquet file (overrides --dataset)"),
    db_path: Path = typer.Option(METADATA_DB_PATH, "--db-path", help="Output SQLite path"),
) -> None:
    """Build the metadata SQLite database from the HuggingFace dataset."""
    rows = _load_dataset(dataset, parquet)
    typer.echo(f"Loaded {len(rows)} rows from dataset")
    _build(rows, db_path)


if __name__ == "__main__":
    app()
