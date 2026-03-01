"""Build a SQLite metadata database from Lok Sabha index JSONL files.

Usage:
    uv run python -m lok_sabha_rag.pipeline.build_metadata_db
    uv run python -m lok_sabha_rag.pipeline.build_metadata_db --db-path data/metadata.db
    uv run python -m lok_sabha_rag.pipeline.build_metadata_db --data-dir /other/data
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "metadata.db"

JSONL_GLOB = "*/index_session_*.jsonl"

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

_DATE_RE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})$")


def _convert_date(raw: str | None) -> str | None:
    """Convert DD.MM.YYYY → YYYY-MM-DD.  Pass through anything else as-is."""
    if not raw:
        return None
    m = _DATE_RE.match(raw.strip())
    if not m:
        return raw.strip() or None
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"


def _pdf_filename_from_url(url: str | None) -> str | None:
    """Extract filename from a sansad.in download URL."""
    if not url:
        return None
    # e.g. "https://sansad.in/getFile/.../AS500_kcKX5O.pdf?source=pqals"
    fname = url.split("/")[-1].split("?")[0]
    return fname if fname else None


# ── Core ──────────────────────────────────────────────────────────────────────

def _build(data_dir: Path, db_path: Path) -> None:
    jsonl_files = sorted(data_dir.glob(JSONL_GLOB))
    if not jsonl_files:
        typer.echo(f"No index JSONL files found matching {data_dir / JSONL_GLOB}")
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(jsonl_files)} index files in {data_dir.resolve()}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    q_count = 0
    mp_count = 0

    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    typer.echo(f"  WARN: {path.name}:{line_no} bad JSON — {exc}")
                    continue

                lok = rec.get("lok_no")
                sess = rec.get("session_no")
                qno = rec.get("ques_no")
                if lok is None or sess is None or qno is None:
                    continue

                conn.execute(
                    "INSERT OR REPLACE INTO questions "
                    "(lok_no, session_no, ques_no, type, date, ministry, subject, pdf_filename) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        lok,
                        sess,
                        qno,
                        rec.get("type"),
                        _convert_date(rec.get("date")),
                        rec.get("ministry"),
                        (rec.get("subjects") or "").strip() or None,
                        _pdf_filename_from_url(rec.get("questionsFilePath")),
                    ),
                )
                q_count += 1

                qtype = rec.get("type")

                # Delete existing MP rows for this question (idempotent on re-run)
                conn.execute(
                    "DELETE FROM question_mps WHERE lok_no=? AND session_no=? AND type=? AND ques_no=?",
                    (lok, sess, qtype, qno),
                )
                for mp in rec.get("members") or []:
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
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, "--data-dir", help="Root data directory containing lok subdirs"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help=f"Output SQLite path (default: <data-dir>/metadata.db)"),
) -> None:
    """Scan index JSONL files and build the metadata SQLite database."""
    if db_path is None:
        db_path = data_dir / "metadata.db"
    _build(data_dir, db_path)


if __name__ == "__main__":
    app()
