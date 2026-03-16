"""Build a SQLite metadata database from the HuggingFace dataset.

Usage:
    uv run python -m lok_sabha_rag.pipeline.build_metadata_db
    uv run python -m lok_sabha_rag.pipeline.build_metadata_db --db-path data/metadata.db
    uv run python -m lok_sabha_rag.pipeline.build_metadata_db --parquet /path/to/local.parquet
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import typer
from huggingface_hub import hf_hub_download, list_repo_tree

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


def _discover_loks(dataset_repo: str) -> list[int]:
    """Discover available Lok Sabha numbers from supplementary/ on HuggingFace."""
    loks = []
    for entry in list_repo_tree(dataset_repo, path_in_repo="supplementary", repo_type="dataset"):
        name = Path(entry.path).name
        if name.isdigit():
            loks.append(int(name))
    return sorted(loks)


def _build_mp_name_map(dataset_repo: str) -> dict[str, str]:
    """Build old_name → canonical_name map using mpNo from members.json.

    Discovers all Lok Sabhas with supplementary data on HuggingFace.  When the
    same mpNo appears in multiple Lok Sabhas with different display names,
    the most recent (highest) Lok Sabha's name is used as the canonical form.

    Returns a dict mapping every known name variant to the canonical name.
    """
    by_mpno: dict[int, list[tuple[int, str]]] = {}
    loks = _discover_loks(dataset_repo)
    typer.echo(f"  Discovered Lok Sabhas with members.json: {loks}")

    for lok in loks:
        try:
            path = hf_hub_download(
                repo_id=dataset_repo,
                filename=f"supplementary/{lok}/members.json",
                repo_type="dataset",
            )
            with open(path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            continue

        for entry in entries:
            mp_no = entry.get("mpNo")
            mp_name = entry.get("mpName")
            if mp_no and mp_name:
                by_mpno.setdefault(mp_no, []).append((lok, mp_name))

    # For each mpNo, pick the name from the latest lok as canonical
    name_map: dict[str, str] = {}
    for mp_no, entries in by_mpno.items():
        canonical = max(entries, key=lambda x: x[0])[1]
        for _, name in entries:
            name_map[name] = canonical

    return name_map


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


def _build(rows: list[dict], db_path: Path, mp_name_map: dict[str, str]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    q_count = 0
    mp_count = 0
    mp_renamed = 0

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
                canonical = mp_name_map.get(mp, mp)
                if canonical != mp:
                    mp_renamed += 1
                conn.execute(
                    "INSERT INTO question_mps (lok_no, session_no, ques_no, type, mp_name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (lok, sess, qno, qtype, canonical),
                )
                mp_count += 1

    conn.commit()
    conn.close()

    typer.echo(
        f"Done — {q_count} questions, {mp_count} MP rows "
        f"({mp_renamed} name canonicalisations) → {db_path.resolve()}"
    )


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

    typer.echo("Building MP name canonicalisation map from members.json ...")
    mp_name_map = _build_mp_name_map(dataset)
    conflicts = sum(1 for k, v in mp_name_map.items() if k != v)
    typer.echo(f"  {len(mp_name_map)} names loaded, {conflicts} will be remapped")

    _build(rows, db_path, mp_name_map)


if __name__ == "__main__":
    app()
