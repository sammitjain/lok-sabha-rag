"""Members endpoint - serves MP names for autocomplete.

Loads members.json from the HuggingFace dataset supplementary files.
Deduplicates across Lok Sabhas using mpNo, preferring the latest lok's name
(e.g. lok 18 names include Shri/Smt/Dr. qualifiers while lok 17 has bare names).
"""

import json
from typing import Dict, List, Tuple

from fastapi import APIRouter
from huggingface_hub import hf_hub_download

from lok_sabha_rag.config import HF_DATASET_REPO

router = APIRouter()

# Raw per-lok data: {lok: [{mpNo, mpName, ...}, ...]}
_raw_cache: Dict[int, List[dict]] = {}

# Known Lok Sabha numbers that have supplementary member data on HF (ascending order)
_KNOWN_LOKS = [17, 18]


def _load_raw(lok: int) -> List[dict]:
    """Load raw member entries for a lok from HF."""
    if lok in _raw_cache:
        return _raw_cache[lok]

    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"supplementary/{lok}/members.json",
            repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = []

    _raw_cache[lok] = data
    return data


def _load_members(lok: int) -> List[str]:
    """Sorted unique MP names for a single lok."""
    entries = _load_raw(lok)
    return sorted(
        {e["mpName"] for e in entries if e.get("mpName")},
        key=str.casefold,
    )


# ── Combined (deduped by mpNo) ────────────────────────────────────────

_combined_cache: List[str] | None = None


def _load_all_members() -> List[str]:
    """Deduplicated MP names across all known loks.

    Uses mpNo as the unique key. When the same mpNo appears in multiple loks,
    the name from the latest (highest) lok wins — it tends to have the most
    complete formatting (Shri/Smt/Dr. qualifiers).
    """
    global _combined_cache
    if _combined_cache is not None:
        return _combined_cache

    # Process loks in ascending order so later loks overwrite earlier ones
    by_mpno: Dict[int, str] = {}
    for lok in sorted(_KNOWN_LOKS):
        for entry in _load_raw(lok):
            mp_no = entry.get("mpNo")
            mp_name = entry.get("mpName")
            if mp_no and mp_name:
                by_mpno[mp_no] = mp_name

    _combined_cache = sorted(by_mpno.values(), key=str.casefold)
    return _combined_cache


@router.get("/members")
def get_all_members() -> List[str]:
    """Combined deduplicated MP list across all indexed Lok Sabhas."""
    return _load_all_members()


@router.get("/members/{lok}")
def get_members(lok: int) -> List[str]:
    return _load_members(lok)
