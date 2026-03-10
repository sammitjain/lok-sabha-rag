"""Ministries endpoint - serves ministry names for autocomplete.

Loads ministries.json from the HuggingFace dataset supplementary files.
Deduplicates across Lok Sabhas using minCode, preferring the latest lok's name.
"""

import json
from typing import Dict, List

from fastapi import APIRouter
from huggingface_hub import hf_hub_download

from lok_sabha_rag.config import HF_DATASET_REPO

router = APIRouter()

_raw_cache: Dict[int, List[dict]] = {}
_KNOWN_LOKS = [17, 18]


def _load_raw(lok: int) -> List[dict]:
    """Load raw ministry entries for a lok from HF."""
    if lok in _raw_cache:
        return _raw_cache[lok]

    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"supplementary/{lok}/ministries.json",
            repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = []

    _raw_cache[lok] = data
    return data


def _load_ministries(lok: int) -> List[str]:
    """Sorted unique ministry names for a single lok."""
    entries = _load_raw(lok)
    return sorted(
        {e["minName"] for e in entries if e.get("minName")},
        key=str.casefold,
    )


_combined_cache: List[str] | None = None


def _load_all_ministries() -> List[str]:
    """Deduplicated ministry names across all known loks.

    Uses minCode as the unique key. When the same minCode appears in multiple
    loks, the name from the latest (highest) lok wins.
    """
    global _combined_cache
    if _combined_cache is not None:
        return _combined_cache

    by_code: Dict[int, str] = {}
    for lok in sorted(_KNOWN_LOKS):
        for entry in _load_raw(lok):
            code = entry.get("minCode")
            name = entry.get("minName")
            if code and name:
                by_code[code] = name

    _combined_cache = sorted(by_code.values(), key=str.casefold)
    return _combined_cache


@router.get("/ministries")
def get_all_ministries() -> List[str]:
    """Combined deduplicated ministry list across all indexed Lok Sabhas."""
    return _load_all_ministries()


@router.get("/ministries/{lok}")
def get_ministries(lok: int) -> List[str]:
    return _load_ministries(lok)
