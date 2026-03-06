"""Members endpoint - serves MP names for autocomplete.

Loads members.json from the HuggingFace dataset supplementary files.
"""

import json
from typing import Dict, List

from fastapi import APIRouter
from huggingface_hub import hf_hub_download

from lok_sabha_rag.config import HF_DATASET_REPO

router = APIRouter()

_cache: Dict[int, List[str]] = {}


def _load_members(lok: int) -> List[str]:
    if lok in _cache:
        return _cache[lok]

    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=f"supplementary/{lok}/members.json",
            repo_type="dataset",
        )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        names = sorted(
            {entry["mpName"] for entry in data if entry.get("mpName")},
            key=str.casefold,
        )
    except Exception:
        # File not found on HF, or parsing error
        names = []

    _cache[lok] = names
    return names


@router.get("/members/{lok}")
def get_members(lok: int) -> List[str]:
    return _load_members(lok)
