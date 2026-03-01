"""Members endpoint - serves MP names for autocomplete."""

import json
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter

router = APIRouter()

DEFAULT_DATA_DIR = "data"

_cache: Dict[int, List[str]] = {}


def _load_members(lok: int) -> List[str]:
    if lok in _cache:
        return _cache[lok]

    path = Path(DEFAULT_DATA_DIR) / str(lok) / "members.json"
    if not path.exists():
        _cache[lok] = []
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    names = sorted(
        {entry["mpName"] for entry in data if entry.get("mpName")},
        key=str.casefold,
    )
    _cache[lok] = names
    return names


@router.get("/members/{lok}")
def get_members(lok: int) -> List[str]:
    return _load_members(lok)
