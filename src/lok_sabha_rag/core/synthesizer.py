"""LLM synthesis with modular prompt loading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Set

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_CITATION_RE = re.compile(r"\[Q(\d+)\]")


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8").strip()


def get_system_prompt() -> str:
    return load_prompt("system.txt")


def get_user_prompt(query: str, context: str) -> str:
    template = load_prompt("user.txt")
    return template.format(query=query, context=context)


def extract_citations(answer: str, max_n: int) -> List[int]:
    found: Set[int] = set()
    for m in _CITATION_RE.finditer(answer):
        idx = int(m.group(1))
        if 1 <= idx <= max_n:
            found.add(idx)
    return sorted(found)


def validate_citations(answer: str, evidence_count: int) -> List[str]:
    errors: List[str] = []
    for m in _CITATION_RE.finditer(answer):
        idx = int(m.group(1))
        if idx < 1 or idx > evidence_count:
            errors.append(f"Invalid citation [Q{idx}] (valid range: 1-{evidence_count})")
    return errors


class Synthesizer:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = "gpt-4o-mini"

    def generate(self, query: str, context: str) -> str:
        system_prompt = get_system_prompt()
        user_prompt = get_user_prompt(query=query, context=context)

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = getattr(resp, "output_text", None)
        if text:
            return text.strip()

        try:
            parts = []
            for item in resp.output:
                if item.type == "message":
                    for c in item.content:
                        if c.type == "output_text":
                            parts.append(c.text)
            return "\n".join(parts).strip()
        except Exception:
            raise RuntimeError("Failed to extract text from OpenAI response")

