"""LLM adapter for DAIL-SQL using the project's LLMFactoryAdapter.

Drop-in replacement for DAIL-SQL's OpenAI-specific ask_llm / init_chatgpt.
Reads provider/model from config/models.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

_DAIL_ROOT    = Path(__file__).resolve().parents[1]   # src/DAIL-SQL/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

for _p in (str(_PROJECT_ROOT), str(_DAIL_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm.model import LLMFactoryAdapter

_adapter: LLMFactoryAdapter | None = None


def get_adapter() -> LLMFactoryAdapter:
    global _adapter
    if _adapter is None:
        _adapter = LLMFactoryAdapter("dail_sql_generation")
    return _adapter


def generate_candidates(
    prompt: str,
    n: int = 1,
    temperature: float = 0.0,
) -> list[str]:
    """Generate n SQL candidates. Returns list of SQL strings (think blocks stripped)."""
    adapter = get_adapter()
    if n == 1:
        result = adapter.get_ans(prompt, temperature=temperature)
        return [result]
    choices = adapter.get_ans(prompt, temperature=temperature, n=n, single=False)
    return [c["message"]["content"] for c in choices]
