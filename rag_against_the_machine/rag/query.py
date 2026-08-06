"""Small, private SQLite FTS query normalizer used by ``RagService``."""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def prepare_fts_query(question: str) -> str:
    """Turn user text into a safe FTS5 OR expression."""
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(question)]
    return " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))
