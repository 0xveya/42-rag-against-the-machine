"""Small, private SQLite FTS query normalizer used by ``RagService``."""

from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "class", "default", "did", "do", "does", "for", "from", "function",
    "how", "in", "is", "it", "its", "method", "module", "of", "on",
    "or", "passed", "return", "returns", "that", "the", "these", "this",
    "those", "to", "use", "used", "using", "value", "values", "vllm",
    "was", "were", "what", "when", "where", "whether", "which", "who",
    "why", "with",
}


def prepare_fts_query(question: str) -> str:
    """Turn user text into a safe, low-noise FTS5 OR expression."""
    tokens = [
        token.lower()
        for token in _TOKEN_PATTERN.findall(question)
        if token.lower() not in _STOPWORDS
    ]
    return " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))


def prepare_identifier_terms(question: str) -> tuple[str, ...]:
    """Return explicit snake-case and CamelCase identifiers from a question."""
    return tuple(
        dict.fromkeys(
            token.lower()
            for token in _TOKEN_PATTERN.findall(question)
            if "_" in token
            or any(character.isupper() for character in token[1:])
        )
    )


def prepare_path_terms(question: str) -> tuple[str, ...]:
    """Extract likely module names and normalize classes to filenames."""
    raw_tokens = [
        token
        for token in _TOKEN_PATTERN.findall(question)
        if token.lower() not in _STOPWORDS and len(token) >= 3
    ]
    terms: list[str] = [token.lower() for token in raw_tokens]
    for token in raw_tokens:
        words = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", token)
        snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words).lower()
        terms.append(snake)
    lowered = [token.lower() for token in raw_tokens]
    for width in (2, 3):
        terms.extend(
            "_".join(lowered[index : index + width])
            for index in range(len(lowered) - width + 1)
        )
    return tuple(dict.fromkeys(term for term in terms if term != "vllm"))
