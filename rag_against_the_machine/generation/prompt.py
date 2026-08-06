"""Prompt construction for grounded answer generation."""

from __future__ import annotations

from rag_against_the_machine.storage.db import SearchHit


_SYSTEM_PROMPT = (
    "Answer using only the provided source excerpts. "
    "If they do not contain the answer, say so."
)


_SYSTEM_PROMPT_MINIMAL = "Use only the provided sources."


_SYSTEM_PROMPT_TINY = "Use the provided sources."


def build_context(hits: list[SearchHit]) -> str:
    """Format retrieved chunks into model-readable context."""
    sections: list[str] = []

    for index, hit in enumerate(hits, start=1):
        sections.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"File: {hit.file_path}",
                    f"Characters: {hit.start_character}:{hit.end_character}",
                    "Content:",
                    hit.text,
                ]
            )
        )

    return "\n\n".join(sections)


def build_messages(
    question: str,
    hits: list[SearchHit],
) -> list[dict[str, str]]:
    """Build chat messages for the local model."""
    context = build_context(hits)

    user_message = f"""Question:
{question}

Retrieved sources:
{context}

Answer using only the retrieved sources.
If the retrieved sources do not contain the answer, say so.
"""

    return [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]
