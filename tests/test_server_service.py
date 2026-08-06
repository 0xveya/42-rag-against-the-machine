"""Tests for the web service's generation adapter."""

import asyncio
from collections.abc import Iterator
from pathlib import Path

from rag_against_the_machine.errors import Ok, Some
from rag_against_the_machine.generation import AnswerFunctions
from rag_against_the_machine.server.models import RetrievedSource
from rag_against_the_machine.server.service import WebRagService
from rag_against_the_machine.storage.db import SearchHit


def test_stream_answer_forwards_transformers_chunks(tmp_path: Path) -> None:
    """WebSocket generation should expose chunks from the configured backend."""
    received_hits: list[SearchHit] = []

    def stream(
        question: str, hits: list[SearchHit], max_new_tokens: int
    ) -> Iterator[str]:
        assert question == "What does this do?"
        assert max_new_tokens == 32
        received_hits.extend(hits)
        yield "first "
        yield "second"

    functions = AnswerFunctions(
        answer=lambda question, hits, max_new_tokens: "unused",
        stream=stream,
    )
    service = WebRagService(
        database_path=tmp_path / "index.db",
        max_new_tokens=32,
        answer_functions=Some(functions),
    )
    source = RetrievedSource(
        file_path="data/raw/repo/example.py",
        first_character_index=10,
        last_character_index=20,
        text="return value",
    )

    async def collect() -> list[str]:
        chunks: list[str] = []
        async for result in service.stream_answer(
            "What does this do?", [source]
        ):
            assert isinstance(result, Ok)
            chunks.append(result.value)
        return chunks

    assert asyncio.run(collect()) == ["first ", "second"]
    assert len(received_hits) == 1
    assert received_hits[0].text == "return value"
