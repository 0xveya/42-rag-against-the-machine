"""Synchronous RAG orchestration with an async server-friendly boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeAlias, cast

from rag_against_the_machine.errors import (
    Err,
    GenerationError,
    Nothing,
    Ok,
    Option,
    Result,
    Some,
    StorageError,
)
from rag_against_the_machine.generation.functional import AnswerFunction
from rag_against_the_machine.generation.service import generate_answer
from rag_against_the_machine.models.rag import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    UnansweredQuestion,
)
from rag_against_the_machine.rag.query import prepare_fts_query
from rag_against_the_machine.storage.db import SearchHit, Store

RagError: TypeAlias = StorageError | GenerationError
RagResponse: TypeAlias = MinimalAnswer | UnansweredQuestion


def _source(hit: SearchHit) -> MinimalSource:
    """Convert an internal database hit to the public source model."""
    return MinimalSource(
        file_path=hit.file_path,
        first_character_index=hit.start_character,
        last_character_index=hit.end_character,
    )


def _question_id(question_id: Option[str]) -> str:
    """Use a supplied question ID or create one for the response."""
    if isinstance(question_id, Some):
        return question_id.value

    import uuid

    return str(uuid.uuid4())


class RagService:
    """Coordinate database retrieval and a previously loaded answer closure."""

    def __init__(self, store: Store, answer_function: AnswerFunction) -> None:
        """Create a service; model loading remains outside request handling."""
        self.store = store
        self.answer_function = answer_function

    def retrieve(
        self,
        question: str,
        k: int = 5,
        question_id: Option[str] = Nothing(),
    ) -> Result[MinimalSearchResults, StorageError]:
        """Retrieve and validate source spans for one question."""
        result = _search(self.store, question, k)
        if isinstance(result, Err):
            return result

        return Ok(
            MinimalSearchResults(
                question_id=_question_id(question_id),
                question=question,
                retrieved_sources=[_source(hit) for hit in result.value],
            )
        )

    def answer(
        self,
        question: str,
        k: int = 5,
        question_id: Option[str] = Nothing(),
        max_new_tokens: int = 256,
    ) -> Result[RagResponse, RagError]:
        """Retrieve chunks and produce a grounded Pydantic response."""
        retrieval = _search(self.store, question, k)
        if isinstance(retrieval, Err):
            return cast(Result[RagResponse, RagError], retrieval)

        hits = retrieval.value
        resolved_id = _question_id(question_id)
        if not hits:
            return Ok(
                UnansweredQuestion(question_id=resolved_id, question=question)
            )

        generated = generate_answer(
            self.answer_function,
            question,
            hits,
            max_new_tokens,
        )
        if isinstance(generated, Err):
            return cast(Result[RagResponse, RagError], generated)

        return Ok(
            MinimalAnswer(
                question_id=resolved_id,
                question=question,
                retrieved_sources=[_source(hit) for hit in hits],
                answer=generated.value,
            )
        )

    async def answer_async(
        self,
        question: str,
        k: int = 5,
        question_id: Option[str] = Nothing(),
        max_new_tokens: int = 256,
    ) -> Result[RagResponse, RagError]:
        """Run the blocking SQLite/model work off an async server loop."""
        return await asyncio.to_thread(
            self.answer, question, k, question_id, max_new_tokens
        )


AsyncAnswer: TypeAlias = Awaitable[Result[RagResponse, RagError]]


def retrieve_hits(
    store: Store,
    question: str,
    k: int = 5,
) -> Result[list[SearchHit], StorageError]:
    """Retrieve internal scored hits for CLI and API adapters."""
    return _search(store, question, k)


def _search(
    store: Store,
    question: str,
    limit: int,
) -> Result[list[SearchHit], StorageError]:
    """Retrieve ranked chunks through the storage query boundary."""
    if limit <= 0 or not question.strip():
        return Ok([])
    query = prepare_fts_query(question)
    if not query:
        return Ok([])
    return store.read(lambda queries: queries.search_chunks(query, limit))
