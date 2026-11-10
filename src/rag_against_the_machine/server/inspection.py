"""Build explorer output from the same sources used for generation."""

from rag_against_the_machine.generation.prompt import build_messages
from rag_against_the_machine.rag.query import prepare_fts_query
from rag_against_the_machine.server.models import (
    InspectedSource,
    InspectionMessage,
    PromptMessage,
    RetrievedSource,
)
from rag_against_the_machine.storage.db import SearchHit


def inspect_sources(question: str, sources: list[RetrievedSource]) -> InspectionMessage:
    """Expose the source evidence and messages before chat-template tokenization."""
    hits = [
        SearchHit(
            chunk_id=index,
            file_path=source.file_path,
            start_character=source.first_character_index,
            end_character=source.last_character_index,
            text=source.text,
            score=source.score,
        )
        for index, source in enumerate(sources)
    ]
    return InspectionMessage(
        query=prepare_fts_query(question),
        sources=[
            InspectedSource(
                file_path=source.file_path,
                first_character_index=source.first_character_index,
                last_character_index=source.last_character_index,
                text=source.text,
                score=source.score,
            )
            for source in sources
        ],
        messages=[PromptMessage(**message) for message in build_messages(question, hits)],
    )
