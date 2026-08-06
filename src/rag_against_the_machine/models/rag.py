"""Pydantic models exchanged by retrieval, generation, and API layers."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A source span that can be shown to a caller."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question for which no grounded answer was produced."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question answered using retrieved source spans."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A collection of answered and unanswered questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieval output without an answer."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieval output together with a grounded answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a batch/query response."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and generated answers for a batch/query response."""

    search_results: list[MinimalAnswer]
    k: int


RagQuestion = AnsweredQuestion | UnansweredQuestion
RagResult = MinimalAnswer | UnansweredQuestion
