"""Transformers generation interface."""

from rag_against_the_machine.generation.functional import AnswerFunction
from rag_against_the_machine.generation.service import (
    create_answer_function,
    generate_answer,
)

__all__ = [
    "AnswerFunction",
    "create_answer_function",
    "generate_answer",
]
