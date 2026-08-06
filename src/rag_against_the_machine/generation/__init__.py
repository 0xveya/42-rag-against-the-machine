"""Transformers generation interface."""

from rag_against_the_machine.generation.functional import (
    AnswerFunction,
    StreamAnswerFunction,
)
from rag_against_the_machine.generation.service import (
    AnswerFunctions,
    create_answer_function,
    create_answer_functions,
    generate_answer,
    generate_answer_stream,
)

__all__ = [
    "AnswerFunction",
    "AnswerFunctions",
    "StreamAnswerFunction",
    "create_answer_function",
    "create_answer_functions",
    "generate_answer",
    "generate_answer_stream",
]
