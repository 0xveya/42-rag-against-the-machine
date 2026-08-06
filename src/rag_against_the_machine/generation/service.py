"""Typed selection and execution for Transformers answer closures."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from rag_against_the_machine.errors import (
    Diagnostic,
    Err,
    GenerationError,
    Ok,
    Result,
)
from rag_against_the_machine.generation.functional import (
    AnswerFunction,
    StreamAnswerFunction,
    TransformersAnswers,
    make_transformers_answers,
)
from rag_against_the_machine.storage.db import SearchHit


@dataclass(frozen=True)
class AnswerFunctions:
    """Typed normal and streaming generation functions."""

    answer: AnswerFunction
    stream: StreamAnswerFunction


def generation_failure(error: BaseException) -> Err[GenerationError]:
    """Convert a backend exception into the project error format."""
    return Err(
        GenerationError.ANSWER_FAILED,
        diagnostic=Diagnostic(
            filename="generation",
            line_num=1,
            line_text="model.generate()",
            col_start=0,
            col_end=len("model.generate()"),
            help_msg=f"{type(error).__name__}: {error}",
        ),
        context_msg="Could not generate an answer",
        namespace="generation",
    )


def create_answer_functions(
    model_name: str = "Qwen/Qwen3-0.6B",
) -> Result[AnswerFunctions, GenerationError]:
    """Load Transformers and return both generation APIs."""
    try:
        functions: TransformersAnswers = make_transformers_answers(model_name)
        return Ok(AnswerFunctions(answer=functions.answer, stream=functions.stream))
    except (ImportError, RuntimeError, ValueError, OSError, TypeError) as error:
        return Err(
            GenerationError.MODEL_LOAD_FAILED,
            context_msg=f"Could not load Transformers model: {error}",
            namespace="generation",
        )


def create_answer_function(
    model_name: str = "Qwen/Qwen3-0.6B",
) -> Result[AnswerFunction, GenerationError]:
    """Backward-compatible factory for the normal answer closure."""
    result = create_answer_functions(model_name)
    if isinstance(result, Err):
        return result
    return Ok(result.value.answer)


def generate_answer(
    answer_function: AnswerFunction,
    question: str,
    hits: list[SearchHit],
    max_new_tokens: int = 256,
) -> Result[str, GenerationError]:
    """Generate one complete answer and normalize backend failures."""
    try:
        return Ok(answer_function(question, hits, max_new_tokens))
    except Exception as error:
        return generation_failure(error)


def generate_answer_stream(
    stream_function: StreamAnswerFunction,
    question: str,
    hits: list[SearchHit],
    max_new_tokens: int = 256,
) -> Iterator[Result[str, GenerationError]]:
    """Yield text chunks, followed by an error if generation fails."""
    try:
        for chunk in stream_function(question, hits, max_new_tokens):
            yield Ok(chunk)
    except Exception as error:
        yield generation_failure(error)
