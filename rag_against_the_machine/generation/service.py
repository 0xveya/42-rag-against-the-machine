"""Typed selection and execution for the Transformers answer closure."""

from __future__ import annotations

from rag_against_the_machine.errors import (
    Diagnostic,
    Err,
    GenerationError,
    Ok,
    Result,
)
from rag_against_the_machine.generation.functional import (
    AnswerFunction,
    make_transformers_answer,
)
from rag_against_the_machine.storage.db import SearchHit


def create_answer_function(
    model_name: str = "Qwen/Qwen3-0.6B",
) -> Result[AnswerFunction, GenerationError]:
    """Load Transformers and return the typed answer closure."""
    try:
        return Ok(make_transformers_answer(model_name))
    except (
        ImportError,
        RuntimeError,
        ValueError,
        OSError,
        TypeError,
    ) as error:
        return Err(
            GenerationError.MODEL_LOAD_FAILED,
            context_msg=f"Could not load Transformers model: {error}",
            namespace="generation",
        )


def generate_answer(
    answer_function: AnswerFunction,
    question: str,
    hits: list[SearchHit],
    max_new_tokens: int = 256,
) -> Result[str, GenerationError]:
    """Run a closure and normalize backend failures into project errors."""
    try:
        return Ok(answer_function(question, hits, max_new_tokens))
    except Exception as error:
        # Backend errors vary by model and device-map configuration. Keep the
        # original exception visible instead of reducing it to a bare enum.
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
