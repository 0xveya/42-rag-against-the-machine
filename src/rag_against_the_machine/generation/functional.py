"""Closure-based Transformers generation with optional streaming."""

from __future__ import annotations

import gc
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from queue import Empty
from threading import Lock, Thread
from typing import Any

from rag_against_the_machine.generation.prompt import build_messages
from rag_against_the_machine.storage.db import SearchHit

AnswerFunction = Callable[[str, list[SearchHit], int], str]
StreamAnswerFunction = Callable[[str, list[SearchHit], int], Iterator[str]]


@dataclass(frozen=True)
class TransformersAnswers:
    """Normal and streaming generation entry points."""

    answer: AnswerFunction
    stream: StreamAnswerFunction


class StreamingGenerationError(RuntimeError):
    """Generation failed after partial output reached the caller."""

    def __init__(self, message: str, *, partial_output_emitted: bool) -> None:
        super().__init__(message)
        self.partial_output_emitted = partial_output_emitted


def make_transformers_answers(
    model_name: str = "Qwen/Qwen3-0.6B",
) -> TransformersAnswers:
    """Load Transformers and return normal and streaming answer closures."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    tokenizer: Any = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=False
    )
    force_cpu = os.environ.get("RAG_FORCE_CPU", "").casefold() in {
        "1", "true", "yes", "on"
    }
    generation_lock = Lock()
    device_map = "cpu" if force_cpu else "auto"

    def notice(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(selected_device_map: str) -> Any:
        loaded_model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=selected_device_map,
            trust_remote_code=False,
        )
        loaded_model.eval()
        return loaded_model

    try:
        model: Any = load_model(device_map)
    except torch.cuda.OutOfMemoryError:
        clear_cuda_cache()
        device_map = "cpu"
        notice("GPU model loading failed; falling back to CPU.")
        model = load_model(device_map)

    def model_input_device() -> Any:
        device = getattr(model, "device", None)
        if getattr(device, "type", None) != "meta":
            return device
        for parameter in model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        raise RuntimeError("Could not determine a non-meta model input device.")

    def make_prompt(question: str, hits: list[SearchHit]) -> str:
        messages = build_messages(question, hits)
        options: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages, enable_thinking=False, **options
                )
            )
        except TypeError:
            return str(tokenizer.apply_chat_template(messages, **options))

    def maximum_input_tokens(max_new_tokens: int) -> int | None:
        limits = [
            limit
            for limit in (
                getattr(model.config, "max_position_embeddings", None),
                getattr(tokenizer, "model_max_length", None),
            )
            if isinstance(limit, int) and 0 < limit < 1_000_000
        ]
        if not limits:
            return None
        available = min(limits) - max_new_tokens
        if available <= 0:
            raise ValueError(
                "max_new_tokens leaves no room for input tokens: "
                f"max_new_tokens={max_new_tokens}, context_limit={min(limits)}"
            )
        return available

    def prepare_inputs(
        question: str, hits: list[SearchHit], max_new_tokens: int
    ) -> Any:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")
        options: dict[str, Any] = {"return_tensors": "pt"}
        max_input_tokens = maximum_input_tokens(max_new_tokens)
        if max_input_tokens is not None:
            options.update({"truncation": True, "max_length": max_input_tokens})
        encoded = tokenizer([make_prompt(question, hits)], **options)
        return encoded.to(model_input_device())

    def replace_model_with_cpu() -> None:
        nonlocal model, device_map
        old_model = model
        model = None
        del old_model
        gc.collect()
        clear_cuda_cache()
        device_map = "cpu"
        model = load_model("cpu")

    def run_streaming_attempt(
        model_inputs: Any, max_new_tokens: int, *, use_cache: bool
    ) -> Iterator[str]:
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=0.25,
        )
        worker_errors: list[BaseException] = []
        generation_options: dict[str, Any] = {
            **model_inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": use_cache,
            "pad_token_id": tokenizer.eos_token_id,
        }

        def generate_in_worker() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**generation_options)
            except BaseException as error:
                worker_errors.append(error)
                try:
                    streamer.end()
                except Exception:
                    pass

        worker = Thread(
            target=generate_in_worker, name="transformers-generation", daemon=True
        )
        worker.start()
        try:
            while True:
                try:
                    chunk = next(streamer)
                except StopIteration:
                    break
                except (Empty, TimeoutError):
                    if worker.is_alive():
                        continue
                    break
                if chunk:
                    yield str(chunk)
        finally:
            worker.join()
        if worker_errors:
            raise worker_errors[0]

    def stream_locked(
        question: str, hits: list[SearchHit], max_new_tokens: int
    ) -> Iterator[str]:
        """Generate while the model and fallback state are locked."""
        model_inputs = prepare_inputs(question, hits, max_new_tokens)
        emitted_to_caller = False

        def emit_attempt(*, use_cache: bool) -> Iterator[str]:
            nonlocal emitted_to_caller
            for chunk in run_streaming_attempt(
                model_inputs, max_new_tokens, use_cache=use_cache
            ):
                emitted_to_caller = True
                yield chunk

        try:
            yield from emit_attempt(use_cache=True)
            return
        except torch.cuda.OutOfMemoryError as first_oom:
            if emitted_to_caller:
                raise StreamingGenerationError(
                    "CUDA ran out of memory after streaming began; generation "
                    "was not restarted to avoid duplicate output.",
                    partial_output_emitted=True,
                ) from first_oom

        clear_cuda_cache()
        notice("GPU KV cache did not fit; retrying generation without cache.")
        try:
            yield from emit_attempt(use_cache=False)
            return
        except torch.cuda.OutOfMemoryError as second_oom:
            if emitted_to_caller:
                raise StreamingGenerationError(
                    "CUDA ran out of memory after streaming began; generation "
                    "was not restarted to avoid duplicate output.",
                    partial_output_emitted=True,
                ) from second_oom

        notice("GPU generation failed; falling back to CPU.")
        del model_inputs
        gc.collect()
        clear_cuda_cache()
        replace_model_with_cpu()
        model_inputs = prepare_inputs(question, hits, max_new_tokens)
        yield from emit_attempt(use_cache=True)

    def stream_answer(
        question: str, hits: list[SearchHit], max_new_tokens: int = 256
    ) -> Iterator[str]:
        """Yield generated text chunks without duplicate fallback output."""
        with generation_lock:
            yield from stream_locked(question, hits, max_new_tokens)

    def answer(
        question: str, hits: list[SearchHit], max_new_tokens: int = 256
    ) -> str:
        """Generate and return one complete answer."""
        return "".join(stream_answer(question, hits, max_new_tokens)).strip()

    return TransformersAnswers(answer=answer, stream=stream_answer)


def make_transformers_answer(
    model_name: str = "Qwen/Qwen3-0.6B",
) -> AnswerFunction:
    """Backward-compatible factory for the non-streaming API."""
    return make_transformers_answers(model_name).answer


def make_transformers_stream(
    model_name: str = "Qwen/Qwen3-0.6B",
) -> StreamAnswerFunction:
    """Create only the public streaming closure."""
    return make_transformers_answers(model_name).stream
