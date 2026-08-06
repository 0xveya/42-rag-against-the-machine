"""Closure-based Transformers generation."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

from rag_against_the_machine.generation.prompt import build_messages
from rag_against_the_machine.storage.db import SearchHit

AnswerFunction = Callable[[str, list[SearchHit], int], str]


def make_transformers_answer(
    model_name: str = "Qwen/Qwen3-0.6B",
) -> AnswerFunction:
    """Load Transformers and return a synchronous answer closure."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(model_name)

    def notice(message: str) -> None:
        """Write fallback status without contaminating JSON stdout."""
        print(message, file=sys.stderr, flush=True)

    # Let Accelerate try the GPU first. A free-memory heuristic is unreliable:
    # a model can fit despite low free memory, or fail later while allocating
    # its KV cache. The load/generation OOM handlers below perform the fallback.
    force_cpu = os.environ.get("RAG_FORCE_CPU", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    device_map: str = "cpu" if force_cpu else "auto"

    def load_model(selected_device_map: str) -> Any:
        """Load the model, allowing the caller to retry on another device."""
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
        # Accelerate may discover the OOM only while placing weights.
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        device_map = "cpu"
        notice("GPU model loading failed; fell back to CPU.")
        model = load_model(device_map)

    def answer(
        question: str,
        hits: list[SearchHit],
        max_new_tokens: int = 256,
    ) -> str:
        nonlocal model, device_map
        messages = build_messages(question, hits)
        template_options: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            template_options["enable_thinking"] = False
            prompt = tokenizer.apply_chat_template(messages, **template_options)
        except TypeError:
            template_options.pop("enable_thinking", None)
            prompt = tokenizer.apply_chat_template(messages, **template_options)

        # ``device_map="auto"`` may place parts of a large model on CPU or
        # disk. Inputs must start on the model's execution device, not on a
        # guessed CUDA device (and never on ``meta``).
        device = model.device
        if getattr(device, "type", None) == "meta":
            device = next(
                parameter.device
                for parameter in model.parameters()
                if parameter.device.type != "meta"
            )

        context_limit = getattr(model.config, "max_position_embeddings", None)
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        limits = [
            limit
            for limit in (context_limit, tokenizer_limit)
            if isinstance(limit, int) and 0 < limit < 1_000_000
        ]
        max_input_tokens = min(limits) - max_new_tokens if limits else None
        tokenize_options: dict[str, Any] = {"return_tensors": "pt"}
        if max_input_tokens is not None and max_input_tokens > 0:
            tokenize_options.update({"truncation": True, "max_length": max_input_tokens})

        model_inputs = tokenizer([prompt], **tokenize_options).to(device)
        input_length = model_inputs["input_ids"].shape[1]

        try:
            with torch.inference_mode():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            if device_map == "cpu":
                raise

            # The weights may fit while the generation KV cache does not.
            # Retry on the GPU without the cache before abandoning the GPU.
            torch.cuda.empty_cache()
            notice("GPU KV cache did not fit; retrying without cache.")
            try:
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            except torch.cuda.OutOfMemoryError:
                import gc

                notice("GPU generation failed; fell back to CPU.")
                del model
                gc.collect()
                torch.cuda.empty_cache()
                device_map = "cpu"
                model = load_model(device_map)
                model_inputs = model_inputs.to("cpu")
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

        return str(
            tokenizer.decode(
                generated_ids[0, input_length:],
                skip_special_tokens=True,
            )
        ).strip()

    return answer
