"""Closure-based Transformers generation."""

from __future__ import annotations

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
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()

    def answer(
        question: str,
        hits: list[SearchHit],
        max_new_tokens: int = 256,
    ) -> str:
        messages = build_messages(question, hits)
        template_options: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            template_options["enable_thinking"] = False
            prompt = tokenizer.apply_chat_template(
                messages, **template_options
            )
        except TypeError:
            template_options.pop("enable_thinking", None)
            prompt = tokenizer.apply_chat_template(
                messages, **template_options
            )

        model_inputs = tokenizer([prompt], return_tensors="pt").to(
            model.device
        )
        input_length = model_inputs["input_ids"].shape[1]

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
