"""Python Fire commands for indexing, retrieval, answering, and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_against_the_machine.errors import Err, Nothing, Option, Some


@dataclass(frozen=True)
class IndexOptions:
    """Arguments for the index command."""

    source_path: str = field(metadata={"help": "Directory to ingest."})
    project_path: str = field(
        metadata={"help": "Project root for stored paths."}
    )
    database_path: str = field(metadata={"help": "SQLite index path."})
    max_chunk_size: int = field(metadata={"help": "Maximum chunk size."})


@dataclass(frozen=True)
class SearchOptions:
    """Arguments for one search query."""

    query: str = field(
        metadata={"help": "Question to search.", "positional": True}
    )
    database_path: str = field(metadata={"help": "SQLite index path."})
    k: int = field(metadata={"help": "Number of sources."})


@dataclass(frozen=True)
class SearchDatasetOptions:
    """Arguments for dataset retrieval."""

    dataset_path: str = field(metadata={"help": "RAG dataset JSON path."})
    database_path: str = field(metadata={"help": "SQLite index path."})
    k: int = field(metadata={"help": "Number of sources."})
    save_directory: str = field(metadata={"help": "Output directory."})


@dataclass(frozen=True)
class AnswerOptions:
    """Arguments for one grounded answer."""

    query: str = field(
        metadata={"help": "Question to answer.", "positional": True}
    )
    database_path: str = field(metadata={"help": "SQLite index path."})
    model: str = field(metadata={"help": "Transformers model name."})
    k: int = field(metadata={"help": "Number of sources."})


@dataclass(frozen=True)
class AnswerDatasetOptions:
    """Arguments for answering retrieved questions."""

    student_search_results_path: str = field(
        metadata={"help": "Search results JSON."}
    )
    database_path: str = field(metadata={"help": "SQLite index path."})
    model: str = field(metadata={"help": "Transformers model name."})
    save_directory: str = field(metadata={"help": "Output directory."})
    max_new_tokens: int = field(metadata={"help": "Generation token limit."})


@dataclass(frozen=True)
class EvaluateOptions:
    """Arguments for recall evaluation."""

    student_search_results_path: str = field(
        metadata={"help": "Search results JSON."}
    )
    dataset_path: str = field(metadata={"help": "Ground-truth dataset JSON."})


def index(
    source_path: str,
    project_path: str,
    database_path: str,
    max_chunk_size: int,
) -> None:
    """Index source files into SQLite."""
    import asyncio

    from rag_against_the_machine.indexing.discovery import discover_files
    from rag_against_the_machine.indexing.pipeline import run_pipeline
    from rag_against_the_machine.storage.db import Store

    if max_chunk_size <= 0:
        print("max_chunk_size must be greater than zero")
        return
    store = Store(Path(database_path))
    initialized = store.init()
    if isinstance(initialized, Err):
        initialized.print_diagnostic()
        return
    discovered = discover_files(Path(source_path), Path(project_path))
    if isinstance(discovered, Err):
        discovered.print_diagnostic()
        return
    result = asyncio.run(run_pipeline(discovered.value, max_chunk_size, store))
    if isinstance(result, Err):
        result.print_diagnostic()
        return
    output = result.value
    print(
        f"Indexed {output.files_processed} files; skipped {output.files_skipped}."
    )


def search(query: str, database_path: str, k: int) -> None:
    """Print top-k sources for a query as JSON."""
    from rag_against_the_machine.models.rag import MinimalSearchResults
    from rag_against_the_machine.rag.service import retrieve_hits
    from rag_against_the_machine.storage.db import Store

    result = retrieve_hits(Store(Path(database_path)), query, k)
    if isinstance(result, Err):
        result.print_diagnostic()
        return
    output = MinimalSearchResults(
        question_id="q1",
        question=query,
        retrieved_sources=[_source(hit) for hit in result.value],
    )
    print(output.model_dump_json(indent=2))


def search_dataset(
    dataset_path: str,
    database_path: str,
    k: int,
    save_directory: str,
) -> None:
    """Search every question in a RAG dataset and save JSON."""
    from rag_against_the_machine.models.rag import (
        MinimalSearchResults,
        RagDataset,
        StudentSearchResults,
    )
    from rag_against_the_machine.rag.service import retrieve_hits
    from rag_against_the_machine.storage.db import Store

    dataset = _load_model(Path(dataset_path), RagDataset)
    if isinstance(dataset, Nothing):
        return
    rag_dataset = dataset.value
    from tqdm import tqdm

    results: list[MinimalSearchResults] = []
    store = Store(Path(database_path))
    for question in tqdm(rag_dataset.rag_questions, desc="Searching"):
        hits = retrieve_hits(store, question.question, k)
        if isinstance(hits, Err):
            hits.print_diagnostic()
            return
        results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[_source(hit) for hit in hits.value],
            )
        )
    _save(
        Path(save_directory),
        "search_results.json",
        StudentSearchResults(search_results=results, k=k),
    )


def answer(query: str, database_path: str, model: str, k: int) -> None:
    """Print one grounded answer as JSON."""
    from rag_against_the_machine.generation import create_answer_function
    from rag_against_the_machine.models.rag import (
        MinimalAnswer,
        UnansweredQuestion,
    )
    from rag_against_the_machine.rag import RagService
    from rag_against_the_machine.storage.db import Store

    backend = create_answer_function(model)
    if isinstance(backend, Err):
        backend.print_diagnostic()
        return
    result = RagService(Store(Path(database_path)), backend.value).answer(
        query, k
    )
    if isinstance(result, Err):
        result.print_diagnostic()
        return
    print(result.value.model_dump_json(indent=2))


def answer_dataset(
    student_search_results_path: str,
    database_path: str,
    model: str,
    save_directory: str,
    max_new_tokens: int,
) -> None:
    """Generate answers for search-results JSON and save JSON."""
    from rag_against_the_machine.generation import create_answer_function
    from rag_against_the_machine.models.rag import (
        MinimalAnswer,
        StudentSearchResults,
        StudentSearchResultsAndAnswer,
    )
    from rag_against_the_machine.rag import RagService
    from rag_against_the_machine.storage.db import Store

    source_result = _load_model(
        Path(student_search_results_path), StudentSearchResults
    )
    if isinstance(source_result, Nothing):
        return
    source = source_result.value
    backend = create_answer_function(model)
    if isinstance(backend, Err):
        backend.print_diagnostic()
        return
    service = RagService(Store(Path(database_path)), backend.value)
    from tqdm import tqdm

    answers: list[MinimalAnswer] = []
    for item in tqdm(source.search_results, desc="Answering"):
        result = service.answer(
            item.question,
            source.k,
            max_new_tokens=max_new_tokens,
        )
        if isinstance(result, Err):
            result.print_diagnostic()
            return
        if isinstance(result.value, MinimalAnswer):
            answers.append(result.value)
    _save(
        Path(save_directory),
        "answers.json",
        StudentSearchResultsAndAnswer(search_results=answers, k=source.k),
    )


def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """Report source recall@k against a ground-truth dataset."""
    from rag_against_the_machine.models.rag import (
        RagDataset,
        StudentSearchResults,
    )

    results_result = _load_model(
        Path(student_search_results_path), StudentSearchResults
    )
    dataset_result = _load_model(Path(dataset_path), RagDataset)
    if isinstance(results_result, Nothing) or isinstance(
        dataset_result, Nothing
    ):
        return
    results = results_result.value
    dataset = dataset_result.value
    expected = {
        question.question_id: {
            (
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )
            for source in question.sources
        }
        for question in dataset.rag_questions
        if hasattr(question, "sources")
    }
    total = len(expected)
    hits = 0
    for result in results.search_results:
        wanted = expected.get(result.question_id, set())
        found = {
            (
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )
            for source in result.retrieved_sources
        }
        if wanted and wanted & found:
            hits += 1
    print(
        json.dumps(
            {"recall_at_k": hits / total if total else 0.0, "k": results.k}
        )
    )


def _source(hit: Any) -> Any:
    from rag_against_the_machine.models.rag import MinimalSource

    return MinimalSource(
        file_path=hit.file_path,
        first_character_index=hit.start_character,
        last_character_index=hit.end_character,
    )


def _load_model(path: Path, model_type: type[Any]) -> Option[Any]:
    try:
        return Some(
            model_type.model_validate_json(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, TypeError) as error:
        print(f"Could not read {path}: {error}")
        return Nothing()


def _save(directory: Path, filename: str, model: Any) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _ = (directory / filename).write_text(
            model.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Wrote {directory / filename}")
    except (OSError, TypeError) as error:
        print(f"Could not write output: {error}")
