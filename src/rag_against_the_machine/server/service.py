"""Long-lived retrieval, watcher, and streaming service for FastAPI."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from pathlib import Path

import structlog

from rag_against_the_machine.errors import (
    Err,
    GenerationError,
    Nothing,
    Ok,
    Option,
    Result,
    Some,
    StorageError,
    WebError,
)
from rag_against_the_machine.fs.watcher_async import AsyncWatcher
from rag_against_the_machine.generation import (
    AnswerFunctions,
    create_answer_functions,
    generate_answer_stream,
)
from rag_against_the_machine.indexing.discovery import discover_files
from rag_against_the_machine.indexing.fsevent_handler import handle_event
from rag_against_the_machine.indexing.pipeline import run_pipeline
from rag_against_the_machine.rag.service import retrieve_hits
from rag_against_the_machine.server.models import (
    ReindexResponse,
    RepositorySummary,
    RetrievedSource,
)
from rag_against_the_machine.storage.db import SearchHit, Store, Transaction

_NO_ANSWERS: Option[AnswerFunctions] = Nothing()


def _next_stream_result(
    stream: Iterator[Result[str, GenerationError]],
) -> Option[Result[str, GenerationError]]:
    """Advance a synchronous generation stream without leaking StopIteration."""
    try:
        return Some(next(stream))
    except StopIteration:
        return Nothing()


class WebRagService:
    """Own resources that should survive across HTTP and WebSocket requests."""

    def __init__(
        self,
        raw_directory: Path = Path("data/raw"),
        database_path: Path = Path("data/processed/index.db"),
        max_chunk_size: int = 2000,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 256,
        answer_functions: Option[AnswerFunctions] = _NO_ANSWERS,
    ) -> None:
        """Configure the service without starting background resources."""
        self.raw_directory = raw_directory
        self.store = Store(database_path)
        self.max_chunk_size = max_chunk_size
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.log = structlog.get_logger(__name__)
        self._answer_functions = answer_functions
        self._watcher: Option[AsyncWatcher] = Nothing()
        self._watch_task: Option[asyncio.Task[None]] = Nothing()
        self._index_lock = asyncio.Lock()

    async def start(self) -> Result[None, WebError]:
        """Initialize storage, generation, and the filesystem watcher."""
        initialized = await asyncio.to_thread(self.store.init)
        if isinstance(initialized, Err):
            return Err(
                WebError.STORAGE_FAILED,
                context_msg=initialized.context_msg or "Storage setup failed",
                namespace="server::service",
            )

        startup_result: Result[None, WebError] = Ok(None)
        if isinstance(self._answer_functions, Nothing):
            loaded = await asyncio.to_thread(
                create_answer_functions, self.model_name
            )
            if isinstance(loaded, Err):
                startup_result = Err(
                    WebError.MODEL_LOAD_FAILED,
                    context_msg=loaded.context_msg or "Model loading failed",
                    namespace="server::service",
                )
            else:
                self._answer_functions = Some(loaded.value)
                await self.log.ainfo(
                    "generation_model_loaded", model=self.model_name
                )

        if not self.raw_directory.is_dir():
            await self.log.awarning(
                "watcher_not_started", path=str(self.raw_directory)
            )
            return startup_result
        opened = AsyncWatcher.open(self.raw_directory, recursive=True)
        if isinstance(opened, Err):
            await self.log.awarning(
                "watcher_not_started", reason=opened.context_msg
            )
            return startup_result
        watcher = opened.value
        self._watcher = Some(watcher)
        started = await watcher.start()
        if isinstance(started, Err):
            watcher.close()
            self._watcher = Nothing()
            await self.log.awarning(
                "watcher_not_started", reason=started.context_msg
            )
            return startup_result
        self._watch_task = Some(
            asyncio.create_task(
                self._watch_changes(), name="rag-index-watcher"
            )
        )
        await self.log.ainfo("watcher_started", path=str(self.raw_directory))
        return startup_result

    async def close(self) -> None:
        """Stop background work and release the inotify descriptor."""
        if isinstance(self._watcher, Some):
            self._watcher.value.close()
        if isinstance(self._watch_task, Some):
            self._watch_task.value.cancel()
            with suppress(asyncio.CancelledError):
                await self._watch_task.value
        self._watcher = Nothing()
        self._watch_task = Nothing()
        await self.log.ainfo("web_rag_service_closed")

    def list_repositories(self) -> list[RepositorySummary]:
        """List repository directories available under data/raw."""
        if not self.raw_directory.is_dir():
            return []
        return [
            RepositorySummary(
                repository_id=path.name,
                name=path.name,
                path=path.as_posix(),
            )
            for path in sorted(self.raw_directory.iterdir())
            if path.is_dir()
        ]

    def _repository_path(self, repository: str) -> Result[Path, WebError]:
        """Resolve a repository ID without permitting path traversal."""
        for item in self.list_repositories():
            if item.repository_id == repository:
                return Ok(Path(item.path))
        return Err(
            WebError.INVALID_REPOSITORY,
            context_msg=f"Unknown repository: {repository}",
            namespace="server::service",
        )

    def _stored_prefix(self, repository_path: Path) -> Result[str, WebError]:
        """Return the project-relative prefix used by the SQLite index."""
        try:
            relative = repository_path.resolve().relative_to(
                Path.cwd().resolve()
            )
        except ValueError:
            return Err(
                WebError.INVALID_REPOSITORY,
                context_msg="The raw directory must be inside the project directory",
                namespace="server::service",
            )
        return Ok(relative.as_posix().rstrip("/") + "/")

    async def reindex(
        self, repository: str
    ) -> Result[ReindexResponse, WebError]:
        """Force-rebuild one selected repository and remove stale records."""
        repository_result = self._repository_path(repository)
        if isinstance(repository_result, Err):
            return repository_result
        repository_path = repository_result.value
        discovered = await asyncio.to_thread(
            discover_files, repository_path, Path.cwd()
        )
        if isinstance(discovered, Err):
            return Err(
                WebError.DISCOVERY_FAILED,
                context_msg=discovered.context_msg or "Discovery failed",
                namespace="server::service",
            )

        async with self._index_lock:
            if discovered.value:
                indexed = await run_pipeline(
                    discovered.value,
                    self.max_chunk_size,
                    self.store,
                    force=True,
                )
                if isinstance(indexed, Err):
                    return Err(
                        WebError.INDEX_FAILED,
                        context_msg=indexed.context_msg or "Reindex failed",
                        namespace="server::service",
                    )
                output = indexed.value
            else:
                output = None

            live_paths = {source.stored_path for source in discovered.value}
            prefix_result = self._stored_prefix(repository_path)
            if isinstance(prefix_result, Err):
                return prefix_result
            prefix = prefix_result.value

            def remove_stale(
                tx: Transaction,
            ) -> Result[None, StorageError]:
                stored = tx.queries.get_all_source_files()
                if isinstance(stored, Err):
                    return stored
                for path, record in stored.value.items():
                    if path.startswith(prefix) and path not in live_paths:
                        deleted = tx.queries.delete_source_file(record.id)
                        if isinstance(deleted, Err):
                            return deleted
                return Ok(None)

            cleaned = await asyncio.to_thread(self.store.with_tx, remove_stale)
            if isinstance(cleaned, Err):
                return Err(
                    WebError.INDEX_CLEANUP_FAILED,
                    context_msg=cleaned.context_msg or "Index cleanup failed",
                    namespace="server::service",
                )

        return Ok(
            ReindexResponse(
                repository_id=repository,
                files_discovered=len(discovered.value),
                files_processed=output.files_processed if output else 0,
                files_skipped=output.files_skipped if output else 0,
            )
        )

    async def retrieve(
        self, repository: str, question: str, k: int
    ) -> Result[list[RetrievedSource], WebError]:
        """Retrieve source locations from the selected repository."""
        repository_result = self._repository_path(repository)
        if isinstance(repository_result, Err):
            return repository_result
        repository_path = repository_result.value
        result = await asyncio.to_thread(
            retrieve_hits, self.store, question, max(k * 10, k)
        )
        if isinstance(result, Err):
            return Err(
                WebError.RETRIEVAL_FAILED,
                context_msg=result.context_msg or "Retrieval failed",
                namespace="server::service",
            )
        prefix_result = self._stored_prefix(repository_path)
        if isinstance(prefix_result, Err):
            return prefix_result
        prefix = prefix_result.value
        sources = [
            RetrievedSource(
                file_path=hit.file_path,
                first_character_index=hit.start_character,
                last_character_index=hit.end_character,
                text=hit.text,
            )
            for hit in result.value
            if hit.file_path.startswith(prefix)
        ][:k]
        await self.log.ainfo(
            "sources_retrieved",
            repository=repository,
            requested=k,
            returned=len(sources),
        )
        return Ok(sources)

    async def stream_answer(
        self, question: str, sources: list[RetrievedSource]
    ) -> AsyncIterator[Result[str, WebError]]:
        """Stream model output without blocking the FastAPI event loop."""
        if isinstance(self._answer_functions, Nothing):
            yield Err(
                WebError.MODEL_LOAD_FAILED,
                context_msg="The generation model is not available",
                namespace="server::service",
            )
            return

        hits = [
            SearchHit(
                chunk_id=index,
                file_path=source.file_path,
                start_character=source.first_character_index,
                end_character=source.last_character_index,
                text=source.text,
                score=0.0,
            )
            for index, source in enumerate(sources)
        ]
        stream = generate_answer_stream(
            self._answer_functions.value.stream,
            question,
            hits,
            self.max_new_tokens,
        )
        await self.log.ainfo(
            "generation_started",
            question_length=len(question),
            source_count=len(sources),
        )
        while True:
            item = await asyncio.to_thread(_next_stream_result, stream)
            if isinstance(item, Nothing):
                break
            result = item.value
            if isinstance(result, Err):
                yield Err(
                    WebError.GENERATION_FAILED,
                    context_msg=result.context_msg or "Generation failed",
                    namespace="server::service",
                )
                return
            yield Ok(result.value)
        await self.log.ainfo("generation_finished")

    async def _watch_changes(self) -> None:
        """Apply source changes while the FastAPI lifespan remains active."""
        watcher_option = self._watcher
        if isinstance(watcher_option, Nothing):
            return
        watcher = watcher_option.value
        async for event_result in watcher:
            if isinstance(event_result, Err):
                await self.log.aerror(
                    "watcher_event_failed", reason=event_result.context_msg
                )
                continue
            event = event_result.value
            async with self._index_lock:
                handled = await handle_event(
                    event, self.max_chunk_size, self.store
                )
            if isinstance(handled, Err):
                await self.log.aerror(
                    "index_update_failed",
                    path=str(event.path),
                    reason=handled.context_msg,
                )
            elif handled.value:
                await self.log.ainfo("index_updated", path=str(event.path))
