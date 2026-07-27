"""Asynchronous dispatch of source-file reading and chunking."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from rag_against_the_machine.errors import (
    Diagnostic,
    Err,
    FileProcessingError,
    Ok,
    PipelineError,
    Result,
)
from rag_against_the_machine.indexing.chunker import chunk_source_file
from rag_against_the_machine.indexing.reader import read_source_file
from rag_against_the_machine.models.chunk import Chunk
from rag_against_the_machine.models.source import SourceFile

from rag_against_the_machine.storage.db import (
    ChunkInsert,
    Store,
    Transaction,
)


_CURRENT_CHUNKER_VERSION = 1
_MAX_FILES_PER_TRANSACTION = 25


@dataclass(frozen=True, slots=True)
class ProcessedFile:
    source_file: SourceFile
    size_bytes: int
    modified_at_ns: int
    content_hash: str
    max_chunk_size: int
    chunker_version: int
    indexed_at_ns: int
    chunks: list[Chunk]


@dataclass(slots=True)
class WorkerOutput:
    processed_files: list[ProcessedFile]
    diagnostics: list[Diagnostic]
    files_processed: int
    files_skipped: int


@dataclass(slots=True)
class PipelineOutput:
    diagnostics: list[Diagnostic]
    files_discovered: int
    files_processed: int
    files_skipped: int


async def run_pipeline(
    source_files: list[SourceFile],
    max_chunk_size: int,
    store: Store,
) -> Result[PipelineOutput, PipelineError]:
    if not source_files:
        return Err(PipelineError.EMPTY_INPUT)

    worker_count = min(
        os.cpu_count() or 4,
        len(source_files),
    )

    file_queue: asyncio.Queue[SourceFile | None] = asyncio.Queue()

    for source_file in source_files:
        await file_queue.put(source_file)

    for _ in range(worker_count):
        await file_queue.put(None)

    worker_tasks = [
        asyncio.create_task(worker(file_queue, max_chunk_size))
        for _ in range(worker_count)
    ]

    try:
        worker_outputs = await asyncio.gather(*worker_tasks)
    except asyncio.CancelledError:
        for task in worker_tasks:
            _ = task.cancel()

        _ = await asyncio.gather(
            *worker_tasks,
            return_exceptions=True,
        )

        return Err(PipelineError.CANCELLED)

    except Exception:
        for task in worker_tasks:
            if not task.done():
                _ = task.cancel()

        _ = await asyncio.gather(
            *worker_tasks,
            return_exceptions=True,
        )

        return Err(PipelineError.TASK_FAILED)

    processed_files: list[ProcessedFile] = []
    diagnostics: list[Diagnostic] = []
    files_skipped = 0

    for output in worker_outputs:
        processed_files.extend(output.processed_files)
        diagnostics.extend(output.diagnostics)
        files_skipped += output.files_skipped

    files_persisted = 0

    try:
        # Commit bounded batches so completed work is flushed incrementally.
        # This avoids one transaction per file without making the whole run
        # one large all-or-nothing transaction.
        batch_size = persistence_batch_size(len(processed_files))
        for batch_start in range(0, len(processed_files), batch_size):
            batch = processed_files[batch_start : batch_start + batch_size]
            await asyncio.to_thread(
                persist_processed_files,
                store,
                batch,
            )
            files_persisted += len(batch)

    except asyncio.CancelledError:
        return Err(PipelineError.CANCELLED)

    except Exception:
        return Err(PipelineError.DATABASE_FAILED)

    return Ok(
        PipelineOutput(
            diagnostics=diagnostics,
            files_discovered=len(source_files),
            files_processed=files_persisted,
            files_skipped=files_skipped,
        )
    )


async def worker(
    file_queue: asyncio.Queue[SourceFile | None],
    max_chunk_size: int,
) -> WorkerOutput:
    processed_files: list[ProcessedFile] = []
    diagnostics: list[Diagnostic] = []
    files_processed = 0
    files_skipped = 0

    while True:
        source_file = await file_queue.get()

        try:
            if source_file is None:
                break

            result = await read_and_chunk(
                source_file,
                max_chunk_size,
            )

            if isinstance(result, Ok):
                processed_files.append(result.unwrap())
                files_processed += 1
            else:
                if result.diagnostic is not None:
                    diagnostics.append(result.diagnostic)

                files_skipped += 1
        finally:
            file_queue.task_done()

    return WorkerOutput(
        processed_files=processed_files,
        diagnostics=diagnostics,
        files_processed=files_processed,
        files_skipped=files_skipped,
    )


async def read_and_chunk(
    source_file: SourceFile,
    max_chunk_size: int,
) -> Result[ProcessedFile, FileProcessingError]:
    read_result = await asyncio.to_thread(
        read_source_file,
        source_file,
    )

    if isinstance(read_result, Err):
        read_error = read_result.error

        return Err(
            FileProcessingError.READ_FAILED,
            diagnostic=Diagnostic(
                filename=source_file.stored_path,
                line_num=1,
                line_text=source_file.stored_path,
                col_start=0,
                col_end=len(source_file.stored_path),
                help_msg=(
                    "Reading failed: "
                    f"{str(read_error).rsplit('.', 1)[-1].replace('_', ' ').lower()}."
                ),
            ),
            context_msg="Source-file processing failed",
            namespace="indexing::pipeline",
        )

    text = read_result.unwrap()

    chunk_result = await asyncio.to_thread(
        chunk_source_file,
        source_file,
        text,
        max_chunk_size,
    )

    if isinstance(chunk_result, Err):
        chunk_error = chunk_result.error

        return Err(
            FileProcessingError.CHUNK_FAILED,
            diagnostic=Diagnostic(
                filename=source_file.stored_path,
                line_num=1,
                line_text=source_file.stored_path,
                col_start=0,
                col_end=len(source_file.stored_path),
                help_msg=(
                    "Chunking failed: "
                    f"{str(chunk_error).rsplit('.', 1)[-1].replace('_', ' ').lower()}."
                ),
            ),
            context_msg="Source-file processing failed",
            namespace="indexing::pipeline",
        )

    try:
        stat = await asyncio.to_thread(source_file.absolute_path.stat)

        content_hash = await asyncio.to_thread(
            hash_file,
            source_file.absolute_path,
        )
    except OSError as error:
        return Err(
            FileProcessingError.READ_FAILED,
            diagnostic=Diagnostic(
                filename=source_file.stored_path,
                line_num=1,
                line_text=source_file.stored_path,
                col_start=0,
                col_end=len(source_file.stored_path),
                help_msg=f"Reading file metadata failed: {error}.",
            ),
            context_msg="Source-file processing failed",
            namespace="indexing::pipeline",
        )

    return Ok(
        ProcessedFile(
            source_file=source_file,
            size_bytes=stat.st_size,
            modified_at_ns=stat.st_mtime_ns,
            content_hash=content_hash,
            max_chunk_size=max_chunk_size,
            chunker_version=_CURRENT_CHUNKER_VERSION,
            indexed_at_ns=time.time_ns(),
            chunks=chunk_result.unwrap(),
        )
    )


def persistence_batch_size(file_count: int) -> int:
    """Choose a bounded batch size from the number of files to persist."""
    if file_count <= 0:
        return 1

    transaction_count = max(
        1,
        math.ceil(file_count / _MAX_FILES_PER_TRANSACTION),
    )
    return math.ceil(file_count / transaction_count)


def persist_processed_files(
    store: Store,
    processed_files: list[ProcessedFile],
) -> None:
    """Persist one bounded batch and commit it as a single transaction."""

    def operation(tx: Transaction) -> None:
        for processed in processed_files:
            persist_processed_file(tx, processed)

    store.with_tx(operation)


def persist_processed_file(
    tx: Transaction,
    processed: ProcessedFile,
) -> None:
    """Write one file using an already-open transaction."""
    source_file_id = tx.queries.upsert_source_file(
        path=processed.source_file.stored_path,
        file_type=processed.source_file.file_type,
        size_bytes=processed.size_bytes,
        modified_at_ns=processed.modified_at_ns,
        content_hash=processed.content_hash,
        max_chunk_size=processed.max_chunk_size,
        chunker_version=processed.chunker_version,
        indexed_at_ns=processed.indexed_at_ns,
    ).expect("Source-file upsert did not return an id")

    tx.queries.delete_chunks_for_source(source_file_id)

    created_at_ns = time.time_ns()
    chunk_inserts = [
        ChunkInsert(
            chunk_index=chunk_index,
            text=chunk.text,
            start_character=chunk.first_character_index,
            end_character=chunk.last_character_index,
            created_at_ns=created_at_ns,
        )
        for chunk_index, chunk in enumerate(processed.chunks)
    ]
    tx.queries.insert_chunks(source_file_id, chunk_inserts)


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            hasher.update(block)

    return hasher.hexdigest()
