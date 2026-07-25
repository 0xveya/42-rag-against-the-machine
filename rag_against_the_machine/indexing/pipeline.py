"""Asynchronous dispatch of source-file reading and chunking."""

import asyncio
import os
from dataclasses import dataclass

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


@dataclass(slots=True)
class PipelineOutput:
    chunks: list[Chunk]
    diagnostics: list[Diagnostic]
    files_discovered: int
    files_processed: int
    files_skipped: int


@dataclass(slots=True)
class WorkerOutput:
    chunks: list[Chunk]
    diagnostics: list[Diagnostic]
    files_processed: int
    files_skipped: int


async def run_pipeline(
    source_files: list[SourceFile],
    max_chunk_size: int,
) -> Result[PipelineOutput, PipelineError]:
    if not source_files:
        return Err(PipelineError.EMPTY_INPUT)
    worker_count = min(os.cpu_count() or 4, len(source_files))
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
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        return Err(PipelineError.CANCELLED)
    except Exception:
        for task in worker_tasks:
            if not task.done():
                _ = task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        return Err(PipelineError.TASK_FAILED)

    chunks: list[Chunk] = []
    diagnostics: list[Diagnostic] = []
    files_processed = 0
    files_skipped = 0

    for output in worker_outputs:
        chunks.extend(output.chunks)
        diagnostics.extend(output.diagnostics)
        files_processed += output.files_processed
        files_skipped += output.files_skipped

    return Ok(
        PipelineOutput(
            chunks=chunks,
            diagnostics=diagnostics,
            files_discovered=len(source_files),
            files_processed=files_processed,
            files_skipped=files_skipped,
        )
    )


async def worker(
    file_queue: asyncio.Queue[SourceFile | None], max_chunk_size: int
) -> WorkerOutput:
    chunks: list[Chunk] = []
    diagnostics: list[Diagnostic] = []
    files_processed = 0
    files_skipped = 0
    while True:
        source_file = await file_queue.get()

        try:
            if source_file is None:
                break

            result = await read_and_chunk(source_file, max_chunk_size)
            if isinstance(result, Ok):
                chunks.extend(result.unwrap())
                files_processed += 1
            else:
                if result.diagnostic is not None:
                    diagnostics.append(result.diagnostic)
                files_skipped += 1
        finally:
            file_queue.task_done()

    return WorkerOutput(
        chunks=chunks,
        diagnostics=diagnostics,
        files_processed=files_processed,
        files_skipped=files_skipped,
    )


async def read_and_chunk(
    source_file: SourceFile,
    max_chunk_size: int,
) -> Result[list[Chunk], FileProcessingError]:
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
                help_msg=f"Reading failed: {read_error.name.replace('_', ' ').lower()}.",
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
                help_msg=f"Chunking failed: {chunk_error.name.replace('_', ' ').lower()}.",
            ),
            context_msg="Source-file processing failed",
            namespace="indexing::pipeline",
        )

    return Ok(chunk_result.unwrap())
