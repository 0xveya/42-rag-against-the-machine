"""Apply normalized filesystem events to the persisted search index."""

import asyncio
from pathlib import Path

from rag_against_the_machine.errors import (
    Err,
    FileProcessingError,
    Nothing,
    Ok,
    Result,
    Some,
    StorageError,
    catch_bubble,
)
from rag_against_the_machine.fs.events import EventKind, FileEvent
from rag_against_the_machine.indexing.pipeline import (
    persist_processed_file,
    read_and_chunk,
    source_needs_reindex,
)
from rag_against_the_machine.models.source import FileType, SourceFile
from rag_against_the_machine.storage.db import Store, Transaction

EventHandlingResult = Ok[bool] | Err[FileProcessingError] | Err[StorageError]
_SUPPORTED_SUFFIXES = {".py", ".md", ".markdown", ".txt"}


async def handle_event(
    event: FileEvent, max_chunk_size: int, store: Store
) -> EventHandlingResult:
    """Update the persisted index for one normalized filesystem event.

    Returns:
        Success, or the processing/storage error to report to the watch loop.
    """
    if event.is_directory:
        return Ok(False)

    match event:
        case FileEvent(
            kind=EventKind.CREATED
            | EventKind.MODIFIED
            | EventKind.METADATA_CHANGED,
            path=path,
        ):
            if not is_supported(path) or not path.is_file():
                return Ok(False)
            return await handle_reindex(store, path, max_chunk_size)

        case FileEvent(kind=EventKind.DELETED, path=path):
            if not is_supported(path):
                return Ok(False)
            return handle_file_deletion(store, path)

        case FileEvent(
            kind=EventKind.RENAMED,
            path=new_path,
            old_path=Some(value=old_path),
        ):
            deleted = handle_file_deletion(store, old_path)
            if isinstance(deleted, Err):
                return deleted
            if not is_supported(new_path) or not new_path.is_file():
                return deleted
            reindexed = await handle_reindex(store, new_path, max_chunk_size)
            if isinstance(reindexed, Err):
                return reindexed
            return Ok(deleted.value or reindexed.value)

        case FileEvent(kind=EventKind.RENAMED, old_path=Nothing()):
            return Ok(False)

    return Ok(False)


def is_supported(path: Path) -> bool:
    """Return whether initial discovery would include this file."""
    return path.suffix.lower() in _SUPPORTED_SUFFIXES


def handle_ext(suffix: str) -> FileType:
    """Determine the file type from a path suffix.

    Returns:
        The file type used by the reader and chunker.
    """
    match suffix.lower():
        case ".py":
            return "python"
        case ".md" | ".markdown":
            return "markdown"
        case _:
            return "text"


def stored_path(path: Path) -> str:
    """Return the same project-relative key used during initial discovery.

    Returns:
        A normalized database path key.
    """
    absolute_path = path.resolve()
    try:
        return absolute_path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return absolute_path.as_posix()


async def handle_reindex(
    store: Store, path: Path, max_chunk_size: int
) -> EventHandlingResult:
    """Read, chunk, and upsert one changed file.

    Returns:
        Success, or the processing/storage error encountered.
    """
    source_file = SourceFile(
        absolute_path=path,
        stored_path=stored_path(path),
        file_type=handle_ext(path.suffix),
    )
    state_result = await asyncio.to_thread(
        source_needs_reindex,
        store,
        source_file,
        max_chunk_size,
    )
    if isinstance(state_result, Err):
        return state_result
    if not state_result.value:
        return Ok(False)

    processed = await read_and_chunk(source_file, max_chunk_size)
    if isinstance(processed, Err):
        return processed

    @catch_bubble
    def operation(tx: Transaction) -> Result[None, StorageError]:
        _ = persist_processed_file(tx, processed.value).q
        return Ok(None)

    persisted = store.with_tx(operation)
    if isinstance(persisted, Err):
        return persisted
    return Ok(True)


def handle_file_deletion(
    store: Store, path: Path
) -> Result[bool, StorageError]:
    """Delete a source row and its cascade-owned chunks if it exists.

    Returns:
        Success, or the storage error encountered.
    """

    @catch_bubble
    def operation(tx: Transaction) -> Result[bool, StorageError]:
        source_id = tx.queries.get_id_for_source_file(stored_path(path)).q
        match source_id:
            case Some(value=value):
                _ = tx.queries.delete_source_file(value).q
                return Ok(True)
            case Nothing():
                return Ok(False)

    return store.with_tx(operation)
