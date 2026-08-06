"""Language-independent chunk construction, splitting, and validation."""

from rag_against_the_machine.errors import ChunkingError, Err, Ok, Result
from rag_against_the_machine.models.chunk import Chunk
from rag_against_the_machine.models.source import SourceFile


def make_chunk(
    source_file: SourceFile, text: str, start: int, end: int
) -> Chunk:
    """Construct a chunk covering an exact character range."""
    if start < 0 or end < start or end > len(text):
        raise ValueError("Invalid source range")
    return Chunk(
        source_file.stored_path,
        start,
        end,
        text[start:end],
        source_file.file_type,
    )


def find_preferred_split(text: str, start: int, maximum_end: int) -> int:
    """Find a paragraph, line, or word boundary before the hard limit."""
    for separator in ("\n\n", "\n", " "):
        position = text.rfind(separator, start, maximum_end)
        if position > start:
            return position + len(separator)
    return maximum_end


def split_range(
    source_file: SourceFile,
    text: str,
    start: int,
    end: int,
    max_chunk_size: int,
) -> list[Chunk]:
    """Split a range without exceeding the configured character limit."""
    if max_chunk_size <= 0 or start < 0 or end < start or end > len(text):
        raise ValueError("Invalid chunk size or source range")
    chunks: list[Chunk] = []
    current = start
    while current < end:
        maximum_end = min(current + max_chunk_size, end)
        chunk_end = (
            end
            if maximum_end == end
            else find_preferred_split(text, current, maximum_end)
        )
        if chunk_end <= current:
            chunk_end = maximum_end
        chunks.append(make_chunk(source_file, text, current, chunk_end))
        current = chunk_end
    return chunks


def append_range_as_chunks(
    chunks: list[Chunk],
    source_file: SourceFile,
    text: str,
    start: int,
    end: int,
    max_chunk_size: int,
) -> None:
    """Append a range as one chunk or as bounded subchunks."""
    if start >= end:
        return
    if end - start <= max_chunk_size:
        chunks.append(make_chunk(source_file, text, start, end))
    else:
        chunks.extend(
            split_range(source_file, text, start, end, max_chunk_size)
        )


def include_range_gaps(
    ranges: list[tuple[int, int]], text_length: int
) -> list[tuple[int, int]]:
    """Add nonempty gaps so the returned ranges cover the whole source."""
    complete: list[tuple[int, int]] = []
    current = 0
    for start, end in sorted(ranges):
        if start < current:
            raise ValueError("Structural ranges overlap")
        if current < start:
            complete.append((current, start))
        if start < end:
            complete.append((start, end))
        current = max(current, end)
    if current < text_length:
        complete.append((current, text_length))
    return complete


def chunk_plain_text(
    source_file: SourceFile, text: str, max_chunk_size: int
) -> Result[list[Chunk], ChunkingError]:
    """Chunk text using generic human-friendly boundaries."""
    if max_chunk_size <= 0:
        return Err(ChunkingError.INVALID_MAX_CHUNK_SIZE)
    return Ok(split_range(source_file, text, 0, len(text), max_chunk_size))


def validate_chunks(
    text: str, chunks: list[Chunk], max_chunk_size: int
) -> None:
    """Ensure chunks are bounded, ordered, lossless, and source-backed."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    previous_end = 0
    for index, chunk in enumerate(chunks):
        start, end = chunk.first_character_index, chunk.last_character_index
        if (
            start != previous_end
            or start < 0
            or end <= start
            or end > len(text)
        ):
            raise ValueError(
                f"Chunk {index} has invalid or non-contiguous range"
            )
        if end - start > max_chunk_size:
            raise ValueError(f"Chunk {index} exceeds max_chunk_size")
        if chunk.text != text[start:end]:
            raise ValueError(f"Chunk {index} text does not match source range")
        previous_end = end
    if chunks and previous_end != len(text):
        raise ValueError("Chunks do not cover the complete source")
    if not chunks and text:
        raise ValueError("Non-empty source produced no chunks")
