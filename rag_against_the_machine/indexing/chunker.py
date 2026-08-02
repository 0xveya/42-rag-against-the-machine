"""Implementation of source-file chunking."""

import ast
import re
from dataclasses import dataclass
from enum import Enum, auto

from rag_against_the_machine.errors import ChunkingError, Err, Ok, Result
from rag_against_the_machine.models.chunk import Chunk
from rag_against_the_machine.models.source import SourceFile


class MarkdownBlockType(Enum):
    """Identify structural Markdown block categories."""

    HEADING = auto()
    PARAGRAPH = auto()
    CODE_FENCE = auto()


@dataclass(frozen=True)
class MarkdownBlock:
    """A structural block inside a Markdown document."""

    start: int
    end: int
    block_type: MarkdownBlockType


_HEADING_PATTERN = re.compile(r"^#{1,6}\s+")

_PYTHON_STRUCTURAL_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def chunk_source_file(
    source_file: SourceFile,
    text: str,
    max_chunk_size: int,
) -> Result[list[Chunk], ChunkingError]:
    """Route a source file to the appropriate chunking strategy.

    Returns:
        Generated chunks, or a categorized chunking error.
    """
    if max_chunk_size <= 0:
        return Err(ChunkingError.INVALID_MAX_CHUNK_SIZE)

    if source_file.file_type == "python":
        return chunk_python(
            source_file=source_file,
            text=text,
            max_chunk_size=max_chunk_size,
        )

    if source_file.file_type == "markdown":
        return chunk_markdown(
            source_file=source_file,
            text=text,
            max_chunk_size=max_chunk_size,
        )

    return chunk_plain_text(
        source_file=source_file,
        text=text,
        max_chunk_size=max_chunk_size,
    )


def make_chunk(
    source_file: SourceFile,
    text: str,
    start: int,
    end: int,
) -> Chunk:
    """Construct a chunk while preserving exact source offsets.

    Returns:
        A chunk covering the requested source range.

    Raises:
        ValueError: If the source range is invalid.
    """
    if start < 0:
        raise ValueError("Chunk start cannot be negative")

    if end < start:
        raise ValueError("Chunk end cannot be before chunk start")

    if end > len(text):
        raise ValueError("Chunk end cannot exceed source length")

    return Chunk(
        file_path=source_file.stored_path,
        first_character_index=start,
        last_character_index=end,
        text=text[start:end],
        file_type=source_file.file_type,
    )


def find_preferred_split(
    text: str,
    start: int,
    maximum_end: int,
) -> int:
    """Find a pleasant split point before maximum_end.

    Preference order:

    1. Blank line
    2. Newline
    3. Space
    4. Hard maximum

    Returns:
        The preferred exclusive split position.
    """
    separators = (
        "\n\n",
        "\n",
        " ",
    )

    for separator in separators:
        separator_position = text.rfind(
            separator,
            start,
            maximum_end,
        )

        if separator_position <= start:
            continue

        return separator_position + len(separator)

    return maximum_end


def split_range(
    source_file: SourceFile,
    text: str,
    start: int,
    end: int,
    max_chunk_size: int,
) -> list[Chunk]:
    """Split one source range into chunks.

    This is the fallback used when a Markdown block, Python definition,
    or plain-text file is larger than max_chunk_size.

    Returns:
        Chunks that exactly cover the requested range.

    Raises:
        ValueError: If the maximum size or source range is invalid.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")

    if start < 0 or end < start or end > len(text):
        raise ValueError("Invalid source range")

    chunks: list[Chunk] = []
    current = start

    while current < end:
        maximum_end = min(
            current + max_chunk_size,
            end,
        )

        if maximum_end == end:
            chunk_end = end
        else:
            chunk_end = find_preferred_split(
                text=text,
                start=current,
                maximum_end=maximum_end,
            )

        if chunk_end <= current:
            chunk_end = maximum_end

        chunks.append(
            make_chunk(
                source_file=source_file,
                text=text,
                start=current,
                end=chunk_end,
            )
        )

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
    """Append one range, splitting it only when necessary."""
    if start >= end:
        return

    if end - start <= max_chunk_size:
        chunks.append(
            make_chunk(
                source_file=source_file,
                text=text,
                start=start,
                end=end,
            )
        )
        return

    chunks.extend(
        split_range(
            source_file=source_file,
            text=text,
            start=start,
            end=end,
            max_chunk_size=max_chunk_size,
        )
    )


def chunk_plain_text(
    source_file: SourceFile,
    text: str,
    max_chunk_size: int,
) -> Result[list[Chunk], ChunkingError]:
    """Chunk plain text using paragraph, line, and space boundaries.

    Returns:
        Generated chunks, or an invalid-size error.
    """
    if max_chunk_size <= 0:
        return Err(ChunkingError.INVALID_MAX_CHUNK_SIZE)

    return Ok(
        split_range(
            source_file=source_file,
            text=text,
            start=0,
            end=len(text),
            max_chunk_size=max_chunk_size,
        )
    )


def find_markdown_blocks(text: str) -> list[MarkdownBlock]:
    """Find Markdown headings, paragraphs, and fenced code blocks.

    Fenced code blocks are returned as one atomic block whenever possible.

    Returns:
        Ordered structural blocks covering the Markdown text.
    """
    lines = text.splitlines(keepends=True)

    blocks: list[MarkdownBlock] = []
    position = 0
    block_start = 0

    inside_fence = False
    fence_marker: str | None = None

    def append_block(
        start: int,
        end: int,
        block_type: MarkdownBlockType,
    ) -> None:
        if start >= end:
            return

        blocks.append(
            MarkdownBlock(
                start=start,
                end=end,
                block_type=block_type,
            )
        )

    for line in lines:
        stripped = line.lstrip()
        line_end = position + len(line)

        opening_marker: str | None = None

        if stripped.startswith("```"):
            opening_marker = "```"
        elif stripped.startswith("~~~"):
            opening_marker = "~~~"

        if not inside_fence and opening_marker is not None:
            append_block(
                start=block_start,
                end=position,
                block_type=MarkdownBlockType.PARAGRAPH,
            )

            inside_fence = True
            fence_marker = opening_marker
            block_start = position

        elif (
            inside_fence
            and fence_marker is not None
            and stripped.startswith(fence_marker)
        ):
            append_block(
                start=block_start,
                end=line_end,
                block_type=MarkdownBlockType.CODE_FENCE,
            )

            inside_fence = False
            fence_marker = None
            block_start = line_end

        elif not inside_fence and _HEADING_PATTERN.match(stripped) is not None:
            append_block(
                start=block_start,
                end=position,
                block_type=MarkdownBlockType.PARAGRAPH,
            )

            append_block(
                start=position,
                end=line_end,
                block_type=MarkdownBlockType.HEADING,
            )

            block_start = line_end

        elif not inside_fence and stripped.strip() == "":
            append_block(
                start=block_start,
                end=line_end,
                block_type=MarkdownBlockType.PARAGRAPH,
            )

            block_start = line_end

        position = line_end

    final_block_type = (
        MarkdownBlockType.CODE_FENCE
        if inside_fence
        else MarkdownBlockType.PARAGRAPH
    )

    append_block(
        start=block_start,
        end=len(text),
        block_type=final_block_type,
    )

    return blocks


def flush_pending_range(
    chunks: list[Chunk],
    source_file: SourceFile,
    text: str,
    start: int | None,
    end: int | None,
    max_chunk_size: int,
) -> None:
    """Append a pending combined range when one exists."""
    if start is None or end is None:
        return

    append_range_as_chunks(
        chunks=chunks,
        source_file=source_file,
        text=text,
        start=start,
        end=end,
        max_chunk_size=max_chunk_size,
    )


def chunk_markdown(
    source_file: SourceFile,
    text: str,
    max_chunk_size: int,
) -> Result[list[Chunk], ChunkingError]:
    """Chunk Markdown around structural blocks.

    Code fences remain intact unless a single fence exceeds max_chunk_size.

    Returns:
        Generated chunks, or an invalid-size error.
    """
    if max_chunk_size <= 0:
        return Err(ChunkingError.INVALID_MAX_CHUNK_SIZE)

    blocks = find_markdown_blocks(text)
    chunks: list[Chunk] = []

    pending_start: int | None = None
    pending_end: int | None = None

    for block in blocks:
        block_size = block.end - block.start

        if block_size > max_chunk_size:
            flush_pending_range(
                chunks=chunks,
                source_file=source_file,
                text=text,
                start=pending_start,
                end=pending_end,
                max_chunk_size=max_chunk_size,
            )

            pending_start = None
            pending_end = None

            append_range_as_chunks(
                chunks=chunks,
                source_file=source_file,
                text=text,
                start=block.start,
                end=block.end,
                max_chunk_size=max_chunk_size,
            )

            continue

        if pending_start is None:
            pending_start = block.start
            pending_end = block.end
            continue

        proposed_size = block.end - pending_start

        if proposed_size <= max_chunk_size:
            pending_end = block.end
            continue

        flush_pending_range(
            chunks=chunks,
            source_file=source_file,
            text=text,
            start=pending_start,
            end=pending_end,
            max_chunk_size=max_chunk_size,
        )

        pending_start = block.start
        pending_end = block.end

    flush_pending_range(
        chunks=chunks,
        source_file=source_file,
        text=text,
        start=pending_start,
        end=pending_end,
        max_chunk_size=max_chunk_size,
    )

    return Ok(chunks)


def build_line_offsets(text: str) -> list[int]:
    """Map zero-based line indexes to absolute character offsets.

    offsets[0] is the beginning of line 1.
    offsets[1] is the beginning of line 2.

    Returns:
        Absolute offsets for every line boundary.
    """
    offsets = [0]

    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    return offsets


def get_python_node_start_line(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> int:
    """Include decorators in a Python definition's range.

    Returns:
        The one-based first line of the decorated definition.
    """
    start_line = node.lineno

    if node.decorator_list:
        start_line = min(decorator.lineno for decorator in node.decorator_list)

    return start_line


def find_python_structural_ranges(
    text: str,
) -> list[tuple[int, int]]:
    """Find top-level Python functions and classes.

    The returned end indexes are exclusive.

    Returns:
        Character ranges for top-level definitions.
    """
    tree = ast.parse(text)
    line_offsets = build_line_offsets(text)

    ranges: list[tuple[int, int]] = []

    for node in tree.body:
        if not isinstance(node, _PYTHON_STRUCTURAL_NODES):
            continue

        if node.end_lineno is None:
            continue

        start_line = get_python_node_start_line(node)

        start = line_offsets[start_line - 1]
        end = line_offsets[node.end_lineno]

        ranges.append((start, end))

    return ranges


def include_range_gaps(
    ranges: list[tuple[int, int]],
    text_length: int,
) -> list[tuple[int, int]]:
    """Add ranges for imports, constants, comments, and module-level code.

    Example:
        [(100, 200), (300, 400)]

    becomes:

        [(0, 100), (100, 200), (200, 300), (300, 400), ...]

    Returns:
        Sorted, nonempty ranges covering gaps and definitions.
    """
    complete_ranges: list[tuple[int, int]] = []
    current = 0

    for start, end in sorted(ranges):
        if current < start:
            complete_ranges.append((current, start))

        complete_ranges.append((start, end))
        current = max(current, end)

    if current < text_length:
        complete_ranges.append((current, text_length))

    return [(start, end) for start, end in complete_ranges if start < end]


def chunk_python(
    source_file: SourceFile,
    text: str,
    max_chunk_size: int,
) -> Result[list[Chunk], ChunkingError]:
    """Chunk Python using top-level function and class boundaries.

    Invalid Python falls back to plain-text chunking. Oversized definitions
    are split using blank lines, then line endings, then spaces.

    Returns:
        Generated chunks, or an invalid-size error.
    """
    if max_chunk_size <= 0:
        return Err(ChunkingError.INVALID_MAX_CHUNK_SIZE)

    try:
        structural_ranges = find_python_structural_ranges(text)
    except SyntaxError:
        return chunk_plain_text(
            source_file=source_file,
            text=text,
            max_chunk_size=max_chunk_size,
        )

    ranges = include_range_gaps(
        ranges=structural_ranges,
        text_length=len(text),
    )

    chunks: list[Chunk] = []

    for start, end in ranges:
        append_range_as_chunks(
            chunks=chunks,
            source_file=source_file,
            text=text,
            start=start,
            end=end,
            max_chunk_size=max_chunk_size,
        )

    return Ok(chunks)


def validate_chunks(
    text: str,
    chunks: list[Chunk],
    max_chunk_size: int,
) -> None:
    """Validate chunk bounds, coverage, size, and stored text.

    Raises:
        ValueError: If any chunk invariant is violated.
    """
    if max_chunk_size <= 0:
        raise ValueError(
            f"max_chunk_size must be positive, got {max_chunk_size}"
        )

    previous_end = 0

    for index, chunk in enumerate(chunks):
        start = chunk.first_character_index
        end = chunk.last_character_index
        chunk_size = end - start

        if start != previous_end:
            raise ValueError(
                f"Chunk {index}: expected start {previous_end}, got {start}"
            )

        if start < 0:
            raise ValueError(
                f"Chunk {index}: start cannot be negative: {start}"
            )

        if end < start:
            raise ValueError(
                f"Chunk {index}: end {end} is before start {start}"
            )

        if end > len(text):
            raise ValueError(
                f"Chunk {index}: end {end} exceeds source length {len(text)}"
            )

        if chunk_size <= 0:
            raise ValueError(f"Chunk {index}: invalid size {chunk_size}")

        if chunk_size > max_chunk_size:
            raise ValueError(
                f"Chunk {index}: size {chunk_size} exceeds max_chunk_size={max_chunk_size}"
            )

        expected_text = text[start:end]

        if chunk.text != expected_text:
            raise ValueError(
                f"Chunk {index}: text does not match source range "
                f"[{start}:{end}]\n"
                f"Expected: {expected_text!r}\n"
                f"Actual:   {chunk.text!r}"
            )

        if len(chunk.text) != chunk_size:
            raise ValueError(
                f"Chunk {index}: stored text length "
                f"{len(chunk.text)} does not match range size "
                f"{chunk_size}"
            )

        previous_end = end

    if chunks and previous_end != len(text):
        raise ValueError(
            f"Chunks stop at character {previous_end}, but source length is {len(text)}"
        )

    if not chunks and text:
        raise ValueError("Non-empty source produced no chunks")
