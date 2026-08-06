"""Markdown-aware chunking."""

import re
from dataclasses import dataclass
from enum import Enum, auto

from rag_against_the_machine.errors import ChunkingError, Err, Ok, Result
from rag_against_the_machine.indexing.chunk_helpers import (
    append_range_as_chunks,
)
from rag_against_the_machine.models.chunk import Chunk
from rag_against_the_machine.models.source import SourceFile


class MarkdownBlockType(Enum):
    """Identify Markdown block categories."""

    HEADING = auto()
    PARAGRAPH = auto()
    CODE_FENCE = auto()


@dataclass(frozen=True)
class MarkdownBlock:
    """A structural Markdown range."""

    start: int
    end: int
    block_type: MarkdownBlockType


_HEADING_PATTERN = re.compile(r"^#{1,6}\s+")


def find_markdown_blocks(text: str) -> list[MarkdownBlock]:
    """Find headings, paragraphs, and fenced code blocks."""
    lines = text.splitlines(keepends=True)
    blocks: list[MarkdownBlock] = []
    position = 0
    block_start = 0
    inside_fence = False
    fence_marker: str | None = None

    def append(start: int, end: int, kind: MarkdownBlockType) -> None:
        if start < end:
            blocks.append(MarkdownBlock(start, end, kind))

    for line in lines:
        stripped = line.lstrip()
        line_end = position + len(line)
        marker = (
            "```"
            if stripped.startswith("```")
            else ("~~~" if stripped.startswith("~~~") else None)
        )
        if not inside_fence and marker:
            append(block_start, position, MarkdownBlockType.PARAGRAPH)
            inside_fence, fence_marker, block_start = True, marker, position
        elif (
            inside_fence and fence_marker and stripped.startswith(fence_marker)
        ):
            append(block_start, line_end, MarkdownBlockType.CODE_FENCE)
            inside_fence, fence_marker, block_start = False, None, line_end
        elif not inside_fence and _HEADING_PATTERN.match(stripped):
            append(block_start, position, MarkdownBlockType.PARAGRAPH)
            append(position, line_end, MarkdownBlockType.HEADING)
            block_start = line_end
        elif not inside_fence and not stripped.strip():
            append(block_start, line_end, MarkdownBlockType.PARAGRAPH)
            block_start = line_end
        position = line_end

    append(
        block_start,
        len(text),
        MarkdownBlockType.CODE_FENCE
        if inside_fence
        else MarkdownBlockType.PARAGRAPH,
    )
    return blocks


def chunk_markdown(
    source_file: SourceFile, text: str, max_chunk_size: int
) -> Result[list[Chunk], ChunkingError]:
    """Chunk Markdown around structural blocks."""
    if max_chunk_size <= 0:
        return Err(ChunkingError.INVALID_MAX_CHUNK_SIZE)
    chunks: list[Chunk] = []
    pending_start: int | None = None
    pending_end: int | None = None

    def flush() -> None:
        nonlocal pending_start, pending_end
        if pending_start is not None and pending_end is not None:
            append_range_as_chunks(
                chunks,
                source_file,
                text,
                pending_start,
                pending_end,
                max_chunk_size,
            )
        pending_start = pending_end = None

    for block in find_markdown_blocks(text):
        if block.end - block.start > max_chunk_size:
            flush()
            append_range_as_chunks(
                chunks,
                source_file,
                text,
                block.start,
                block.end,
                max_chunk_size,
            )
        elif pending_start is None:
            pending_start, pending_end = block.start, block.end
        elif block.end - pending_start <= max_chunk_size:
            pending_end = block.end
        else:
            flush()
            pending_start, pending_end = block.start, block.end
    flush()
    return Ok(chunks)
