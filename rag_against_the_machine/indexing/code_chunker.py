"""Tree-sitter-backed source-code chunking."""

from dataclasses import dataclass

from tree_sitter import Node, Parser

from rag_against_the_machine.errors import ChunkingError, Err, Ok, Result
from rag_against_the_machine.indexing.chunk_helpers import (
    append_range_as_chunks,
    chunk_plain_text,
    include_range_gaps,
)
from rag_against_the_machine.indexing.languages.config import (
    LanguageChunkConfig,
)
from rag_against_the_machine.models.chunk import Chunk
from rag_against_the_machine.models.source import SourceFile


@dataclass(frozen=True, slots=True)
class ByteCharacterMap:
    """Map UTF-8 byte offsets to Python character indexes."""

    values: tuple[int, ...]

    def to_character(self, byte_offset: int) -> int:
        """Translate a Tree-sitter byte offset."""
        if byte_offset < 0 or byte_offset >= len(self.values):
            raise ValueError(f"Invalid Tree-sitter byte offset: {byte_offset}")
        return self.values[byte_offset]


def build_byte_character_map(text: str) -> ByteCharacterMap:
    """Build a UTF-8-byte to character lookup."""
    values: list[int] = []
    for index, character in enumerate(text):
        values.extend([index] * len(character.encode("utf-8")))
    values.append(len(text))
    return ByteCharacterMap(tuple(values))


def parse_root(
    text: str, config: LanguageChunkConfig
) -> tuple[Node, ByteCharacterMap]:
    """Parse text with the configured grammar."""
    parser = Parser(config.language)
    tree = parser.parse(text.encode("utf-8"))
    return tree.root_node, build_byte_character_map(text)


def find_structural_ranges(
    text: str, config: LanguageChunkConfig
) -> list[tuple[int, int]]:
    """Return valid top-level syntax ranges in character offsets."""
    root, offsets = parse_root(text, config)
    ranges: list[tuple[int, int]] = []
    for node in root.named_children:
        if node.has_error or node.type not in config.structural_nodes:
            continue
        start = offsets.to_character(node.start_byte)
        end = offsets.to_character(node.end_byte)
        if start < end:
            ranges.append((start, end))
    return ranges


def chunk_code(
    source_file: SourceFile,
    text: str,
    max_chunk_size: int,
    config: LanguageChunkConfig,
) -> Result[list[Chunk], ChunkingError]:
    """Chunk source code using configured top-level syntax boundaries."""
    if max_chunk_size <= 0:
        return Err(ChunkingError.INVALID_MAX_CHUNK_SIZE)
    try:
        structural_ranges = find_structural_ranges(text, config)
        ranges = include_range_gaps(structural_ranges, len(text))
    except (UnicodeError, ValueError):
        return chunk_plain_text(source_file, text, max_chunk_size)
    chunks: list[Chunk] = []
    for start, end in ranges:
        append_range_as_chunks(
            chunks, source_file, text, start, end, max_chunk_size
        )
    return Ok(chunks)
