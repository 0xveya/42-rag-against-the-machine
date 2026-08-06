"""Route source files to the configured chunking strategy."""

from rag_against_the_machine.errors import ChunkingError, Err, Result
from rag_against_the_machine.indexing.chunk_helpers import chunk_plain_text
from rag_against_the_machine.indexing.code_chunker import chunk_code
from rag_against_the_machine.indexing.languages.registry import (
    get_language_config,
)
from rag_against_the_machine.indexing.markdown_chunker import chunk_markdown
from rag_against_the_machine.models.chunk import Chunk
from rag_against_the_machine.models.source import SourceFile


def chunk_source_file(
    source_file: SourceFile, text: str, max_chunk_size: int
) -> Result[list[Chunk], ChunkingError]:
    """Route source text to Markdown, Tree-sitter, or plain-text chunking."""
    if max_chunk_size <= 0:
        return Err(ChunkingError.INVALID_MAX_CHUNK_SIZE)
    if source_file.file_type == "markdown":
        return chunk_markdown(source_file, text, max_chunk_size)
    config = get_language_config(
        source_file.file_type, source_file.stored_path
    )
    if config is not None:
        return chunk_code(source_file, text, max_chunk_size, config)
    return chunk_plain_text(source_file, text, max_chunk_size)
