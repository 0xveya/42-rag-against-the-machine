"""Models used during source-file discovery."""

from dataclasses import dataclass

from rag_against_the_machine.models.source import FileType


@dataclass(frozen=True)
class Chunk:
    """A chunk of text from a source file."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    file_type: FileType
    search_text: str | None = None
