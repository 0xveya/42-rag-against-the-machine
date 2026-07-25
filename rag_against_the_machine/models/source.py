"""Models used during source-file discovery."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias


FileType: TypeAlias = Literal["python", "markdown", "text"]


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A file selected for indexing."""

    absolute_path: Path
    stored_path: str
    file_type: FileType


@dataclass(frozen=True, slots=True)
class Chunk:
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    file_type: FileType
