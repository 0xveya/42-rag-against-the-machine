"""Models used during source-file discovery."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

# Canonical language names come from the indexing language registry.
FileType: TypeAlias = str


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A file selected for indexing."""

    absolute_path: Path
    stored_path: str
    file_type: FileType
