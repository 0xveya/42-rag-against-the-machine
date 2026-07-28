from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class EventKind(Enum):
    CREATED = auto()
    MODIFIED = auto()
    DELETED = auto()
    RENAMED = auto()
    METADATA_CHANGED = auto()


@dataclass(frozen=True)
class FileEvent:
    kind: EventKind
    path: Path
    is_directory: bool = False
    old_path: Path | None = None

    @classmethod
    def renamed(
        cls,
        old_path: Path,
        new_path: Path,
        *,
        is_directory: bool,
    ) -> FileEvent:
        return cls(
            kind=EventKind.RENAMED,
            old_path=old_path,
            path=new_path,
            is_directory=is_directory,
        )
