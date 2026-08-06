"""Normalized filesystem event types shared by watcher frontends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from rag_against_the_machine.errors import Nothing, Option, Some


class EventKind(Enum):
    """Identify the normalized kind of a filesystem change."""

    CREATED = auto()
    MODIFIED = auto()
    DELETED = auto()
    RENAMED = auto()
    METADATA_CHANGED = auto()


@dataclass(frozen=True)
class FileEvent:
    """Describe one normalized filesystem change."""

    kind: EventKind
    path: Path
    is_directory: bool = False
    old_path: Option[Path] = Nothing()

    @classmethod
    def renamed(
        cls,
        old_path: Path,
        new_path: Path,
        *,
        is_directory: bool,
    ) -> FileEvent:
        """Create a normalized rename event.

        Returns:
            A rename event containing both old and new paths.
        """
        return cls(
            kind=EventKind.RENAMED,
            old_path=Some(old_path),
            path=new_path,
            is_directory=is_directory,
        )
