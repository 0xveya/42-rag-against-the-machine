from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from rag_against_the_machine.errors import Err, Nothing, Ok, Result, WatchError
from rag_against_the_machine.fs.events import EventKind, FileEvent
from rag_against_the_machine.fs.inotify import (
    DEFAULT_MASK,
    IN_ATTRIB,
    IN_CLOSE_WRITE,
    IN_CREATE,
    IN_DELETE,
    IN_DELETE_SELF,
    IN_IGNORED,
    IN_MODIFY,
    IN_MOVE_SELF,
    IN_MOVED_FROM,
    IN_MOVED_TO,
    IN_Q_OVERFLOW,
    Inotify,
    RawEvent,
)


@dataclass(frozen=True)
class WatcherOptions:
    recursive: bool = True
    follow_symlinks: bool = False
    mask: int = DEFAULT_MASK


@dataclass(frozen=True)
class PendingMove:
    path: Path
    is_dir: bool
    created_at: float


class WatchCoordinator:
    def __init__(self, backend: Inotify, root: Path, options: WatcherOptions) -> None:
        self.backend = backend
        self.root = root.resolve()
        self.options = options

        self.wd_to_path: dict[int, Path] = {}
        self.path_to_wd: dict[Path, int] = {}
        self.pending_moves: dict[int, PendingMove] = {}

    def initialize(self) -> Result[None, WatchError]:
        if not self.root.exists():
            return Err(
                WatchError.ROOT_NOT_FOUND,
                context_msg=f"Watch root does not exist: {self.root}",
                namespace="watch",
            )
        if not self.root.is_dir():
            return Err(
                WatchError.ROOT_NOT_DIRECTORY,
                context_msg=f"Path is not a directory: {self.root}",
                namespace="watch",
            )
        if not os.access(self.root, os.R_OK):
            return Err(
                WatchError.ROOT_NOT_READABLE,
                context_msg=f"Path is not readable: {self.root}",
                namespace="watch",
            )
        if self.options.recursive:
            return self.add_tree(self.root)
        return self.add_dir(self.root)

    def add_dir(self, path: Path) -> Result[None, WatchError]:
        directory = path.resolve()
        if directory in self.path_to_wd:
            return Ok(None)

        result = self.backend.add_watch(directory, self.options.mask)
        if isinstance(result, Err):
            return Err(
                WatchError.WATCH_REGISTRATION_FAILED,
                context_msg=result.context_msg,
                namespace="watch",
            )

        wd = result.value
        self.wd_to_path[wd] = directory
        self.path_to_wd[directory] = wd
        return Ok(None)

    def add_tree(self, root: Path) -> Result[None, WatchError]:
        root = root.resolve()
        try:
            for current, directory_names, _ in os.walk(
                root, followlinks=self.options.follow_symlinks
            ):
                current_path = Path(current)
                result = self.add_dir(current_path)
                if isinstance(result, Err):
                    return result

                if not self.options.follow_symlinks:
                    directory_names[:] = [
                        name for name in directory_names if not (current_path / name).is_symlink()
                    ]
        except OSError as error:
            return Err(
                WatchError.INITIAL_SCAN_FAILED,
                context_msg=f"Could not scan {root}: {error}",
                namespace="watch",
            )
        return Ok(None)

    def remove_mapping(self, wd: int) -> None:
        path = self.wd_to_path.pop(wd, Nothing())
        if isinstance(path, Nothing):
            return
        _ = self.path_to_wd.pop(path, None)

    def remove_subtree(self, root: Path) -> None:
        root = root.resolve()
        removals: list[tuple[int, Path]] = []
        for wd, watched_path in self.wd_to_path.items():
            try:
                watched_path.relative_to(root)
            except ValueError:
                continue
            removals.append((wd, watched_path))

        for wd, watched_path in removals:
            self.wd_to_path.pop(wd, None)
            self.path_to_wd.pop(watched_path, None)

    def rewrite_subtree(self, old_root: Path, new_root: Path) -> None:
        old_root = old_root.resolve()
        new_root = new_root.resolve()
        updates: list[tuple[int, Path, Path]] = []
        for wd, watched_path in self.wd_to_path.items():
            try:
                relative = watched_path.relative_to(old_root)
            except ValueError:
                continue
            updates.append((wd, watched_path, new_root / relative))

        for wd, old_path, new_path in updates:
            self.wd_to_path[wd] = new_path
            self.path_to_wd.pop(old_path, None)
            self.path_to_wd[new_path] = wd

    def process(self, raw: RawEvent) -> Result[list[FileEvent], WatchError]:
        if raw.mask & IN_Q_OVERFLOW:
            return Err(
                WatchError.EVENT_QUEUE_OVERFLOW,
                context_msg=(
                    "The inotify event queue overflowed. Some filesystem "
                    "events were lost and the tree must be rescanned."
                ),
                namespace="watch",
            )

        if raw.mask & IN_IGNORED:
            self.remove_mapping(raw.watch_descriptor)
            return Ok([])

        parent = self.wd_to_path.get(raw.watch_descriptor)
        if parent is None:
            return Err(
                WatchError.UNKNOWN_WATCH_DESCRIPTOR,
                context_msg=(
                    f"Received an event for unknown watch descriptor {raw.watch_descriptor}"
                ),
                namespace="watch",
            )

        path = parent / raw.name if raw.name else parent
        is_directory = raw.is_directory

        if raw.mask & IN_MOVED_FROM:
            self.pending_moves[raw.cookie] = PendingMove(
                path=path,
                is_dir=is_directory,
                created_at=time.monotonic(),
            )
            return Ok([])

        if raw.mask & IN_MOVED_TO:
            previous = self.pending_moves.pop(raw.cookie, Nothing())
            if isinstance(previous, Nothing):
                if self.options.recursive and is_directory:
                    result = self.add_tree(path)
                    if isinstance(result, Err):
                        return result
                return Ok([FileEvent(EventKind.CREATED, path, is_directory)])

            if previous.is_dir:
                self.rewrite_subtree(previous.path, path)
            return Ok([FileEvent.renamed(previous.path, path, is_directory=is_directory)])

        if raw.mask & IN_CREATE:
            if self.options.recursive and is_directory:
                result = self.add_tree(path)
                if isinstance(result, Err):
                    return result
            return Ok([FileEvent(EventKind.CREATED, path, is_directory)])

        if raw.mask & IN_DELETE:
            if is_directory:
                self.remove_subtree(path)
            return Ok([FileEvent(EventKind.DELETED, path, is_directory)])

        if raw.mask & IN_DELETE_SELF:
            self.remove_subtree(path)
            return Ok([FileEvent(EventKind.DELETED, path, True)])

        if raw.mask & IN_MOVE_SELF:
            return Ok([])

        if raw.mask & IN_ATTRIB:
            return Ok([FileEvent(EventKind.METADATA_CHANGED, path, is_directory)])

        if raw.mask & (IN_MODIFY | IN_CLOSE_WRITE):
            return Ok([FileEvent(EventKind.MODIFIED, path, is_directory)])

        return Ok([])

    def flush_expired_moves(self, max_age: float = 0.1) -> list[FileEvent]:
        now = time.monotonic()
        expired = [
            cookie
            for cookie, pending in self.pending_moves.items()
            if now - pending.created_at >= max_age
        ]
        events: list[FileEvent] = []
        for cookie in expired:
            pending = self.pending_moves.pop(cookie)
            if pending.is_dir:
                self.remove_subtree(pending.path)
            events.append(FileEvent(EventKind.DELETED, pending.path, pending.is_dir))
        return events

    def reset(self) -> Result[None, WatchError]:
        self.wd_to_path.clear()
        self.path_to_wd.clear()
        self.pending_moves.clear()
        if self.options.recursive:
            return self.add_tree(self.root)
        return self.add_dir(self.root)
