from __future__ import annotations

import select
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from rag_against_the_machine.errors import Err, Ok, Result, WatchError
from rag_against_the_machine.fs.coordinator import (
    WatchCoordinator,
    WatcherOptions,
)
from rag_against_the_machine.fs.events import FileEvent
from rag_against_the_machine.fs.inotify import Inotify


class Watcher:
    def __init__(
        self,
        backend: Inotify,
        coordinator: WatchCoordinator,
    ) -> None:
        self._backend = backend
        self._coordinator = coordinator
        self._pending: deque[FileEvent] = deque()
        self._closed = False

        self._poller = select.poll()
        self._poller.register(
            backend.fd,
            select.POLLIN | select.POLLERR | select.POLLHUP,
        )

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        recursive: bool = True,
        follow_symlinks: bool = False,
    ) -> Result[Watcher, WatchError]:
        backend_result = Inotify.open()

        if isinstance(backend_result, Err):
            return Err(
                WatchError.WATCH_REGISTRATION_FAILED,
                context_msg=backend_result.context_msg,
                namespace="watch",
            )

        backend = backend_result.value
        coordinator = WatchCoordinator(
            backend=backend,
            root=Path(path).resolve(),
            options=WatcherOptions(
                recursive=recursive,
                follow_symlinks=follow_symlinks,
            ),
        )

        initialize_result = coordinator.initialize()
        if isinstance(initialize_result, Err):
            backend.close()
            return Err(
                cast(WatchError, initialize_result.error),
                diagnostic=initialize_result.diagnostic,
                context_msg=initialize_result.context_msg,
                namespace=initialize_result.namespace,
            )

        return Ok(cls(backend, coordinator))

    def recv(self) -> Result[FileEvent, WatchError]:
        if self._pending:
            return Ok(self._pending.popleft())

        if self._closed:
            return Err(
                WatchError.WATCHER_CLOSED,
                context_msg="Cannot receive from a closed watcher",
                namespace="watch",
            )

        while True:
            try:
                ready = self._poller.poll(100)
            except InterruptedError:
                continue
            except OSError as error:
                return Err(
                    WatchError.POLL_FAILED,
                    context_msg=f"Polling the inotify descriptor failed: {error}",
                    namespace="watch",
                )

            expired_moves = self._coordinator.flush_expired_moves()
            if expired_moves:
                self._pending.extend(expired_moves)
                return Ok(self._pending.popleft())

            if not ready:
                continue

            for _, flags in ready:
                if flags & (select.POLLERR | select.POLLHUP):
                    return Err(
                        WatchError.EVENT_READ_FAILED,
                        context_msg=(f"The inotify descriptor reported poll flags {flags}"),
                        namespace="watch",
                    )

            raw_result = self._backend.read()
            if isinstance(raw_result, Err):
                return Err(
                    WatchError.EVENT_READ_FAILED,
                    context_msg=raw_result.context_msg,
                    namespace="watch",
                )

            for raw_event in raw_result.value:
                event_result = self._coordinator.process(raw_event)
                if isinstance(event_result, Err):
                    return Err(
                        cast(WatchError, event_result.error),
                        diagnostic=event_result.diagnostic,
                        context_msg=event_result.context_msg,
                        namespace=event_result.namespace,
                    )
                self._pending.extend(event_result.value)

            if self._pending:
                return Ok(self._pending.popleft())

    def __iter__(self) -> Iterator[Result[FileEvent, WatchError]]:
        while not self._closed:
            result = self.recv()
            yield result
            if isinstance(result, Err):
                return

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        try:
            self._poller.unregister(self._backend.fd)
        except KeyError:
            pass

        self._backend.close()

    def __enter__(self) -> Watcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
