"""Asynchronous event-loop bridge for normalized inotify events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TypeAlias, cast

from rag_against_the_machine.errors import (
    Err,
    Nothing,
    Ok,
    Option,
    Result,
    Some,
    WatchError,
)
from rag_against_the_machine.fs.coordinator import (
    WatchCoordinator,
    WatcherOptions,
)
from rag_against_the_machine.fs.events import FileEvent
from rag_against_the_machine.fs.inotify import DEFAULT_MASK, Inotify


@dataclass(frozen=True)
class _Closed:
    """Internal queue marker used to wake blocked receivers."""


_CLOSED = _Closed()

QueueItem: TypeAlias = Result[FileEvent, WatchError] | _Closed


class AsyncWatcher:
    """Bridge normalized inotify events into an asyncio queue."""

    def __init__(
        self,
        backend: Inotify,
        coordinator: WatchCoordinator,
        *,
        queue_maxsize: int = 0,
    ) -> None:
        """Create an unstarted watcher around an initialized coordinator."""
        self._backend = backend
        self._coordinator = coordinator

        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=queue_maxsize)

        self._loop: Option[asyncio.AbstractEventLoop] = Nothing()
        self._rename_timer: Option[asyncio.TimerHandle] = Nothing()

        self._started = False
        self._closed = False

        self._queue_maxsize = queue_maxsize

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        recursive: bool = True,
        follow_symlinks: bool = False,
        mask: int = DEFAULT_MASK,
        queue_maxsize: int = 0,
    ) -> Result[AsyncWatcher, WatchError]:
        """Create and initialize an unstarted watcher.

        The shared coordinator owns initial watch registration, including
        recursive registration when requested.

        Returns:
            The initialized watcher, or a categorized watch error.
        """
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

        initialized = coordinator.initialize()

        if isinstance(initialized, Err):
            backend.close()
            return Err(
                cast(WatchError, initialized.error),
                diagnostic=initialized.diagnostic,
                context_msg=initialized.context_msg,
                namespace=initialized.namespace,
            )

        return Ok(
            cls(
                backend,
                coordinator,
                queue_maxsize=queue_maxsize,
            )
        )

    async def start(
        self,
    ) -> Result[None, WatchError]:
        """Attach the inotify descriptor to the running event loop.

        Returns:
            Success, or a closure or event-loop registration error.
        """
        if self._closed:
            return Err(
                WatchError.WATCHER_CLOSED,
                context_msg="Cannot start a closed watcher.",
                namespace="watch",
            )

        if self._started:
            return Ok(None)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return Err(
                WatchError.EVENT_LOOP_UNAVAILABLE,
                context_msg="No running asyncio event loop.",
                namespace="watch",
            )

        try:
            loop.add_reader(
                self._backend.fd,
                self._on_readable,
            )
        except Exception as error:
            return Err(
                WatchError.EVENT_LOOP_REGISTRATION_FAILED,
                context_msg=(f"Failed to register the inotify file descriptor: {error}"),
                namespace="watch",
            )

        self._loop = Some(loop)
        self._started = True

        return Ok(None)

    async def recv(
        self,
    ) -> Result[FileEvent, WatchError]:
        """Wait for and return one normalized filesystem event.

        Returns:
            The next event result, or a watcher-closed error.
        """
        if self._closed and self._queue.empty():
            return Err(
                WatchError.WATCHER_CLOSED,
                context_msg="Cannot receive from a closed watcher",
                namespace="watch",
            )

        item = await self._queue.get()

        if isinstance(item, _Closed):
            return Err(
                WatchError.WATCHER_CLOSED,
                context_msg="Watcher was closed",
                namespace="watch",
            )

        return item

    async def recv_timeout(
        self,
        timeout: float,
    ) -> Result[FileEvent, WatchError]:
        """Wait up to ``timeout`` seconds for one normalized event.

        Returns:
            The next event result, or a timeout or closure error.
        """
        try:
            return await asyncio.wait_for(self.recv(), timeout)
        except TimeoutError:
            return Err(
                WatchError.RECEIVE_TIMEOUT,
                context_msg=f"No filesystem event received within {timeout} seconds.",
                namespace="watch",
            )

    def close(self) -> None:
        """Detach from the event loop, close the backend, and wake receivers."""
        if self._closed:
            return

        self._closed = True
        self._started = False

        match self._loop:
            case Some(value=raw_loop):
                loop = cast(asyncio.AbstractEventLoop, raw_loop)
                _ = loop.remove_reader(self._backend.fd)
            case Nothing():
                pass

        self._loop = Nothing()

        match self._rename_timer:
            case Some(value=raw_timer):
                timer = cast(asyncio.TimerHandle, raw_timer)
                timer.cancel()
            case Nothing():
                pass

        self._rename_timer = Nothing()

        self._backend.close()
        self._queue.put_nowait(_CLOSED)

    def is_started(self) -> bool:
        """Return whether the watcher is registered with an event loop."""
        return self._started

    def is_closed(self) -> bool:
        """Return whether the watcher has been closed."""
        return self._closed

    def __aiter__(
        self,
    ) -> AsyncIterator[Result[FileEvent, WatchError]]:
        """Return this watcher as its own asynchronous iterator."""
        return self

    async def __anext__(
        self,
    ) -> Result[FileEvent, WatchError]:
        """Return the next queued result, or stop after closure.

        Raises:
            StopAsyncIteration: When the watcher has closed.
        """
        item = await self._queue.get()
        if isinstance(item, _Closed):
            raise StopAsyncIteration
        return item

    async def __aenter__(self) -> AsyncWatcher:
        """Start the watcher and enter its asynchronous context.

        Returns:
            This started watcher.
        """
        result = await self.start()
        if isinstance(result, Err):
            result.unwrap()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the watcher when leaving its asynchronous context."""
        self.close()

    def _on_readable(self) -> None:
        """Read, normalize, and enqueue all currently available raw events."""
        raw_result = self._backend.read()
        if isinstance(raw_result, Err):
            self._enqueue(
                Err(
                    WatchError.EVENT_READ_FAILED,
                    context_msg=raw_result.context_msg,
                    namespace="watch",
                )
            )
            return

        for raw_event in raw_result.value:
            event_result = self._coordinator.process(raw_event)
            if isinstance(event_result, Err):
                self._enqueue(
                    Err(
                        cast(WatchError, event_result.error),
                        diagnostic=event_result.diagnostic,
                        context_msg=event_result.context_msg,
                        namespace=event_result.namespace,
                    )
                )
                continue
            for event in event_result.value:
                self._enqueue(Ok(event))

        self._schedule_move_flush()

    def _enqueue(
        self,
        result: Result[FileEvent, WatchError],
    ) -> None:
        """Insert a normalized result without blocking the event loop."""
        self._queue.put_nowait(result)

    def _schedule_move_flush(self) -> None:
        """Schedule one timer when the coordinator has pending moves."""
        if not self._coordinator.pending_moves or isinstance(self._loop, Nothing):
            return
        if isinstance(self._rename_timer, Some):
            return
        self._rename_timer = Some(self._loop.value.call_later(0.1, self._flush_expired_moves))

    def _flush_expired_moves(self) -> None:
        """Enqueue expired moves and reschedule while moves remain pending."""
        expired = self._coordinator.flush_expired_moves()
        for event in expired:
            self._enqueue(Ok(event))

        self._rename_timer = Nothing()
        self._schedule_move_flush()
