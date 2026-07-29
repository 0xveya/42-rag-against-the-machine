from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from rag_against_the_machine.errors import Err, Ok, WatchError
from rag_against_the_machine.fs import watcher as watcher_module
from rag_against_the_machine.fs.coordinator import WatchCoordinator
from rag_against_the_machine.fs.events import EventKind, FileEvent
from rag_against_the_machine.fs.inotify import Inotify
from rag_against_the_machine.fs.watcher import Watcher


class FakePoller:
    def __init__(self, ready: list[tuple[int, int]]) -> None:
        self.ready = ready
        self.registered: tuple[int, int] | None = None
        self.unregistered: int | None = None
        self.poll_calls = 0

    def register(self, fd: int, events: int) -> None:
        self.registered = (fd, events)

    def poll(self, _timeout: int) -> list[tuple[int, int]]:
        self.poll_calls += 1
        return self.ready

    def unregister(self, fd: int) -> None:
        self.unregistered = fd


class FakeBackend:
    fd = 42

    def __init__(self, raw_events: list[object] | None = None) -> None:
        self.raw_events = raw_events or []
        self.closed = False
        self.read_calls = 0

    def read(self) -> Ok[list[object]]:
        self.read_calls += 1
        return Ok(self.raw_events)

    def close(self) -> None:
        self.closed = True


class FakeCoordinator:
    def __init__(self, events: dict[object, list[FileEvent]]) -> None:
        self.events = events
        self.initialized = False
        self.flush_calls = 0

    def initialize(self) -> Ok[None]:
        self.initialized = True
        return Ok(None)

    def flush_expired_moves(self) -> list[FileEvent]:
        self.flush_calls += 1
        return []

    def process(self, raw_event: object) -> Ok[list[FileEvent]]:
        return Ok(self.events[raw_event])


def make_watcher(
    monkeypatch: pytest.MonkeyPatch,
    backend: FakeBackend,
    coordinator: FakeCoordinator,
    poller: FakePoller,
) -> Watcher:
    monkeypatch.setattr(watcher_module.select, "poll", lambda: poller)
    return Watcher(cast(Inotify, backend), cast(WatchCoordinator, coordinator))


def test_recv_preserves_all_events_from_one_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = object()
    second = object()
    created = FileEvent(EventKind.CREATED, Path("created.txt"))
    modified = FileEvent(EventKind.MODIFIED, Path("created.txt"))
    backend = FakeBackend([first, second])
    coordinator = FakeCoordinator({first: [created], second: [modified]})
    poller = FakePoller([(backend.fd, watcher_module.select.POLLIN)])
    watcher = make_watcher(monkeypatch, backend, coordinator, poller)

    assert watcher.recv() == Ok(created)
    assert watcher.recv() == Ok(modified)
    assert backend.read_calls == 1
    assert poller.poll_calls == 1


def test_recv_returns_backend_errors_as_watch_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    coordinator = FakeCoordinator({})
    poller = FakePoller([(backend.fd, watcher_module.select.POLLERR)])
    watcher = make_watcher(monkeypatch, backend, coordinator, poller)

    result = watcher.recv()

    assert isinstance(result, Err)
    assert result.error is WatchError.EVENT_READ_FAILED
    assert backend.read_calls == 0


def test_close_is_idempotent_and_recv_reports_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    coordinator = FakeCoordinator({})
    poller = FakePoller([])
    watcher = make_watcher(monkeypatch, backend, coordinator, poller)

    watcher.close()
    watcher.close()

    result = watcher.recv()
    assert isinstance(result, Err)
    assert result.error is WatchError.WATCHER_CLOSED
    assert backend.closed
    assert poller.unregistered == backend.fd


def test_open_initializes_coordinator_and_closes_backend_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()

    class FakeInotify:
        @classmethod
        def open(cls) -> Ok[FakeBackend]:
            return Ok(backend)

    class FailingCoordinator:
        def __init__(self, **_: object) -> None:
            pass

        def initialize(self) -> Err[WatchError]:
            return Err(WatchError.ROOT_NOT_FOUND)

    monkeypatch.setattr(watcher_module, "Inotify", FakeInotify)
    monkeypatch.setattr(watcher_module, "WatchCoordinator", FailingCoordinator)

    result = Watcher.open("missing")

    assert isinstance(result, Err)
    assert result.error is WatchError.ROOT_NOT_FOUND
    assert backend.closed


def test_context_manager_closes_watcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    coordinator = FakeCoordinator({})
    poller = FakePoller([])
    watcher = make_watcher(monkeypatch, backend, coordinator, poller)

    with watcher as entered:
        assert entered is watcher

    assert backend.closed
