"""Integration tests for the asyncio inotify bridge."""

import asyncio
from pathlib import Path

import pytest

from rag_against_the_machine.errors import Err, Ok, WatchError
from rag_against_the_machine.fs.events import EventKind
from rag_against_the_machine.fs.watcher_async import AsyncWatcher


def test_async_watcher_delivers_normalized_create(tmp_path: Path) -> None:
    """Deliver coordinator output through the public async queue."""

    async def exercise() -> None:
        opened = AsyncWatcher.open(tmp_path)
        assert isinstance(opened, Ok)
        watcher = opened.q

        started = await watcher.start()
        assert started == Ok(None)
        created = tmp_path / "created.txt"
        created.write_text("hello")

        result = await watcher.recv_timeout(1.0)
        watcher.close()

        assert isinstance(result, Ok)
        assert result.q.kind is EventKind.CREATED
        assert result.q.path == created

    asyncio.run(exercise())


def test_async_watcher_close_wakes_receiver(tmp_path: Path) -> None:
    """Return a typed closure error to a receiver after close."""

    async def exercise() -> None:
        opened = AsyncWatcher.open(tmp_path)
        assert isinstance(opened, Ok)
        watcher = opened.q
        assert await watcher.start() == Ok(None)

        watcher.close()
        result = await watcher.recv()

        assert isinstance(result, Err)
        assert result.error is WatchError.WATCHER_CLOSED

    asyncio.run(exercise())


def test_async_iteration_stops_after_close(tmp_path: Path) -> None:
    """Translate the internal closed marker into async iteration completion."""

    async def exercise() -> None:
        opened = AsyncWatcher.open(tmp_path)
        assert isinstance(opened, Ok)
        watcher = opened.q
        watcher.close()

        with pytest.raises(StopAsyncIteration):
            await watcher.__anext__()

    asyncio.run(exercise())
