"""Repository-scoped explorer checks against real indexing and SQLite."""

import asyncio
from pathlib import Path

import pytest

from rag_against_the_machine.errors import Ok
from rag_against_the_machine.server.inspection import inspect_sources
from rag_against_the_machine.server.service import WebRagService


def test_repository_search_and_index_removal_are_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another repository's better matches must not exhaust the result limit."""
    monkeypatch.chdir(tmp_path)
    raw = Path("repos")
    selected = raw / "selected"
    other = raw / "other"
    selected.mkdir(parents=True)
    other.mkdir()
    selected_text = "def retrieve_sources():\n    return 'selected repository'\n"
    (selected / "search.py").write_text(selected_text)
    for index in range(12):
        (other / f"search{index}.py").write_text(
            "def retrieve_sources():\n    return 'retrieve_sources'\n"
        )
    service = WebRagService(raw_directory=raw, database_path=Path("index.db"))
    assert service.store.init() == Ok(None)

    async def check() -> None:
        assert isinstance(await service.reindex("other"), Ok)
        assert isinstance(await service.reindex("selected"), Ok)
        sources = await service.retrieve("selected", "retrieve_sources", 1)
        assert isinstance(sources, Ok)
        assert len(sources.value) == 1
        source = sources.value[0]
        assert source.file_path == "repos/selected/search.py"
        assert source.text == selected_text[
            source.first_character_index:source.last_character_index
        ]
        assert source.score < 0
        inspection = inspect_sources("retrieve_sources", sources.value)
        assert inspection.query == '"retrieve_sources"'
        assert source.text in inspection.messages[1].content
        assert inspection.sources[0].score == source.score
        removed = await service.remove_index("selected")
        assert isinstance(removed, Ok)
        assert removed.value.files == removed.value.chunks == 0
        assert (await service.retrieve("selected", "retrieve_sources", 1)) == Ok([])
        remaining = await service.index_status("other")
        assert isinstance(remaining, Ok)
        assert remaining.value.files == 12
        assert isinstance(await service.retrieve("other", "retrieve_sources", 1), Ok)
        assert (selected / "search.py").read_text() == selected_text

    asyncio.run(check())
