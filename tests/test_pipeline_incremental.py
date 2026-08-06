"""Integration coverage for incremental indexing state."""

import asyncio
import sqlite3
from pathlib import Path

from rag_against_the_machine.errors import Ok
from rag_against_the_machine.indexing.pipeline import run_pipeline
from rag_against_the_machine.models.source import SourceFile
from rag_against_the_machine.storage.db import Store


def test_startup_skips_current_file_and_reindexes_appended_tail(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "current.md"
    original = "# Current state\n\nThe original indexed content.\n"
    appended = "\n## Fixed tail chunk\n\nunique_database_tail_marker\n"
    source_path.write_text(original, encoding="utf-8")

    source = SourceFile(
        absolute_path=source_path,
        stored_path="current.md",
        file_type="markdown",
    )
    database_path = tmp_path / "index.db"
    store = Store(database_path)
    assert store.init() == Ok(None)

    first = asyncio.run(run_pipeline([source], 40, store)).unwrap()
    assert first.files_processed == 1
    assert first.files_skipped == 0

    unchanged = asyncio.run(run_pipeline([source], 40, store)).unwrap()
    assert unchanged.files_processed == 0
    assert unchanged.files_skipped == 1

    source_path.write_text(original + appended, encoding="utf-8")
    updated = asyncio.run(run_pipeline([source], 40, store)).unwrap()
    assert updated.files_processed == 1
    assert updated.files_skipped == 0

    current_again = asyncio.run(run_pipeline([source], 40, store)).unwrap()
    assert current_again.files_processed == 0
    assert current_again.files_skipped == 1

    with sqlite3.connect(database_path) as connection:
        source_rows = connection.execute(
            "SELECT COUNT(*) FROM source_files WHERE path = ?",
            (source.stored_path,),
        ).fetchone()
        chunk_rows = connection.execute(
            """
            SELECT text
            FROM chunks
            WHERE source_file_id = (
                SELECT id FROM source_files WHERE path = ?
            )
            ORDER BY chunk_index
            """,
            (source.stored_path,),
        ).fetchall()
        fts_rows = connection.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
            ("unique_database_tail_marker",),
        ).fetchone()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()

    assert source_rows == (1,)
    assert "".join(row[0] for row in chunk_rows) == original + appended
    assert "unique_database_tail_marker" in chunk_rows[-1][0]
    assert fts_rows == (1,)
    assert integrity == ("ok",)
