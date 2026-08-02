"""Tests for exception-free SQLite storage results."""

from pathlib import Path

from rag_against_the_machine.errors import Err, Nothing, Ok, StorageError
from rag_against_the_machine.storage.db import Store, Transaction


def test_store_initializes_and_returns_options(tmp_path: Path) -> None:
    """Initialize storage and represent a missing row with Nothing."""
    store = Store(tmp_path / "nested" / "index.db")
    assert store.init() == Ok(None)

    def lookup(tx: Transaction):
        return tx.queries.get_source_file("missing.py")

    assert store.with_tx(lookup) == Ok(Nothing())


def test_store_converts_directory_failure_to_result(tmp_path: Path) -> None:
    """Return directory creation failures instead of raising OSError."""
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("data")

    result = Store(parent_file / "index.db").init()

    assert isinstance(result, Err)
    assert result.error is StorageError.DIRECTORY_CREATION_FAILED
