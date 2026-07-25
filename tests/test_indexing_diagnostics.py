"""Tests for source discovery and reading diagnostics."""

from collections.abc import Callable
from pathlib import Path

import pytest

from rag_against_the_machine.errors import DiscoveryError, Err, ReadError
from rag_against_the_machine.indexing.discovery import discover_files
from rag_against_the_machine.indexing.reader import read_source_file
from rag_against_the_machine.models.source import SourceFile


def _source_file(path: Path) -> SourceFile:
    return SourceFile(
        absolute_path=path,
        stored_path="sources/example.py",
        file_type="python",
    )


def _assert_path_diagnostic(err: Err[DiscoveryError] | Err[ReadError], filename: str) -> None:
    assert err.diagnostic is not None
    diagnostic = err.diagnostic
    assert diagnostic.filename == filename
    assert diagnostic.line_num == 1
    assert diagnostic.line_text == filename
    assert diagnostic.col_start == 0
    assert diagnostic.col_end == len(filename)


def test_discovery_missing_source_uses_complete_diagnostic(tmp_path: Path) -> None:
    source_root = tmp_path / "missing"

    result = discover_files(source_root, tmp_path)

    assert isinstance(result, Err)
    assert result.error is DiscoveryError.SOURCE_DOES_NOT_EXIST
    _assert_path_diagnostic(result, str(source_root))
    assert result.namespace == "indexing::discovery"
    assert result.context_msg == "Source-file discovery failed"


@pytest.mark.parametrize(
    ("operation", "expected_error"),
    [
        (lambda source: source.absolute_path.unlink(), ReadError.FILE_NOT_FOUND),
        (
            lambda source: source.absolute_path.write_bytes(b"\xff"),
            ReadError.FILE_DECODE_FAILED,
        ),
    ],
)
def test_reader_path_failures_use_complete_diagnostics(
    tmp_path: Path,
    operation: Callable[[SourceFile], None],
    expected_error: ReadError,
) -> None:
    path = tmp_path / "example.py"
    path.write_text("print('ok')\n", encoding="utf-8")
    source = _source_file(path)
    operation(source)

    result = read_source_file(source)

    assert isinstance(result, Err)
    assert result.error is expected_error
    _assert_path_diagnostic(result, source.stored_path)
    assert result.namespace == "indexing::read"
    assert result.context_msg == "Source-file reading failed"


def test_reader_directory_uses_specific_error(tmp_path: Path) -> None:
    source = _source_file(tmp_path)

    result = read_source_file(source)

    assert isinstance(result, Err)
    assert result.error is ReadError.FILE_IS_DIRECTORY
    _assert_path_diagnostic(result, source.stored_path)
