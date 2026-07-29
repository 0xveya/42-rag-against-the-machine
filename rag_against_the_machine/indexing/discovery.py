"""Discover source files that should be indexed."""

import os
from pathlib import Path

from rag_against_the_machine.errors import (
    Diagnostic,
    DiscoveryError,
    Ok,
    Result,
)
from rag_against_the_machine.indexing.error_helpers import (
    make_discovery_error,
)
from rag_against_the_machine.models.source import (
    FileType,
    SourceFile,
)

_SUFFIX_TYPES: dict[str, FileType] = {
    ".py": "python",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}

_IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}


def _path_diagnostic(filename: str, help_msg: str) -> Diagnostic:
    """Describe a path failure using a complete diagnostic location."""
    return Diagnostic(
        filename=filename,
        line_num=1,
        line_text=filename,
        col_start=0,
        col_end=len(filename),
        help_msg=help_msg,
    )


def discover_files(
    source_root: Path,
    project_root: Path,
) -> Result[list[SourceFile], DiscoveryError]:
    """Discover supported files below a source directory.

    Args:
        source_root: Directory containing files to index.
        project_root: Project directory used to create stored paths.

    Returns:
        An ordered list of discovered files, or a discovery error.
    """
    source_root = source_root.resolve()
    project_root = project_root.resolve()

    if not project_root.exists():
        return make_discovery_error(
            DiscoveryError.PROJECT_DOES_NOT_EXIST,
            _path_diagnostic(
                str(project_root),
                "Provide the path to the project root.",
            ),
        )

    if not project_root.is_dir():
        return make_discovery_error(
            DiscoveryError.PROJECT_IS_NOT_A_DIRECTORY,
            _path_diagnostic(
                str(project_root),
                "The project root must refer to a directory.",
            ),
        )

    if not source_root.exists():
        return make_discovery_error(
            DiscoveryError.SOURCE_DOES_NOT_EXIST,
            _path_diagnostic(
                str(source_root),
                "Check that the source path exists.",
            ),
        )

    if not source_root.is_dir():
        return make_discovery_error(
            DiscoveryError.SOURCE_IS_NOT_A_DIRECTORY,
            _path_diagnostic(
                str(source_root),
                "Provide a directory containing source files.",
            ),
        )

    if not os.access(source_root, os.R_OK | os.X_OK):
        return make_discovery_error(
            DiscoveryError.SOURCE_IS_NOT_READABLE,
            _path_diagnostic(
                str(source_root),
                "The directory requires read and traversal permissions.",
            ),
        )

    try:
        source_root.relative_to(project_root)
    except ValueError:
        return make_discovery_error(
            DiscoveryError.SOURCE_OUTSIDE_PROJECT,
            _path_diagnostic(
                str(source_root),
                "Place the source directory inside the project root "
                "so stored paths can be project-relative.",
            ),
        )

    files: list[SourceFile] = []

    try:
        for path in source_root.rglob("*"):
            if not path.is_file():
                continue

            if any(folder in _IGNORED_DIRECTORY_NAMES for folder in path.parts):
                continue

            file_type = _SUFFIX_TYPES.get(path.suffix.lower())
            if file_type is None:
                continue

            if not os.access(path, os.R_OK):
                continue

            files.append(
                SourceFile(
                    absolute_path=path,
                    stored_path=(path.relative_to(project_root).as_posix()),
                    file_type=file_type,
                )
            )

    except OSError as exc:
        return make_discovery_error(
            DiscoveryError.DIRECTORY_TRAVERSAL_FAILED,
            _path_diagnostic(str(source_root), str(exc)),
        )

    files.sort(key=lambda source_file: source_file.stored_path)
    return Ok(files)
