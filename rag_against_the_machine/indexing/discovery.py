"""Discover source files that should be indexed."""

import os
from pathlib import Path

from rag_against_the_machine.errors import DiscoveryError
from rag_against_the_machine.indexing.error_helpers import (
    make_discovery_error,
)
from rag_against_the_machine.models.source import (
    FileType,
    SourceFile,
)
from rag_against_the_machine.errors import Ok, Result


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
            filename=str(project_root),
            help_msg="Provide the path to the project root.",
        )

    if not project_root.is_dir():
        return make_discovery_error(
            DiscoveryError.PROJECT_IS_NOT_A_DIRECTORY,
            filename=str(project_root),
            help_msg="The project root must refer to a directory.",
        )

    if not source_root.exists():
        return make_discovery_error(
            DiscoveryError.SOURCE_DOES_NOT_EXIST,
            filename=str(source_root),
            help_msg="Check that the source path exists.",
        )

    if not source_root.is_dir():
        return make_discovery_error(
            DiscoveryError.SOURCE_IS_NOT_A_DIRECTORY,
            filename=str(source_root),
            help_msg="Provide a directory containing source files.",
        )

    if not os.access(source_root, os.R_OK | os.X_OK):
        return make_discovery_error(
            DiscoveryError.SOURCE_IS_NOT_READABLE,
            filename=str(source_root),
            help_msg=(
                "The directory requires read and traversal permissions."
            ),
        )

    try:
        source_root.relative_to(project_root)
    except ValueError:
        return make_discovery_error(
            DiscoveryError.SOURCE_OUTSIDE_PROJECT,
            filename=str(source_root),
            help_msg=(
                "Place the source directory inside the project root "
                "so stored paths can be project-relative."
            ),
        )

    files: list[SourceFile] = []

    try:
        for path in source_root.rglob("*"):
            if not path.is_file():
                continue

            if any(
                folder in _IGNORED_DIRECTORY_NAMES for folder in path.parts
            ):
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
            filename=str(source_root),
            help_msg=str(exc),
        )

    files.sort(key=lambda source_file: source_file.stored_path)
    return Ok(files)
