"""Read discovered source files as UTF-8 text."""

from __future__ import annotations

import hashlib

from rag_against_the_machine.errors import Diagnostic, Ok, ReadError, Result
from rag_against_the_machine.indexing.error_helpers import (
    make_read_error,
)
from rag_against_the_machine.models.source import SourceFile


def _file_diagnostic(source_file: SourceFile, help_msg: str) -> Diagnostic:
    """Describe a source-file failure using a complete diagnostic location.

    Returns:
        A diagnostic spanning the stored source path.
    """
    filename = source_file.stored_path
    return Diagnostic(
        filename=filename,
        line_num=1,
        line_text=filename,
        col_start=0,
        col_end=len(filename),
        help_msg=help_msg,
    )


def read_source_file(
    source_file: SourceFile,
) -> Result[str, ReadError]:
    """Read a discovered source file as UTF-8 text.

    Returns:
        Decoded text, or a categorized read error.
    """
    result = read_source_file_with_hash(source_file)
    if isinstance(result, Ok):
        return Ok(result.value[0])
    return result


def read_source_file_with_hash(
    source_file: SourceFile,
) -> Result[tuple[str, str], ReadError]:
    """Read and hash a source file in one filesystem pass.

    Returns:
        Newline-normalized text and the SHA-256 hash of the original bytes.
    """
    path = source_file.absolute_path

    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
        # Match TextIOWrapper's universal-newline behavior used previously.
        text = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
        return Ok((text, hashlib.sha256(content).hexdigest()))

    except FileNotFoundError:
        return make_read_error(
            ReadError.FILE_NOT_FOUND,
            _file_diagnostic(source_file, "Run source-file discovery again."),
        )

    except PermissionError:
        return make_read_error(
            ReadError.FILE_NOT_READABLE,
            _file_diagnostic(source_file, "Check the file permissions."),
        )

    except IsADirectoryError:
        return make_read_error(
            ReadError.FILE_IS_DIRECTORY,
            _file_diagnostic(
                source_file, "Provide a source file, not a directory."
            ),
        )

    except UnicodeDecodeError:
        return make_read_error(
            ReadError.FILE_DECODE_FAILED,
            _file_diagnostic(
                source_file, "Ensure the file uses UTF-8 encoding."
            ),
        )

    except OSError:
        return make_read_error(
            ReadError.FILE_READ_FAILED,
            _file_diagnostic(
                source_file, "Check the file and filesystem state."
            ),
        )
