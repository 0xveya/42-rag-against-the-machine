"""Helpers for constructing indexing errors."""

from rag_against_the_machine.errors import (
    Diagnostic,
    DiscoveryError,
    Err,
    ReadError,
)


def make_discovery_error(
    error: DiscoveryError,
    diag: Diagnostic,
) -> Err[DiscoveryError]:
    """Create a consistently formatted discovery error."""
    return Err(
        error=error,
        diagnostic=diag,
        namespace="indexing::discovery",
        context_msg="Source-file discovery failed",
    )


def make_read_error(
    error: ReadError,
    diag: Diagnostic,
) -> Err[ReadError]:
    """Create a consistently formatted source-file reading error."""
    return Err(
        error=error,
        diagnostic=diag,
        namespace="indexing::read",
        context_msg="Source-file reading failed",
    )
