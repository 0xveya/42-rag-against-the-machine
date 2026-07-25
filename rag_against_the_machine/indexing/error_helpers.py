"""Helpers for constructing indexing errors."""

from rag_against_the_machine.errors import (
    Diagnostic,
    DiscoveryError,
    Err,
)


def make_discovery_error(
    error: DiscoveryError,
    *,
    filename: str,
    help_msg: str | None = None,
) -> Err[DiscoveryError]:
    """Create a consistently formatted discovery error."""
    return Err(
        error=error,
        diagnostic=Diagnostic(
            filename=filename,
            help_msg=help_msg,
        ),
        namespace="indexing::discovery",
        context_msg="Source-file discovery failed",
    )
