"""Allowlisted subprocess actions exposed by the local-only web UI."""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from rag_against_the_machine.errors import Err, Ok, Result, WebError
from rag_against_the_machine.server.models import (
    AddRepositoryRequest,
    CommandRequest,
    CommandResponse,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIRECTORY = ROOT / "data" / "raw"
_MAX_OUTPUT_CHARACTERS = 200_000
_COMMAND_LOCK = asyncio.Lock()


def _output(text: bytes) -> str:
    """Decode and bound subprocess output returned to the browser."""
    decoded = text.decode("utf-8", errors="replace")
    if len(decoded) <= _MAX_OUTPUT_CHARACTERS:
        return decoded
    return "[earlier output truncated]\n" + decoded[-_MAX_OUTPUT_CHARACTERS:]


async def _run(
    argv: list[str], label: str
) -> Result[CommandResponse, WebError]:
    """Run one argv-only subprocess without passing input through a shell."""
    try:
        async with _COMMAND_LOCK:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
    except OSError as error:
        return Err(
            WebError.COMMAND_FAILED,
            context_msg=f"Could not start {label}: {error}",
            namespace="server::actions",
        )

    return Ok(
        CommandResponse(
            command=label,
            exit_code=process.returncode or 0,
            stdout=_output(stdout),
            stderr=_output(stderr),
        )
    )


async def run_command(
    request: CommandRequest,
) -> Result[CommandResponse, WebError]:
    """Run an allowlisted project CLI command or the moulinette target."""
    if request.command == "moulinette":
        if request.arguments:
            return Err(
                WebError.INVALID_COMMAND,
                context_msg="moulinette does not accept CLI arguments here",
                namespace="server::actions",
            )
        return await _run(
            ["make", "--no-print-directory", "moulinette"], "moulinette"
        )

    argv = [sys.executable, "-m", "src", request.command, *request.arguments]
    return await _run(argv, request.command)


async def add_repository(
    request: AddRepositoryRequest,
    raw_directory: Path = DEFAULT_RAW_DIRECTORY,
) -> Result[CommandResponse, WebError]:
    """Clone one HTTPS Git repository below the configured raw directory."""
    parsed = urlparse(request.url)
    if parsed.scheme != "https" or not parsed.netloc:
        return Err(
            WebError.INVALID_COMMAND,
            context_msg="Repository URL must be an HTTPS URL",
            namespace="server::actions",
        )

    inferred_name = Path(parsed.path.rstrip("/")).name.removesuffix(".git")
    name = request.name or inferred_name
    if not name or re.fullmatch(r"[A-Za-z0-9._-]+", name) is None:
        return Err(
            WebError.INVALID_COMMAND,
            context_msg="Repository name may contain letters, numbers, '.', '_' and '-'",
            namespace="server::actions",
        )

    target = raw_directory / name
    if target.exists():
        return Err(
            WebError.REPOSITORY_CLONE_FAILED,
            context_msg=f"{target.relative_to(ROOT)} already exists",
            namespace="server::actions",
        )

    raw_directory.mkdir(parents=True, exist_ok=True)
    return await _run(
        ["git", "clone", "--depth", "1", "--", request.url, str(target)],
        f"add repository {name}",
    )
