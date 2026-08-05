"""Python Fire command-line entry point for the project scaffold."""

import sys
from dataclasses import dataclass, field
from os.path import basename

from rag_against_the_machine.cli_fw import Command
from rag_against_the_machine.errors import Err


@dataclass(frozen=True)
class ServeOptions:
    """Document the options accepted by the development server command."""

    port: int = field(
        default=8000, metadata={"help": "Port for the local API server."}
    )


def tmp() -> None:
    """Run the temporary indexing demonstration command."""
    import asyncio

    asyncio.run(_tmp())


async def _tmp() -> None:
    """Run initial indexing, then watch for filesystem changes."""
    from pathlib import Path

    from rag_against_the_machine.errors import Err
    from rag_against_the_machine.fs.watcher_async import AsyncWatcher
    from rag_against_the_machine.indexing.discovery import discover_files
    from rag_against_the_machine.indexing.fsevent_handler import handle_event
    from rag_against_the_machine.indexing.pipeline import run_pipeline
    from rag_against_the_machine.storage.db import Store

    source_root = Path("data/raw/gns3util")
    project_root = Path.cwd()

    store = Store(Path("data/output/stuff.db"))

    initialized = store.init()
    if isinstance(initialized, Err):
        initialized.print_diagnostic()
        return

    discovered = discover_files(
        source_root=source_root,
        project_root=project_root,
    )
    if isinstance(discovered, Err):
        discovered.print_diagnostic()
        return

    pipeline_result = await run_pipeline(
        discovered.unwrap(),
        max_chunk_size=1500,
        store=store,
    )
    if isinstance(pipeline_result, Err):
        pipeline_result.print_diagnostic()
        return

    output = pipeline_result.unwrap()

    print(
        f"Initial indexing complete: "
        f"{output.files_processed} files indexed; "
        f"{output.files_skipped} files skipped."
    )

    for diagnostic in output.diagnostics:
        print(f"warning: {diagnostic.filename}: {diagnostic.help_msg}")

    watcher_result = AsyncWatcher.open(
        source_root,
        recursive=True,
    )
    if isinstance(watcher_result, Err):
        watcher_result.print_diagnostic()
        return

    watcher = watcher_result.unwrap()

    print(f"Watching for changes in {source_root.resolve()}")

    async with watcher:
        async for event_result in watcher:
            if isinstance(event_result, Err):
                event_result.print_diagnostic()
                continue

            event = event_result.unwrap()
            handled = await handle_event(event, 1500, store)
            if isinstance(handled, Err):
                handled.print_diagnostic()
            elif handled.value:
                print(f"Index updated: {event.path}")


def serve(port: int = 8000) -> None:
    """Run the bootstrap FastAPI server with its health endpoint."""
    import uvicorn

    from .server.app import create_app

    uvicorn.run(create_app(), host="127.0.0.1", port=port)


def build_help_command() -> Command:
    """Build the CLI-framework command tree used to render help text.

    Returns:
        The root application command.
    """
    root = Command(
        name="rag-against-the-machine",
        short="Local RAG system for the supplied vLLM repository.",
        example="uv run python -m rag_against_the_machine serve --port 8000",
    )
    _ = root.add_command(
        Command(
            name="serve",
            short="Run the bootstrap FastAPI server.",
            schema=ServeOptions,
        )
    )
    _ = root.add_command(
        Command(
            name="tmp",
            short="Temporary testing command.",
        )
    )

    return root


def render_framework_help(argv: list[str]) -> bool:
    """Render help with the bundled CLI framework when requested.

    Returns:
        Whether help was rendered.
    """
    if argv and "--help" not in argv and "-h" not in argv:
        return False

    root = build_help_command()
    if argv and argv[0] == "serve":
        root.commands["serve"].help()
    else:
        root.help()
    return True


def validate_framework_arguments(argv: list[str]) -> bool:
    """Validate command syntax with cli_fw without replacing Fire execution.

    Returns:
        Whether validation succeeded.
    """
    full_cli = [basename(sys.argv[0]), *argv]
    result = build_help_command().execute(argv, diagnostic_argv=full_cli)
    if isinstance(result, Err):
        result.print_diagnostic()
        return False
    return True


def run_with_fire() -> None:
    """Load Python Fire only once framework validation has succeeded."""
    import fire

    fire.Fire({"serve": serve, "tmp": tmp})


def main() -> None:
    """Render framework help and delegate command execution to Python Fire.

    Raises:
        SystemExit: When help or argument validation terminates the command.
    """
    argv = sys.argv[1:]
    if render_framework_help(argv):
        return
    if not validate_framework_arguments(argv):
        raise SystemExit(2)
    run_with_fire()


if __name__ == "__main__":
    main()
