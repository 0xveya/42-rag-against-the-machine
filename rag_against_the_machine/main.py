"""Python Fire command-line entry point for the project scaffold."""

import sys
from dataclasses import dataclass, field
from os.path import basename

from .cli_fw import Command
from .errors import Err


@dataclass(frozen=True)
class ServeOptions:
    """Document the options accepted by the development server command."""

    port: int = field(default=8000, metadata={"help": "Port for the local API server."})


def tmp() -> None:
    import asyncio

    asyncio.run(_tmp())


async def _tmp() -> None:
    """Temporary testing command."""
    from pathlib import Path

    from .errors import Err
    from .indexing.discovery import discover_files
    from .indexing.pipeline import run_pipeline
    from .storage.db import Store, Transaction

    store = Store(Path("data/output/stuff.db"))
    store.init()

    def show_tables(tx: Transaction) -> None:
        rows = tx.conn.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
            ORDER BY name
            """
        ).fetchall()

        for row in rows:
            print(f"{row['type']}: {row['name']}")

    store.with_tx(show_tables)

    discovered = discover_files(
        source_root=Path("data/raw/gns3util"),
        project_root=Path.cwd(),
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
    print(f"Indexed {output.files_processed} files; skipped {output.files_skipped}.")
    for diagnostic in output.diagnostics:
        print(f"warning: {diagnostic.filename}: {diagnostic.help_msg}")
    # print(output)
    # for file in files:
    #     # print(file)
    #     content = read_source_file(file).unwrap()
    #     chunks = chunk_source_file(file, content, 1_500).unwrap()
    #     validate_chunks(content, chunks, 1500)
    #     # print(json.dumps([asdict(x) for x in chunks], separators=(",", ":")))


def serve(port: int = 8000) -> None:
    """Run the bootstrap FastAPI server with its health endpoint."""
    import uvicorn

    from .server.app import create_app

    uvicorn.run(create_app(), host="127.0.0.1", port=port)


def build_help_command() -> Command:
    """Build the CLI-framework command tree used to render help text."""
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
    """Render help with the bundled CLI framework when the user requests it."""
    if argv and "--help" not in argv and "-h" not in argv:
        return False

    root = build_help_command()
    if argv and argv[0] == "serve":
        root.commands["serve"].help()
    else:
        root.help()
    return True


def validate_framework_arguments(argv: list[str]) -> bool:
    """Validate command syntax with cli_fw without replacing Fire execution."""
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
    """Render framework help and delegate command execution to Python Fire."""
    argv = sys.argv[1:]
    if render_framework_help(argv):
        return
    if not validate_framework_arguments(argv):
        raise SystemExit(2)
    run_with_fire()


if __name__ == "__main__":
    main()
