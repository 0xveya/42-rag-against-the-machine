"""Python Fire command-line entry point for the project scaffold."""

import sys
from dataclasses import dataclass, field

import fire
import uvicorn

from .cli_fw import Command
from .server.app import create_app


@dataclass(frozen=True)
class ServeOptions:
    """Document the options accepted by the development server command."""

    port: int = field(default=8000, metadata={"help": "Port for the local API server."})


def serve(port: int = 8000) -> None:
    """Run the bootstrap FastAPI server with its health endpoint."""
    uvicorn.run(create_app(), host="127.0.0.1", port=port)


def build_help_command() -> Command:
    """Build the CLI-framework command tree used to render help text."""
    root = Command(
        name="rag-against-the-machine",
        short="Local RAG system for the supplied vLLM repository.",
        example="uv run python -m rag_against_the_machine serve --port 8000",
    )
    root.add_command(
        Command(
            name="serve",
            short="Run the bootstrap FastAPI server.",
            schema=ServeOptions,
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


def main() -> None:
    """Render framework help and delegate command execution to Python Fire."""
    if render_framework_help(sys.argv[1:]):
        return
    fire.Fire({"serve": serve})
