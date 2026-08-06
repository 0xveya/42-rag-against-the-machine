"""Python Fire command-line entry point for the project scaffold."""

import sys
from dataclasses import dataclass, field
from os.path import basename
from pathlib import Path

from rag_against_the_machine.cli_commands import (
    AnswerDatasetOptions,
    AnswerOptions,
    EvaluateOptions,
    IndexOptions,
    SearchDatasetOptions,
    SearchOptions,
    answer,
    answer_dataset,
    evaluate,
    index,
    search,
    search_dataset,
)
from rag_against_the_machine.cli_fw import Command
from rag_against_the_machine.errors import Err, Ok, Some


@dataclass(frozen=True)
class ServeOptions:
    """Document the options accepted by the development server command."""

    port: int = field(
        default=8000,
        metadata={"help": "Port for the local API server (0-65535)."},
    )
    raw_directory: str = field(
        default="data/raw",
        metadata={"help": "Directory whose child folders are repositories."},
    )
    database_path: str = field(
        default="data/processed/index.db",
        metadata={"help": "SQLite search-index path."},
    )
    model: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "Transformers model used for streamed answers."},
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum tokens generated for each answer."},
    )


def _valid_port(port: int) -> bool:
    """Return whether *port* can be passed to a TCP bind call."""
    return 0 <= port <= 65535


def serve(
    port: int = 8000,
    raw_directory: str = "data/raw",
    database_path: str = "data/processed/index.db",
    model: str = "Qwen/Qwen3-0.6B",
    max_new_tokens: int = 256,
) -> None:
    """Run the FastAPI UI, WebSocket API, and live index watcher."""
    if not _valid_port(port):
        raise ValueError(
            f"invalid port {port}: expected a value from 0 to 65535"
        )
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

    import uvicorn

    from .server.app import create_app
    from .server.service import WebRagService

    service = WebRagService(
        raw_directory=Path(raw_directory),
        database_path=Path(database_path),
        model_name=model,
        max_new_tokens=max_new_tokens,
    )
    uvicorn.run(create_app(Some(service)), host="127.0.0.1", port=port)


def build_help_command() -> Command:
    """Build the CLI-framework command tree used to render help text.

    Returns:
        The root application command.
    """
    root = Command(
        name="rag-against-the-machine",
        short="Local RAG system for the supplied codebase.",
        example="uv run python -m src serve --port 8000",
    )
    _ = root.add_command(
        Command(
            name="serve",
            short="Run the learning UI and WebSocket API.",
            schema=ServeOptions,
        )
    )
    for name, short, schema in (
        ("index", "Index source files.", IndexOptions),
        ("search", "Search one question.", SearchOptions),
        ("search_dataset", "Search a RAG dataset.", SearchDatasetOptions),
        ("answer", "Answer one question.", AnswerOptions),
        (
            "answer_dataset",
            "Answer a search-results dataset.",
            AnswerDatasetOptions,
        ),
        ("evaluate", "Evaluate recall@k.", EvaluateOptions),
    ):
        _ = root.add_command(Command(name=name, short=short, schema=schema))

    return root


def render_framework_help(argv: list[str]) -> bool:
    """Render help with the bundled CLI framework when requested.

    Returns:
        Whether help was rendered.
    """
    if argv and "--help" not in argv and "-h" not in argv:
        return False

    root = build_help_command()
    if argv and argv[0] in root.commands:
        root.commands[argv[0]].help()
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
    if isinstance(result, Ok) and isinstance(result.value, ServeOptions):
        if not _valid_port(result.value.port):
            print(
                f"Invalid port {result.value.port}: --port must be between "
                "0 and 65535.",
                file=sys.stderr,
            )
            return False
        if result.value.max_new_tokens <= 0:
            print(
                "Invalid --max_new_tokens: value must be greater than zero.",
                file=sys.stderr,
            )
            return False
    return True


def run_with_fire() -> None:
    """Load Python Fire only once framework validation has succeeded."""
    import fire

    fire.Fire(
        {
            "serve": serve,
            "index": index,
            "search": search,
            "search_dataset": search_dataset,
            "answer": answer,
            "answer_dataset": answer_dataset,
            "evaluate": evaluate,
        }
    )


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
    try:
        run_with_fire()
    except KeyboardInterrupt:
        print("\nServer stopped. Goodbye!")


if __name__ == "__main__":
    main()
