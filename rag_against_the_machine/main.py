"""Python Fire command-line entry point for the project scaffold."""

import fire
import uvicorn

from .server.app import create_app


def serve(port: int = 8000) -> None:
    """Run the bootstrap FastAPI server with its health endpoint."""
    uvicorn.run(create_app(), host="127.0.0.1", port=port)


def main() -> None:
    """Expose the scaffold commands through Python Fire."""
    fire.Fire({"serve": serve})
