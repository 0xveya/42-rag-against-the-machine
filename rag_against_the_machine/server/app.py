"""Minimal FastAPI application factory for the optional REST API."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response schema for the bootstrap health endpoint."""

    status: Literal["ok"] = "ok"
    service: str = "rag-against-the-machine"


def create_app() -> FastAPI:
    """Create the unconfigured API application.

    Search and answer routes will be attached when RagService is implemented.
    """
    app = FastAPI(title="rag-against-the-machine")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report that the scaffold server is running."""
        return HealthResponse()

    return app
