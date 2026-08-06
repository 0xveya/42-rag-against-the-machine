"""FastAPI application serving the learning UI and WebSocket scaffold."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, cast

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from rag_against_the_machine.errors import Err, Nothing, Option, Some, WebError
from rag_against_the_machine.server.actions import add_repository, run_command
from rag_against_the_machine.server.models import (
    AddRepositoryRequest,
    CommandRequest,
    CommandResponse,
    DoneMessage,
    ErrorMessage,
    QuestionRequest,
    ReindexResponse,
    RepositorySummary,
    SourcesMessage,
    StatusMessage,
    TokenMessage,
)
from rag_against_the_machine.server.service import WebRagService

ROOT = Path(__file__).resolve().parents[3]
WEB_DIRECTORY = ROOT / "web"
_DEFAULT_SERVICE: Option[WebRagService] = Nothing()


class HealthResponse(BaseModel):
    """Response schema for the health endpoint."""

    status: Literal["ok"] = "ok"
    service: str = "rag-against-the-machine"


def configure_logging() -> None:
    """Configure compact structured console logs for local development."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )


def _error_response(
    error: Err[WebError], status_code: int = 400
) -> JSONResponse:
    """Convert a project web error into a small HTTP error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error.error.name,
            "detail": error.context_msg or "Web operation failed",
        },
    )


def create_app(service: Option[WebRagService] = _DEFAULT_SERVICE) -> FastAPI:
    """Create the API and bind service lifetime to the application."""
    configure_logging()
    rag_service = (
        service.value if isinstance(service, Some) else WebRagService()
    )
    log = structlog.get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.rag = rag_service
        started = await rag_service.start()
        if isinstance(started, Err):
            await log.aerror(
                "api_start_degraded",
                reason=started.context_msg,
                error=started.error.name,
            )
        await log.ainfo("api_started")
        try:
            yield
        finally:
            await rag_service.close()
            await log.ainfo("api_stopped")

    app = FastAPI(title="rag-against-the-machine", lifespan=lifespan)

    @app.get("/", response_class=FileResponse)
    async def frontend() -> FileResponse:
        """Serve the intentionally unstyled learning page."""
        return FileResponse(WEB_DIRECTORY / "index.html")

    @app.get("/eval.html", response_class=FileResponse)
    async def evaluation_frontend() -> FileResponse:
        """Serve the raw HTML command and evaluation page."""
        return FileResponse(WEB_DIRECTORY / "eval.html")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Report that the scaffold server is running."""
        return HealthResponse()

    @app.get("/api/repositories", response_model=list[RepositorySummary])
    async def repositories() -> list[RepositorySummary]:
        """Return repository choices for the HTML select element."""
        rag = cast(WebRagService, app.state.rag)
        return rag.list_repositories()

    @app.post(
        "/api/repositories/{repository}/reindex",
        response_model=ReindexResponse,
    )
    async def reindex_repository(
        repository: str,
    ) -> ReindexResponse | JSONResponse:
        """Force-rebuild the selected repository's search index."""
        rag = cast(WebRagService, app.state.rag)
        result = await rag.reindex(repository)
        if isinstance(result, Err):
            status_code = (
                404 if result.error is WebError.INVALID_REPOSITORY else 500
            )
            return _error_response(result, status_code)
        return result.value

    @app.post("/api/commands", response_model=CommandResponse)
    async def command(
        request: CommandRequest,
    ) -> CommandResponse | JSONResponse:
        """Run one allowlisted CLI command from the evaluation page."""
        result = await run_command(request)
        if isinstance(result, Err):
            return _error_response(result)
        return result.value

    @app.post("/api/repositories", response_model=CommandResponse)
    async def clone_repository(
        request: AddRepositoryRequest,
    ) -> CommandResponse | JSONResponse:
        """Clone one repository directly below data/raw."""
        rag = cast(WebRagService, app.state.rag)
        result = await add_repository(request, rag.raw_directory)
        if isinstance(result, Err):
            return _error_response(result)
        return result.value

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """Demonstrate typed retrieval and streamed response messages."""
        await websocket.accept()
        rag = cast(WebRagService, app.state.rag)
        try:
            while True:
                try:
                    request = QuestionRequest.model_validate(
                        await websocket.receive_json()
                    )
                    await websocket.send_json(
                        StatusMessage(data="retrieving").model_dump()
                    )
                    retrieved = await rag.retrieve(
                        request.repository, request.question, request.k
                    )
                    if isinstance(retrieved, Err):
                        await websocket.send_json(
                            ErrorMessage(
                                data=retrieved.context_msg
                                or "Retrieval failed"
                            ).model_dump()
                        )
                        continue
                    sources = retrieved.value
                    await websocket.send_json(
                        SourcesMessage(data=sources).model_dump()
                    )
                    await websocket.send_json(
                        StatusMessage(data="generating").model_dump()
                    )
                    generation_failed = False
                    async for fragment in rag.stream_answer(
                        request.question, sources
                    ):
                        if isinstance(fragment, Err):
                            generation_failed = True
                            await websocket.send_json(
                                ErrorMessage(
                                    data=fragment.context_msg
                                    or "Generation failed"
                                ).model_dump()
                            )
                            break
                        await websocket.send_json(
                            TokenMessage(data=fragment.value).model_dump()
                        )
                    if not generation_failed:
                        await websocket.send_json(DoneMessage().model_dump())
                except (
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    ValidationError,
                ) as error:
                    await log.awarning(
                        "websocket_request_rejected", reason=str(error)
                    )
                    await websocket.send_json(
                        ErrorMessage(data=str(error)).model_dump()
                    )
        except WebSocketDisconnect:
            await log.ainfo("websocket_disconnected")

    return app
