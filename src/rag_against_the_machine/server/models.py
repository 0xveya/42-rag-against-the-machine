"""Pydantic messages exchanged by the learning web API."""

from typing import Literal

from pydantic import BaseModel, Field


class RepositorySummary(BaseModel):
    """One repository selectable by the browser."""

    repository_id: str
    name: str
    path: str


class ReindexResponse(BaseModel):
    """Summarize a repository reindex requested by the browser."""

    repository_id: str
    files_discovered: int
    files_processed: int
    files_skipped: int


class CommandRequest(BaseModel):
    """Request one allowlisted CLI or evaluator command."""

    command: Literal[
        "index",
        "search",
        "search_dataset",
        "answer",
        "answer_dataset",
        "evaluate",
        "moulinette",
    ]
    arguments: list[str] = Field(default_factory=list)


class CommandResponse(BaseModel):
    """Return captured output from an allowlisted command."""

    command: str
    exit_code: int
    stdout: str
    stderr: str


class AddRepositoryRequest(BaseModel):
    """Request cloning one repository below data/raw."""

    url: str = Field(min_length=1)
    name: str = Field(default="", pattern=r"^[A-Za-z0-9._-]*$")


class QuestionRequest(BaseModel):
    """Question received over the WebSocket."""

    repository: str
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    inspect: bool = False
    generate: bool = True


class RetrievedSource(BaseModel):
    """Retrieved source with private text available to generation."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str = Field(exclude=True)
    score: float = Field(default=0.0, exclude=True)


class IndexStatus(BaseModel):
    """Counts for a selectable repository's database records."""

    repository: str
    files: int
    chunks: int


class InspectedSource(BaseModel):
    """Exact indexed text and its original SQLite BM25 score."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    score: float


class PromptMessage(BaseModel):
    """One message passed to the model's chat template."""

    role: str
    content: str


class InspectionMessage(BaseModel):
    """Expose the actual retrieval and source prompt for the explorer."""

    type: Literal["inspection"] = "inspection"
    query: str
    sources: list[InspectedSource]
    messages: list[PromptMessage]


class StatusMessage(BaseModel):
    """Report the current server-side stage."""

    type: Literal["status"] = "status"
    data: Literal["retrieving", "generating"]


class SourcesMessage(BaseModel):
    """Send retrieved source locations to the browser."""

    type: Literal["sources"] = "sources"
    data: list[RetrievedSource]


class TokenMessage(BaseModel):
    """Send one decoded generation fragment."""

    type: Literal["token"] = "token"
    data: str


class DoneMessage(BaseModel):
    """Mark completion of one request."""

    type: Literal["done"] = "done"


class ErrorMessage(BaseModel):
    """Report a recoverable request failure."""

    type: Literal["error"] = "error"
    data: str
