"""Shared result and error types for Python CLI and parsing utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from typing import (
    Any,
    ClassVar,
    Generic,
    NoReturn,
    ParamSpec,
    TypeAlias,
    TypeVar,
)


class CliError(Enum):
    """Enumerate failure modes encountered during CLI argument parsing."""

    UNKNOWN_ARGUMENT = auto()
    MISSING_REQUIRED_ARGUMENT = auto()
    INVALID_CHOICE = auto()
    INVALID_ARGUMENT_TYPE = auto()
    MISSING_ARGUMENT_VALUE = auto()


class DiscoveryError(Enum):
    """Enumerate failure modes encountered during file discovery."""

    SOURCE_DOES_NOT_EXIST = auto()
    SOURCE_IS_NOT_A_DIRECTORY = auto()
    SOURCE_IS_NOT_READABLE = auto()
    SOURCE_OUTSIDE_PROJECT = auto()
    PROJECT_DOES_NOT_EXIST = auto()
    PROJECT_IS_NOT_A_DIRECTORY = auto()
    DIRECTORY_TRAVERSAL_FAILED = auto()


class ReadError(Enum):
    """Enumerate source-file reading failures."""

    FILE_NOT_FOUND = auto()
    FILE_NOT_READABLE = auto()
    FILE_IS_DIRECTORY = auto()
    FILE_DECODE_FAILED = auto()
    FILE_READ_FAILED = auto()


class ChunkingError(Enum):
    """Enumerate source-text chunking failures."""

    INVALID_MAX_CHUNK_SIZE = auto()
    PYTHON_PARSE_FAILED = auto()
    MARKDOWN_PARSE_FAILED = auto()
    INVALID_CHARACTER_RANGE = auto()
    CHUNK_TOO_LARGE = auto()
    CHUNKING_FAILED = auto()


class IndexingError(Enum):
    """Enumerate index persistence failures."""

    INDEX_WRITE_FAILED = auto()
    INDEX_SAVE_FAILED = auto()


class PipelineError(Enum):
    """Enumerate asynchronous indexing pipeline failures."""

    DISCOVERY_FAILED = auto()
    TASK_FAILED = auto()
    DATABASE_FAILED = auto()
    CANCELLED = auto()
    EMPTY_INPUT = auto()


class FileProcessingStage(Enum):
    """Identify a stage of source-file processing."""

    READ = auto()
    CHUNK = auto()


class FileProcessingError(Enum):
    """Failure categories returned when processing one discovered file."""

    READ_FAILED = auto()
    CHUNK_FAILED = auto()


class InotifyError(Enum):
    """Enumerate low-level inotify backend failures."""

    INITIALIZATION_FAILED = auto()
    WATCH_ADD_FAILED = auto()
    WATCH_REMOVE_FAILED = auto()
    READ_FAILED = auto()
    POLL_FAILED = auto()
    INSTANCE_CLOSED = auto()
    MALFORMED_EVENT_BUFFER = auto()


class WatchError(Enum):
    """Enumerate normalized watcher frontend failures."""

    ROOT_NOT_FOUND = auto()
    ROOT_NOT_DIRECTORY = auto()
    ROOT_NOT_READABLE = auto()

    INITIAL_SCAN_FAILED = auto()
    WATCH_REGISTRATION_FAILED = auto()

    EVENT_READ_FAILED = auto()
    EVENT_QUEUE_OVERFLOW = auto()
    UNKNOWN_WATCH_DESCRIPTOR = auto()

    WATCHER_NOT_STARTED = auto()
    WATCHER_CLOSED = auto()

    EVENT_LOOP_UNAVAILABLE = auto()
    EVENT_LOOP_REGISTRATION_FAILED = auto()

    RECEIVE_TIMEOUT = auto()
    PUBLIC_QUEUE_FULL = auto()

    POLL_FAILED = auto()


class StorageError(Enum):
    """Enumerate failures exposed by the SQLite storage API."""

    DIRECTORY_CREATION_FAILED = auto()
    CONNECTION_FAILED = auto()
    QUERY_FAILED = auto()
    INVALID_QUERY_RESULT = auto()
    TRANSACTION_FAILED = auto()
    OPERATION_FAILED = auto()


class GenerationError(Enum):
    """Enumerate failures exposed by local model backends."""

    MODEL_LOAD_FAILED = auto()
    ANSWER_FAILED = auto()


E = TypeVar("E", bound=Enum)
T = TypeVar("T")


@dataclass(frozen=True)
class Diagnostic:
    """Stores the error location and context for diagnostic reporting."""

    filename: str
    line_num: int
    line_text: str
    col_start: int
    col_end: int
    help_msg: str | None = None


class BubbleUpError(Exception):
    """Internal exception to bubble Err results up to a catch_bubble decorator."""

    def __init__(self, err_payload: Err[Any]):
        """Store the error being propagated."""
        super().__init__(f"BubbleUpError: {err_payload.error}")
        self.err_payload = err_payload


def catch_bubble(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a function to convert bubbled errors into return values.

    Returns:
        A wrapper that catches and returns bubbled error payloads.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except BubbleUpError as b:
            return b.err_payload

    return wrapper


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Wrap a successful result value of type T."""

    value: T

    def unwrap(self) -> T:
        """Return the contained Ok value."""
        return self.value

    @property
    def q(self) -> T:
        """Rust-like ? operator. Returns the Ok value."""
        return self.value


@dataclass(frozen=True)
class Err(Generic[E]):
    """Wrap failure results with error variants and optional diagnostics."""

    error: E
    diagnostic: Diagnostic | None = None
    context_msg: str | None = None
    PROJECT_NAME: ClassVar[str] = "rag_against_the_machine"
    namespace: str | None = None

    def unwrap(self) -> NoReturn:
        """Print diagnostic context and reject unwrapping an error.

        Raises:
            ValueError: Always, because an error has no success value.
        """
        self.print_diagnostic()
        raise ValueError(
            f"Called `Result::unwrap()` on an `Err` value: {self.error.name}"
        )

    def print_diagnostic(self) -> None:
        """Prints a diagnostic message with dynamic caret alignment."""
        RED = "\033[1;31m"
        PINK = "\033[1;35m"
        BLUE = "\033[1;34m"
        CYAN = "\033[1;36m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        if self.namespace:
            sub_ns = self.namespace.lower().strip(":")
        else:
            raw_classname = self.error.__class__.__name__
            sub_ns = raw_classname.removesuffix("Error").lower()

        err_name_str = str(self.error.name).lower().replace("_", "::")
        full_namespace = f"{self.PROJECT_NAME}::{sub_ns}::{err_name_str}"

        print(f"{BOLD}Error:{RESET} {PINK}{full_namespace}{RESET}\n")

        if not self.diagnostic:
            print(f" {RED}×{RESET} {BOLD}Operation failed{RESET}")
            print(
                f"   {RED}╰─▶{RESET} {err_name_str.replace('_', ' ').title()}"
            )
            return

        d = self.diagnostic
        summary = self.context_msg or "Validation failed"
        print(f" {RED}×{RESET} {BOLD}{summary}{RESET}")
        print(
            f"   {BLUE}╭─[{RESET}{BOLD}{d.filename}:"
            f"{d.line_num}:{d.col_start + 1}{RESET}{BLUE}]{RESET}"
        )
        print(f"{d.line_num:2} {BLUE}│{RESET} {d.line_text}")

        hook_text = "╰─── "
        prefix = f"{RED}{hook_text}{BOLD}"
        carets = "^" * max(1, (d.col_end - d.col_start))
        hook_width = len(hook_text)

        if d.col_start >= hook_width:
            padding = " " * (d.col_start - hook_width)
            print(f"   {BLUE}·{RESET} {padding}{prefix}{carets}{RESET}")
        else:
            padding = " " * d.col_start
            print(f"   {BLUE}·{RESET} {padding}{RED}{carets}{RESET}")

        if d.help_msg:
            print(f"\n   {CYAN}help:{RESET} {d.help_msg}")

    @property
    def q(self) -> NoReturn:
        """Propagate this error to a ``catch_bubble`` wrapper.

        Raises:
            BubbleUpError: Always, carrying this error result.
        """
        raise BubbleUpError(self)


Result: TypeAlias = Ok[T] | Err[E]

P = ParamSpec("P")


class BubbleUpNothing(Exception):
    """Internal exception used to bubble Nothing through a decorator."""

    nothing: Nothing

    def __init__(self, nothing: Nothing) -> None:
        """Store the absent option being propagated."""
        super().__init__("Attempted to bubble up Nothing")
        self.nothing = nothing


@dataclass(frozen=True)
class Some(Generic[T]):
    """Wrap a present optional value."""

    value: T

    def unwrap(self) -> T:
        """Return the contained value."""
        return self.value

    def unwrap_or(self, _default: T) -> T:
        """Return the contained value."""
        return self.value

    def expect(self, _message: str) -> T:
        """Return the contained value."""
        return self.value

    @property
    def q(self) -> T:
        """The contained value used for propagation."""
        return self.value


@dataclass(frozen=True)
class Nothing:
    """Represent the absence of a value."""

    def unwrap(self) -> NoReturn:
        """Reject unwrapping an absent option.

        Raises:
            ValueError: Always, because no value is present.
        """
        raise ValueError("Called `Option::unwrap()` on a `Nothing` value")

    def unwrap_or(self, default: T) -> T:
        """Return the provided default value."""
        return default

    def expect(self, message: str) -> NoReturn:
        """Reject absence with a caller-provided explanation.

        Raises:
            ValueError: Always, using ``message``.
        """
        raise ValueError(message)

    @property
    def q(self) -> NoReturn:
        """Propagate absence to a ``catch_nothing`` wrapper.

        Raises:
            BubbleUpNothing: Always, carrying this absent option.
        """
        raise BubbleUpNothing(self)


Option: TypeAlias = Some[T] | Nothing


def catch_nothing(
    func: Callable[P, Option[T]],
) -> Callable[P, Option[T]]:
    """Convert bubbled Nothing values into returned Nothing values.

    Returns:
        A wrapper that catches propagated absence.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Option[T]:
        try:
            return func(*args, **kwargs)
        except BubbleUpNothing as bubbled:
            return bubbled.nothing

    return wrapper
