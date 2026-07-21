"""Shared result and error types for Python CLI and parsing utilities."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, Generic, TypeAlias, TypeVar, Any, NoReturn, Callable


class CliError(Enum):
    """Enumerate failure modes encountered during CLI argument parsing."""

    UNKNOWN_ARGUMENT = auto()
    MISSING_REQUIRED_ARGUMENT = auto()
    INVALID_CHOICE = auto()
    INVALID_ARGUMENT_TYPE = auto()
    MISSING_ARGUMENT_VALUE = auto()


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

    def __init__(self, err_payload: "Err[Any]"):
        super().__init__(f"BubbleUpError: {err_payload.error}")
        self.err_payload = err_payload


def catch_bubble(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to catch bubbled errors and return them cleanly as an Err."""

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
        """Panics and prints the diagnostic error context."""
        self.print_diagnostic()
        raise ValueError(f"Called `Result::unwrap()` on an `Err` value: {self.error.name}")

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
            print(f"   {RED}╰─▶{RESET} {err_name_str.replace('_', ' ').title()}")
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
        """Rust-like ? operator. Raises BubbleUpError to bubble the Err up."""
        raise BubbleUpError(self)


Result: TypeAlias = Ok[T] | Err[E]
