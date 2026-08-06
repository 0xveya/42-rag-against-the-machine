"""Thin, non-recursive bindings for the Linux inotify syscalls."""

from __future__ import annotations

import ctypes
import os
import select
import struct
from dataclasses import dataclass
from pathlib import Path

from rag_against_the_machine.errors import (
    Err,
    InotifyError,
    Nothing,
    Ok,
    Option,
    Result,
    Some,
)

libc = ctypes.CDLL(None, use_errno=True)

libc.inotify_init1.argtypes = [ctypes.c_int]
libc.inotify_init1.restype = ctypes.c_int
libc.inotify_add_watch.argtypes = [
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_uint32,
]
libc.inotify_add_watch.restype = ctypes.c_int
libc.inotify_rm_watch.argtypes = [ctypes.c_int, ctypes.c_int]
libc.inotify_rm_watch.restype = ctypes.c_int

_NO_PATH: Option[Path] = Nothing()

IN_ACCESS = 0x00000001
IN_MODIFY = 0x00000002
IN_ATTRIB = 0x00000004
IN_CLOSE_WRITE = 0x00000008
IN_CLOSE_NOWRITE = 0x00000010
IN_OPEN = 0x00000020
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
IN_IGNORED = 0x00008000
IN_Q_OVERFLOW = 0x00004000
IN_ISDIR = 0x40000000

DEFAULT_MASK = (
    IN_CREATE
    | IN_MODIFY
    | IN_CLOSE_WRITE
    | IN_DELETE
    | IN_MOVED_FROM
    | IN_MOVED_TO
    | IN_DELETE_SELF
    | IN_MOVE_SELF
)

# struct inotify_event: int wd; uint32_t mask, cookie, len; char name[]
_EVENT_HEADER = struct.Struct("iIII")


@dataclass(frozen=True)
class RawEvent:
    """Represent one event decoded directly from the kernel buffer."""

    watch_descriptor: int
    mask: int
    cookie: int
    name: str

    @property
    def is_directory(self) -> bool:
        """Whether the kernel marked the event target as a directory."""
        return bool(self.mask & IN_ISDIR)


def _errno_err(
    error: InotifyError,
    operation: str,
    *,
    path: Option[Path] = _NO_PATH,
) -> Err[InotifyError]:
    """Build a categorized backend error without raising an OSError.

    Returns:
        An error containing the current C errno and operation context.
    """
    error_number = ctypes.get_errno()
    path_suffix = f" for {path.value}" if isinstance(path, Some) else ""
    return Err(
        error,
        context_msg=(
            f"{operation}{path_suffix} failed: [errno {error_number}] {os.strerror(error_number)}"
        ),
        namespace="inotify",
    )


def _closed_err(operation: str) -> Err[InotifyError]:
    return Err(
        InotifyError.INSTANCE_CLOSED,
        context_msg=f"Cannot {operation} on a closed inotify instance",
        namespace="inotify",
    )


class Inotify:
    """One inotify instance; directory recursion belongs to a higher layer."""

    def __init__(self, fd: int) -> None:
        """Wrap an initialized inotify file descriptor."""
        self.fd = fd
        self._closed = False

    @classmethod
    def open(cls) -> Result[Inotify, InotifyError]:
        """Create a nonblocking close-on-exec inotify instance.

        Returns:
            A backend instance, or an initialization error.
        """
        flags = os.O_NONBLOCK | os.O_CLOEXEC
        fd = libc.inotify_init1(flags)
        if fd == -1:
            return _errno_err(InotifyError.INITIALIZATION_FAILED, "inotify_init1")
        return Ok(cls(fd))

    def add_watch(
        self,
        path: str | os.PathLike[str],
        mask: int = DEFAULT_MASK,
    ) -> Result[int, InotifyError]:
        """Register ``path`` with the requested event mask.

        Returns:
            The kernel watch descriptor, or a categorized backend error.
        """
        if self._closed:
            return _closed_err("add a watch")

        try:
            normalized_path = Path(path)
            encoded_path = os.fsencode(os.fspath(normalized_path))
        except (OSError, TypeError, ValueError):
            return Err(
                InotifyError.WATCH_ADD_FAILED,
                context_msg=f"inotify_add_watch for {path!s} failed: invalid path",
                namespace="inotify",
            )

        watch_descriptor = libc.inotify_add_watch(self.fd, encoded_path, mask)
        if watch_descriptor == -1:
            return _errno_err(
                InotifyError.WATCH_ADD_FAILED,
                "inotify_add_watch",
                path=Some(normalized_path),
            )
        return Ok(watch_descriptor)

    def remove_watch(self, watch_descriptor: int) -> Result[None, InotifyError]:
        """Remove a kernel watch descriptor.

        Returns:
            Success, or a categorized backend error.
        """
        if self._closed:
            return _closed_err("remove a watch")

        result = libc.inotify_rm_watch(self.fd, watch_descriptor)
        if result == -1:
            return _errno_err(InotifyError.WATCH_REMOVE_FAILED, "inotify_rm_watch")
        return Ok(None)

    def read(self) -> Result[list[RawEvent], InotifyError]:
        """Decode all currently readable kernel events without blocking.

        Returns:
            Decoded raw events, or a read or buffer error.
        """
        if self._closed:
            return _closed_err("read")

        try:
            data = os.read(self.fd, 64 * 1024)
        except (BlockingIOError, InterruptedError):
            return Ok([])
        except OSError:
            return _errno_err(InotifyError.READ_FAILED, "read")

        events: list[RawEvent] = []
        offset = 0
        while offset < len(data):
            remaining = len(data) - offset
            if remaining < _EVENT_HEADER.size:
                return Err(
                    InotifyError.MALFORMED_EVENT_BUFFER,
                    context_msg=(
                        f"Only {remaining} bytes remain, but an event header "
                        f"requires {_EVENT_HEADER.size}"
                    ),
                    namespace="inotify",
                )

            wd, mask, cookie, name_length = _EVENT_HEADER.unpack_from(data, offset)
            offset += _EVENT_HEADER.size
            if name_length > len(data) - offset:
                return Err(
                    InotifyError.MALFORMED_EVENT_BUFFER,
                    context_msg=(
                        f"Event declares a {name_length}-byte name, but only "
                        f"{len(data) - offset} bytes remain"
                    ),
                    namespace="inotify",
                )

            raw_name = data[offset : offset + name_length]
            offset += name_length
            events.append(
                RawEvent(
                    watch_descriptor=wd,
                    mask=mask,
                    cookie=cookie,
                    name=os.fsdecode(raw_name.rstrip(b"\0")),
                )
            )

        return Ok(events)

    def poll(self, timeout: int = -1) -> Result[list[tuple[int, int]], InotifyError]:
        """Wait for readiness, returning poll records rather than raising.

        Returns:
            Poll records, or a categorized polling error.
        """
        if self._closed:
            return _closed_err("poll")

        poller = select.poll()
        try:
            poller.register(self.fd, select.POLLIN | select.POLLERR | select.POLLHUP)
            return Ok(poller.poll(timeout))
        except InterruptedError:
            return Ok([])
        except OSError:
            return _errno_err(InotifyError.POLL_FAILED, "poll")

    def close(self) -> None:
        """Close the file descriptor idempotently."""
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self.fd)
        except OSError:
            pass

    def __enter__(self) -> Inotify:
        """Return this backend for synchronous context management."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the backend when leaving its context."""
        self.close()
