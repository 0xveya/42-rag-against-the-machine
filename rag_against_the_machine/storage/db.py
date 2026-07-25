"""SQLite database operations"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar, cast

from rag_against_the_machine.errors import Option, Nothing, Some


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SourceFileRecord:
    id: int
    path: str
    file_type: str
    size_bytes: int
    modified_at_ns: int
    content_hash: str
    max_chunk_size: int
    chunker_version: int
    indexed_at_ns: int


@dataclass(frozen=True, slots=True)
class ChunkInsert:
    chunk_index: int
    text: str
    start_character: int
    end_character: int
    created_at_ns: int


class Queries:
    """Database queries bound to one connection."""

    conn: sqlite3.Connection

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get_source_file(self, path: str) -> Option[SourceFileRecord]:
        row = cast(
            sqlite3.Row | None,
            self.conn.execute(
                """
                SELECT
                    id,
                    path,
                    file_type,
                    size_bytes,
                    modified_at_ns,
                    content_hash,
                    max_chunk_size,
                    chunker_version,
                    indexed_at_ns
                FROM source_files
                WHERE path = ?
                """,
                (path,),
            ).fetchone(),
        )

        if row is None:
            return Nothing()

        record = SourceFileRecord(
            id=cast(int, row["id"]),
            path=cast(str, row["path"]),
            file_type=cast(str, row["file_type"]),
            size_bytes=cast(int, row["size_bytes"]),
            modified_at_ns=cast(int, row["modified_at_ns"]),
            content_hash=cast(str, row["content_hash"]),
            max_chunk_size=cast(int, row["max_chunk_size"]),
            chunker_version=cast(int, row["chunker_version"]),
            indexed_at_ns=cast(int, row["indexed_at_ns"]),
        )

        return Some(record)

    def upsert_source_file(
        self,
        *,
        path: str,
        file_type: str,
        size_bytes: int,
        modified_at_ns: int,
        content_hash: str,
        max_chunk_size: int,
        chunker_version: int,
        indexed_at_ns: int,
    ) -> Option[int]:
        row = cast(
            sqlite3.Row | None,
            self.conn.execute(
                """
                INSERT INTO source_files (
                    path,
                    file_type,
                    size_bytes,
                    modified_at_ns,
                    content_hash,
                    max_chunk_size,
                    chunker_version,
                    indexed_at_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    file_type = excluded.file_type,
                    size_bytes = excluded.size_bytes,
                    modified_at_ns = excluded.modified_at_ns,
                    content_hash = excluded.content_hash,
                    max_chunk_size = excluded.max_chunk_size,
                    chunker_version = excluded.chunker_version,
                    indexed_at_ns = excluded.indexed_at_ns
                RETURNING id
                """,
                (
                    path,
                    file_type,
                    size_bytes,
                    modified_at_ns,
                    content_hash,
                    max_chunk_size,
                    chunker_version,
                    indexed_at_ns,
                ),
            ).fetchone(),
        )

        if row is None:
            raise RuntimeError(
                "Source-file upsert completed without returning an id"
            )

        return Some(cast(int, row["id"]))

    def delete_chunks_for_source(self, source_file_id: int) -> None:
        _ = self.conn.execute(
            """
            DELETE FROM chunks
            WHERE source_file_id = ?
            """,
            (source_file_id,),
        )

    def insert_chunks(
        self,
        source_file_id: int,
        chunks: list[ChunkInsert],
    ) -> None:
        _ = self.conn.executemany(
            """
            INSERT INTO chunks (
                source_file_id,
                chunk_index,
                text,
                start_character,
                end_character,
                created_at_ns
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    source_file_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.start_character,
                    chunk.end_character,
                    chunk.created_at_ns,
                )
                for chunk in chunks
            ],
        )

    def delete_source_file(self, source_file_id: int) -> None:
        _ = self.conn.execute(
            """
            DELETE FROM source_files
            WHERE id = ?
            """,
            (source_file_id,),
        )


class Transaction:
    """Transaction-scoped typed queries and raw connection access."""

    conn: sqlite3.Connection
    queries: Queries

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.queries = Queries(conn)


class Store:
    """SQLite-backed application storage."""

    db_path: Path

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def init(self) -> None:
        """Create the database directory and initialize its schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        def initialize(tx: Transaction) -> None:
            _ = tx.conn.executescript(_SCHEMA)

        self.with_tx(initialize)

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Open and configure one SQLite connection."""
        conn: sqlite3.Connection = sqlite3.connect(
            self.db_path,
            timeout=5.0,
        )
        conn.row_factory = sqlite3.Row

        try:
            _ = conn.execute("pragma foreign_keys = on")
            _ = conn.execute("pragma journal_mode = wal")
            _ = conn.execute("pragma synchronous = normal")
            _ = conn.execute("pragma busy_timeout = 5000")
            yield conn
        finally:
            conn.close()

    def with_tx(
        self,
        operation: Callable[[Transaction], T],
    ) -> T:
        """Execute raw SQL and typed queries in one transaction."""
        with self.connect() as conn:
            try:
                _ = conn.execute("begin")
                result = operation(Transaction(conn))
            except Exception:
                conn.rollback()
                raise
            else:
                conn.commit()
                return result


# normally i would use am embdded migrations dir to handle the db but with the scope of this projects its okay to just have the schma as a string
_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    file_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_at_ns INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    max_chunk_size INTEGER NOT NULL,
    chunker_version INTEGER NOT NULL DEFAULT 1,
    indexed_at_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_character INTEGER NOT NULL,
    end_character INTEGER NOT NULL,
    created_at_ns INTEGER NOT NULL,
    FOREIGN KEY (source_file_id) REFERENCES source_files(id) ON DELETE CASCADE,
    UNIQUE (source_file_id, chunk_index),
    CHECK (chunk_index >= 0),
    CHECK (start_character >= 0),
    CHECK (end_character >= start_character)
);

CREATE INDEX IF NOT EXISTS idx_chunks_source_file_id ON chunks(source_file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    source_file_id UNINDEXED,
    content = 'chunks',
    content_rowid = 'id',
    tokenize = "unicode61 tokenchars '_'"
);

CREATE TRIGGER IF NOT EXISTS chunks_after_insert
AFTER
INSERT
    ON chunks
BEGIN
INSERT INTO
    chunks_fts (
        rowid,
        text,
        source_file_id
    )
VALUES
    (
        new.id,
        new.text,
        new.source_file_id
    );

END;

CREATE TRIGGER IF NOT EXISTS chunks_after_delete
AFTER
    DELETE ON chunks
BEGIN
INSERT INTO
    chunks_fts (
        chunks_fts,
        rowid,
        text,
        source_file_id
    )
VALUES
    (
        'delete',
        old.id,
        old.text,
        old.source_file_id
    );

END;

CREATE TRIGGER IF NOT EXISTS chunks_after_update
AFTER
UPDATE
    ON chunks
BEGIN
INSERT INTO
    chunks_fts (
        chunks_fts,
        rowid,
        text,
        source_file_id
    )
VALUES
    (
        'delete',
        old.id,
        old.text,
        old.source_file_id
    );

INSERT INTO
    chunks_fts (
        rowid,
        text,
        source_file_id
    )
VALUES
    (
        new.id,
        new.text,
        new.source_file_id
    );

END;
"""
