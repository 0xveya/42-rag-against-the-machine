"""Exception-free SQLite database operations."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from rag_against_the_machine.errors import (
    Err,
    Nothing,
    Ok,
    Option,
    Result,
    Some,
    StorageError,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked chunk returned by full-text search."""

    chunk_id: int
    file_path: str
    start_character: int
    end_character: int
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class SourceFileRecord:
    """Represent one persisted source-file row."""

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
    """Describe one chunk to insert for a source file."""

    chunk_index: int
    text: str
    start_character: int
    end_character: int
    created_at_ns: int


class Queries:
    """Database queries bound to one connection."""

    conn: sqlite3.Connection

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind typed queries to ``conn``."""
        self.conn = conn

    def get_source_file(
        self, path: str
    ) -> Result[Option[SourceFileRecord], StorageError]:
        """Return the source record for ``path`` when it exists."""
        try:
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
        except sqlite3.Error as error:
            return _query_error("get source file", error)

        if row is None:
            return Ok(Nothing())

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

        return Ok(Some(record))

    def search_chunks(
        self,
        query: str,
        limit: int,
        path_terms: tuple[str, ...] = (),
    ) -> Result[list[SearchHit], StorageError]:
        """Return the highest-ranking chunks for an FTS5 query."""
        if limit <= 0:
            return Ok([])

        try:
            rows = cast(
                list[sqlite3.Row],
                self.conn.execute(
                    """
                    SELECT
                        chunks.id AS chunk_id,
                        source_files.path AS file_path,
                        chunks.start_character,
                        chunks.end_character,
                        chunks.text,
                        bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks
                        ON chunks.id = chunks_fts.rowid
                    JOIN source_files
                        ON source_files.id = chunks.source_file_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY score ASC
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall(),
            )
            lexical_ids = {cast(int, row["chunk_id"]) for row in rows}
            path_rows = self._path_matches(path_terms, query, lexical_ids)
            rows = path_rows + rows
        except sqlite3.Error as error:
            return _query_error("search chunks", error)

        hits = [
            SearchHit(
                chunk_id=cast(int, row["chunk_id"]),
                file_path=cast(str, row["file_path"]),
                start_character=cast(int, row["start_character"]),
                end_character=cast(int, row["end_character"]),
                text=cast(str, row["text"]),
                score=cast(float, row["score"]),
            )
            for row in rows
        ]

        unique_hits: dict[int, SearchHit] = {}
        for hit in hits:
            unique_hits.setdefault(hit.chunk_id, hit)
        return Ok(list(unique_hits.values())[:limit])

    def _path_matches(
        self,
        path_terms: tuple[str, ...],
        query: str,
        excluded_ids: set[int],
    ) -> list[sqlite3.Row]:
        """Find chunks from files whose names match query identifiers."""
        useful_terms = tuple(
            term.lower()
            for term in path_terms
            if len(term) >= 5 and term.lower() not in _PATH_STOPWORDS
        )
        if not useful_terms:
            return []
        clauses = " OR ".join("lower(path) LIKE ?" for _ in useful_terms)
        paths = tuple(f"%{term}%" for term in useful_terms)
        file_rows = self.conn.execute(
            f"SELECT id, path FROM source_files WHERE {clauses} LIMIT 80",
            paths,
        ).fetchall()
        if not file_rows:
            return []
        file_ids = tuple(cast(int, row["id"]) for row in file_rows)
        placeholders = ",".join("?" for _ in file_ids)
        rows = self.conn.execute(
            f"""
            SELECT chunks.id AS chunk_id, source_files.path AS file_path,
                   chunks.start_character, chunks.end_character,
                   chunks.text, 0.0 AS score
            FROM chunks JOIN source_files
              ON source_files.id = chunks.source_file_id
            WHERE source_files.id IN ({placeholders})
            """,
            file_ids,
        ).fetchall()
        query_terms = tuple(term.strip('"').lower() for term in path_terms)
        ranked = sorted(
            (row for row in rows if cast(int, row["chunk_id"]) not in excluded_ids),
            key=lambda row: (
                -sum(term in cast(str, row["file_path"]).lower() for term in useful_terms),
                -sum(term in cast(str, row["text"]).lower() for term in query_terms),
                cast(int, row["chunk_id"]),
            ),
        )
        return ranked[: max(20, len(useful_terms) * 4)]

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
    ) -> Result[int, StorageError]:
        """Insert or update a source file and return its database id.

        Returns:
            The database id, or a categorized storage error.
        """
        try:
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
        except sqlite3.Error as error:
            return _query_error("upsert source file", error)

        if row is None:
            return Err(
                StorageError.INVALID_QUERY_RESULT,
                context_msg="Source-file upsert completed without returning an id",
                namespace="storage",
            )

        return Ok(cast(int, row["id"]))

    def delete_chunks_for_source(
        self, source_file_id: int
    ) -> Result[None, StorageError]:
        """Delete all chunks belonging to a source file.

        Returns:
            Success, or a categorized storage error.
        """
        try:
            _ = self.conn.execute(
                """
                DELETE FROM chunks
                WHERE source_file_id = ?
                """,
                (source_file_id,),
            )
        except sqlite3.Error as error:
            return _query_error("delete source chunks", error)
        return Ok(None)

    def insert_chunks(
        self,
        source_file_id: int,
        chunks: list[ChunkInsert],
    ) -> Result[None, StorageError]:
        """Insert a collection of chunks for one source file.

        Returns:
            Success, or a categorized storage error.
        """
        try:
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
        except sqlite3.Error as error:
            return _query_error("insert source chunks", error)
        return Ok(None)

    def get_id_for_source_file(
        self, path: str
    ) -> Result[Option[int], StorageError]:
        """Return the source record for ``path`` when it exists."""
        try:
            row = cast(
                sqlite3.Row | None,
                self.conn.execute(
                    """
                SELECT
                    id
                FROM source_files
                WHERE path = ?
                """,
                    (path,),
                ).fetchone(),
            )
        except sqlite3.Error as error:
            return _query_error("get source file", error)

        if row is None:
            return Ok(Nothing())

        return Ok(Some(cast(int, row["id"])))

    def delete_source_file(
        self, source_file_id: int
    ) -> Result[None, StorageError]:
        """Delete a source file and its cascaded chunks.

        Returns:
            Success, or a categorized storage error.
        """
        try:
            _ = self.conn.execute(
                """
                DELETE FROM source_files
                WHERE id = ?
                """,
                (source_file_id,),
            )
        except sqlite3.Error as error:
            return _query_error("delete source file", error)
        return Ok(None)

    def get_all_source_files(
        self,
    ) -> Result[dict[str, SourceFileRecord], StorageError]:
        """Load all indexed source-file metadata keyed by path."""
        try:
            rows = cast(
                list[sqlite3.Row],
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
                    """
                ).fetchall(),
            )
        except sqlite3.Error as error:
            return _query_error("get all source files", error)

        records = {
            cast(str, row["path"]): SourceFileRecord(
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
            for row in rows
        }

        return Ok(records)


class Transaction:
    """Transaction-scoped typed queries and raw connection access."""

    conn: sqlite3.Connection
    queries: Queries

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Expose transaction-scoped typed queries."""
        self.conn = conn
        self.queries = Queries(conn)


class Store:
    """SQLite-backed application storage."""

    db_path: Path

    def __init__(self, db_path: Path) -> None:
        """Configure storage at ``db_path`` without opening it."""
        self.db_path = db_path

    def init(self) -> Result[None, StorageError]:
        """Create the database directory and initialize its schema.

        Returns:
            Success, or a categorized storage error.
        """
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            return Err(
                StorageError.DIRECTORY_CREATION_FAILED,
                context_msg=f"Could not create the database directory: {error}",
                namespace="storage",
            )

        def initialize(tx: Transaction) -> Result[None, StorageError]:
            try:
                _ = tx.conn.executescript(_SCHEMA)
            except sqlite3.Error as error:
                return _query_error("initialize schema", error)
            return Ok(None)

        return self.with_tx(initialize)

    def with_tx(
        self,
        operation: Callable[[Transaction], Result[T, StorageError]],
    ) -> Result[T, StorageError]:
        """Execute a result-returning operation in one transaction.

        Returns:
            The operation result, or a categorized transaction error.
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
        except sqlite3.Error as error:
            return Err(
                StorageError.CONNECTION_FAILED,
                context_msg=f"Could not open the database: {error}",
                namespace="storage",
            )

        conn.row_factory = sqlite3.Row
        try:
            _ = conn.execute("pragma foreign_keys = on")
            _ = conn.execute("pragma journal_mode = wal")
            _ = conn.execute("pragma synchronous = normal")
            _ = conn.execute("pragma busy_timeout = 5000")
            _ = conn.execute("pragma temp_store = memory")
            _ = conn.execute("pragma cache_size = -131072")
            _ = conn.execute("pragma mmap_size = 268435456")
            _ = conn.execute("begin")
            result = operation(Transaction(conn))
            if isinstance(result, Err):
                conn.rollback()
                return result
            conn.commit()
            return result
        except sqlite3.Error as error:
            conn.rollback()
            return Err(
                StorageError.TRANSACTION_FAILED,
                context_msg=f"Database transaction failed: {error}",
                namespace="storage",
            )
        except Exception as error:
            conn.rollback()
            return Err(
                StorageError.OPERATION_FAILED,
                context_msg=f"Storage operation failed: {error}",
                namespace="storage",
            )
        finally:
            conn.close()

    def read(
        self,
        operation: Callable[[Queries], Result[T, StorageError]],
    ) -> Result[T, StorageError]:
        """Execute a read-only database operation."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
        except sqlite3.Error as error:
            return Err(
                StorageError.CONNECTION_FAILED,
                context_msg=f"Could not open the database: {error}",
                namespace="storage",
            )

        conn.row_factory = sqlite3.Row

        try:
            _ = conn.execute("pragma foreign_keys = on")
            _ = conn.execute("pragma busy_timeout = 5000")
            _ = conn.execute("pragma temp_store = memory")
            _ = conn.execute("pragma cache_size = -131072")
            _ = conn.execute("pragma mmap_size = 268435456")
            return operation(Queries(conn))
        except sqlite3.Error as error:
            return Err(
                StorageError.QUERY_FAILED,
                context_msg=f"Read operation failed: {error}",
                namespace="storage",
            )
        finally:
            conn.close()

    def get_all_source_files(
        self,
    ) -> Result[dict[str, SourceFileRecord], StorageError]:
        """Load all indexed source-file metadata in one read connection.

        Returns:
            Records keyed by their stored path, or a storage error.
        """
        return self.read(lambda queries: queries.get_all_source_files())


def _query_error(operation: str, error: sqlite3.Error) -> Err[StorageError]:
    """Convert a SQLite query exception into a storage result.

    Returns:
        A categorized query error.
    """
    return Err(
        StorageError.QUERY_FAILED,
        context_msg=f"Could not {operation}: {error}",
        namespace="storage",
    )


# normally i would use am embdded migrations dir to handle the db but
# with the scope of this projects its okay to just have the schma as a string
_PATH_STOPWORDS = {
    "about", "after", "class", "does", "false", "from", "given", "which",
    "where", "what", "when", "with", "would", "value", "values",
}

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
