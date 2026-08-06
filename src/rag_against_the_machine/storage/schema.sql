PRAGMA foreign_keys = ON;

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
    search_text TEXT NOT NULL,
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
    search_text,
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
        search_text,
        source_file_id
    )
VALUES
    (
        new.id,
        new.search_text,
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
        search_text,
        source_file_id
    )
VALUES
    (
        'delete',
        old.id,
        old.search_text,
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
        search_text,
        source_file_id
    )
VALUES
    (
        'delete',
        old.id,
        old.search_text,
        old.source_file_id
    );

INSERT INTO
    chunks_fts (
        rowid,
        search_text,
        source_file_id
    )
VALUES
    (
        new.id,
        new.search_text,
        new.source_file_id
    );

END;
