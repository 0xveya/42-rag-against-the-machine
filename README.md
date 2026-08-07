_This project has been created as part of the 42 curriculum by sfurst._

# RAG Against the Machine

RAG Against the Machine is a local Retrieval-Augmented Generation (RAG) system
for answering questions about source-code repositories. It discovers and chunks
a codebase, stores the chunks in a SQLite full-text index, retrieves relevant
source locations with BM25, and gives that evidence to Qwen3-0.6B to produce a
grounded answer.

## Description

A language model does not automatically know the contents of a private or
recently changed repository. RAG solves this by retrieving information at answer
time instead of retraining the model. This project implements the four stages
directly:

1. **Indexing:** discover files, divide them into useful chunks, and persist a
   searchable representation.
2. **Retrieval:** rank chunks against a question and return the best exact source
   locations.
3. **Augmentation:** place those sources into a bounded model context.
4. **Generation:** ask a local 0.6B model to answer from the supplied evidence.

The main focus is retrieval quality and transparent systems design rather than
wrapping a large RAG framework. The implementation uses Tree-sitter, SQLite
FTS5, `asyncio`, threads, processes, and Linux inotify. The mandatory Python
Fire CLI is accompanied by an optional FastAPI/WebSocket interface.

## Features

- syntax-aware Tree-sitter chunking for Python, C, C++, Go, JavaScript,
  TypeScript, Rust, and Zig;
- a separate Markdown/plain-text chunker;
- SQLite FTS5 search with BM25 ranking;
- identifier, file-name, and path-aware retrieval without modifying returned
  evidence;
- exact file paths and character ranges compatible with the subject evaluator;
- concurrent indexing with `asyncio`, worker threads, and processes for large
  corpora;
- incremental indexing and a native Linux inotify watcher;
- local answer generation with `Qwen/Qwen3-0.6B`;
- Pydantic output models and categorized `Result`/`Ok`/`Err` error flow;
- an optional local HTTP and WebSocket API.

## System Architecture

```text
data/raw repositories
        |
        v
 file discovery ---- stored metadata
        |                  |
        +---- unchanged ---+--> skip
        |
        v
 async worker queue
        |
        +--> read and hash
        +--> Tree-sitter or document chunker
        +--> validate exact character ranges
        |
        v
 batched SQLite writes --> source_files + chunks + FTS5
                                      |
question --> query normalization --> BM25 search
                                      |
                         path/identifier reranking
                                      |
                                      v
                          top-k source locations
                                      |
                     bounded prompt + Qwen3-0.6B
                                      |
                                      v
                         grounded answer + sources

Linux inotify --> asyncio loop.add_reader --> reindex affected file only
```

SQLite keeps source metadata, chunk content, and the search index in one local,
transactional database. `asyncio` coordinates the pipeline; blocking file work
runs in threads, while CPU-heavy work can move to processes. This keeps the
event loop responsive without pretending that asynchronous code makes parsing
CPU-parallel by itself.

## Chunking Strategy

Source code is segmented around Tree-sitter syntax nodes rather than arbitrary
character boundaries. Function, class, method, and declaration boundaries make
each chunk more meaningful to both retrieval and generation. Oversized ranges
are split to respect the configured maximum size; Python structural ranges use
overlap to retain context across a split.

### How Tree-sitter is used

Tree-sitter is an incremental parser generator: instead of returning only
tokens, it builds a concrete syntax tree whose nodes retain their exact byte
ranges in the original file. The project loads a grammar based on the file
extension and defines useful top-level node types per language. Examples
include:

| Language | Structural nodes used as boundaries |
|---|---|
| Python | functions, classes, decorated definitions |
| C | functions, declarations, typedefs, linkage and preprocessor blocks |
| C++ | functions, templates, namespaces, types, concepts and modules |
| JavaScript/TypeScript | functions, classes, interfaces, types, enums and exports |
| Rust | functions, structs, enums, traits, implementations and modules |
| Go | functions, methods, types, constants and variables |
| Zig | functions, declarations, tests and container members |

The parser walks the root node's named children and selects nodes configured as
structural boundaries. Anonymous punctuation nodes are irrelevant for this
decision. A node containing a syntax error is skipped instead of being trusted
as a source boundary. Gaps between accepted nodes are added back for most
languages, so imports, comments, module variables, and other non-structural text
are not silently lost. If parsing or offset conversion fails, the file falls
back to bounded plain-text chunking rather than failing the entire index build.

Tree-sitter returns UTF-8 byte offsets, but Python strings and the moulinette use
Unicode character indices. `ByteCharacterMap` builds a lookup from every byte
position to its owning Python character. For example, one four-byte emoji maps
all four byte positions to one character index. The final end position maps to
`len(text)`. This avoids the subtle drift that appears when a file contains
accented identifiers, non-ASCII comments, or other multibyte text.

After chunking, every range is checked for ordering, size, and correspondence
with the source:

```text
chunk.text == complete_file[
    chunk.first_character_index:chunk.last_character_index
]
```

This invariant matters because retrieval quality is useless if the reported
location points at different text.

### Python-specific structural chunks

Python definitions receive a specialized path because large classes and
decorated definitions are common in the evaluation corpus. Each selected
definition keeps its structural header, and definitions larger than
`max_chunk_size` are divided into windows with up to 400 characters of overlap
(capped at one quarter of the configured chunk size). The overlap helps retain
context around a split method or docstring.

For search only, a Python chunk is prefixed with its file path and structure
header. Identifiers are also decomposed: `load_lora_adapter` contributes
`load`, `lora`, and `adapter`, while `OpenAIServing` contributes `openai` and
`serving`. Natural-language questions can consequently match code naming
conventions without changing the exact text returned to the evaluator.

### Markdown and plain text

Markdown is scanned into headings, fenced code, paragraphs, and blank-line
boundaries. Adjacent blocks are combined while they fit; an oversized block is
split at a preferred newline or whitespace boundary. Plain-text files use the
same bounded splitting primitive without Markdown semantics. Both strategies
favor readable passages while guaranteeing the maximum character size.

Every stored chunk has two representations:

- `text` is the exact original source slice returned to the evaluator;
- `search_text` enriches that content with paths, structural headers, and aliases
  such as `snake_case` and `CamelCase` variants.

## Indexing Pipeline

Indexing is organized as a bounded producer/consumer pipeline rather than one
large sequential loop:

1. discovery walks `data/raw`, classifies supported extensions, ignores known
   generated/vendor directories, and creates `SourceFile` records containing
   both absolute read paths and corpus-relative stored paths;
2. stored metadata is compared before expensive work begins;
3. changed files enter an `asyncio.Queue`, while a fixed number of workers
   provide backpressure;
4. workers read bytes, calculate a SHA-256 digest, decode and normalize text,
   choose the appropriate chunker, and validate every range;
5. processed files are persisted in bounded batches and diagnostics from failed
   files are retained instead of cancelling unrelated work.

`asyncio` owns task lifetime, cancellation, queues, and progress reporting.
Ordinary blocking file work runs through `asyncio.to_thread`. Above the large
corpus threshold, complete files can instead be distributed through a
`ProcessPoolExecutor`, avoiding the GIL for parsing work while keeping each
worker result serializable. The process pool uses a limited worker count and a
map chunk size of eight to avoid spending more time on IPC and startup than on
the files themselves.

Incremental checks compare file size, nanosecond modification time, file type,
maximum chunk size, and an internal chunker version before doing any hashing.
If the recorded metadata matches, the file is skipped immediately. When it
differs, SHA-256 distinguishes a real content change from a timestamp-only
update; the indexing settings are also retained in the source record for cache
provenance and diagnostics.

## Retrieval Method

SQLite FTS5 performs lexical matching and its built-in BM25 function supplies
the initial ranking. Queries are normalized into safe FTS terms, with useful
identifiers and path fragments preserved. Candidate chunks are then adjusted by
small code-oriented signals:

- exact or partial file-path matches;
- file-stem matches;
- identifier overlap, including split naming conventions;
- structural context stored only in `search_text`.

Results are deduplicated and returned as top-k `MinimalSource` objects. Each
contains the corpus-relative `file_path`, `first_character_index`, and
`last_character_index` required by the moulinette. A source is considered
correct when it is in the same file as a reference and its range overlaps with
an IoU of at least 0.05.

SQLite FTS5 was chosen over a vector database because it is local, fast,
dependency-light, and especially effective for exact identifiers and paths.
The tradeoff is that a purely semantic paraphrase with no shared useful terms
can still be missed.

## Data and Storage Model

The generated database lives below `data/processed/`. It separates file-level
state from retrievable passages:

```text
source_files (one row per corpus file)
    |
    | 1:N, ON DELETE CASCADE
    v
chunks (exact text, search text, character range)
    |
    | external-content rowid
    v
chunks_fts (FTS5 search index)
```

`source_files` stores the unique relative path, file type, byte size,
nanosecond mtime, SHA-256 content hash, chunk size, chunker version, and last
index timestamp. This is both provenance and the cache key used to decide
whether a file needs processing.

`chunks` stores a stable per-file sequence number, exact source text, enriched
search text, character start/end positions, and creation time. Database checks
reject negative indices and inverted ranges, while a unique constraint prevents
two chunks from claiming the same sequence position in one file.

`chunks_fts` is an FTS5 external-content table: authoritative content stays in
`chunks`, and the full-text table refers to its row IDs. The `unicode61`
tokenizer is configured to keep `_` as a token character, which preserves exact
code identifiers while the added aliases still support word-level matching.
Insert, update, and delete triggers normally keep FTS synchronized. Deleting a
source cascades to its chunks, whose delete triggers remove the search entries.

Fresh imports take a faster controlled path. FTS maintenance triggers are
temporarily removed, rows are inserted with `executemany` in bounded
transactions, FTS is rebuilt once from the completed content table, and the
triggers are restored. Incremental updates keep the normal trigger path because
correct atomic maintenance matters more than bulk throughput for one file.

The evaluator-facing Pydantic models are deliberately smaller than the internal
records. `MinimalSource` exposes only the exact path and character range;
`StudentSearchResults` groups retrieved sources by question; and
`StudentSearchResultsAndAnswer` adds generated answers. Internal hashes,
timestamps, BM25 scores, and enriched text never leak into the required JSON.

## Instructions

### Requirements

- Python 3.10 or later;
- [`uv`](https://docs.astral.sh/uv/);
- enough disk space for the corpus and local model;
- Linux when using the inotify watcher.

Install dependencies:

```bash
make install
```

Show all CLI commands and current options:

```bash
uv run python -m src --help
```

The subject data setup expects `vllm-0.10.1.zip` in `~/Downloads` or `data/`:

```bash
make subject-setup
```

### Example Usage

Index `data/raw/` into `data/processed/`:

```bash
uv run python -m src index --max_chunk_size 2000
```

Search one question:

```bash
uv run python -m src search \
  "Where are LoRA adapters registered?" --k 10
```

Generate one grounded answer:

```bash
uv run python -m src answer \
  "How does vLLM apply a LoRA adapter?" \
  --k 10 --model Qwen/Qwen3-0.6B
```

Run retrieval on a dataset:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

Evaluate those results locally:

```bash
uv run python -m src evaluate \
  --student_search_results_path \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path \
    data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Generate answers for the retrieved dataset:

```bash
uv run python -m src answer_dataset \
  --student_search_results_path \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory \
    data/output/search_results_and_answer/UnansweredQuestions
```

Equivalent convenience targets are available as `make index`, `make search`,
`make search-dataset`, `make answer`, `make answer-dataset`, and `make evaluate`.
For example:

```bash
make search QUERY="How are LoRA adapters loaded?" K=10
make search-dataset DATASET_SCOPE=code K=10
```

Run the complete subject flow-setup, fresh indexing, both dataset searches,
official moulinette evaluation, tests, and linting-with:

```bash
make subject-eval
```

All dataset and output paths are CLI arguments; none are hard-coded into the
application.

## Output Models

Search output uses `StudentSearchResults`; generated output uses
`StudentSearchResultsAndAnswer`. The minimal search shape is:

```json
{
  "search_results": [
    {
      "question_id": "q1",
      "question": "How is an OpenAI-compatible server configured?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
          "first_character_index": 9867,
          "last_character_index": 10100
        }
      ]
    }
  ],
  "k": 10
}
```

Answer output adds an `answer` field to each search result. Malformed JSON,
missing paths, empty queries, nonsensical queries, and non-positive `k` values
are handled as categorized errors rather than uncaught tracebacks.

## Performance Analysis

The subject requires indexing within 300 seconds, searching roughly 200
questions within 90 seconds, documentation recall@5 of at least 0.80, and code
recall@5 of at least 0.50.

The following fresh comparison was produced with `make moulinette-compare` on
the same corpus and machine. Search time covers both public datasets (199
questions total).

| Implementation | Docs R@5 | Code R@5 | Index | Search (199) |
|---|---:|---:|---:|---:|
| This project | 0.820 | 0.778 | 3.080s | 4.292s |
| [mpouillo reference](https://github.com/mpouillo/42-rag-against-the-machine) | 0.850 | 0.778 | 104.535s | 65.400s |
| Subject minimum | 0.800 | 0.500 | 300s | 90s |

This implementation clears both recall thresholds and is substantially faster
in this run. The comparison can be reproduced with:

```bash
make moulinette-compare
```

The main performance choices are batched transactions and `executemany`, a
single FTS rebuild after clean bulk imports, native SQLite ranking, concurrent
file processing, and metadata-based skipping or one-file reindexing for later
updates. Timings vary with hardware, filesystem cache, and dependency state, so
the table should be treated as one measured run rather than a universal ratio.

## Design Decisions

### Direct components instead of a RAG framework

The indexing, retrieval, context, and generation stages are explicit project
modules. This makes ranking behavior and source-range correctness easy to test
and explain during evaluation.

### SQLite for storage and retrieval

FTS5 provides persistence, transactions, BM25, and metadata joins without a
separate search service. An external-content index avoids maintaining two
independent copies of chunk text. WAL and connection pragmas improve normal
operation, while fresh imports use a dedicated bulk path.

### Search enrichment without changing evidence

Paths, headers, and identifier aliases improve matching, but only exact corpus
text is returned. Retrieval hints therefore cannot corrupt the character ranges
checked by the evaluator.

### Explicit concurrency boundaries

`asyncio` handles orchestration and cancellation, threads handle blocking file
operations, and processes are reserved for sufficiently large CPU-bound jobs.
Persistence remains bounded and transactional instead of allowing every worker
to write independently.

### Typed errors and validated boundaries

Core operations return `Result`, `Ok`, or `Err` values with error categories.
Pydantic validates evaluator-facing data, Python Fire is kept at the CLI
boundary, and FastAPI acts only as an optional adapter over the same service.

## Challenges Faced

- **Byte and character offsets:** Tree-sitter positions are byte-based, but the
  evaluator expects character indices. Explicit UTF-8 conversion and source
  slice invariants fixed failures on non-ASCII files.
- **Natural language versus identifiers:** Questions often phrase a concept
  differently from its source symbol. Split identifier aliases, path terms,
  and structural search text improved retrieval without altering evidence.
- **Slow initial FTS imports:** Maintaining FTS triggers for every individual
  insert was expensive. Batched inserts followed by one FTS rebuild reduced
  fresh indexing time considerably.
- **Consistent incremental updates:** Source rows, chunks, and FTS entries must
  change atomically. File replacement and deletion now use transactions.
- **Filesystem renames:** inotify reports renames as paired events rather than a
  single high-level operation. Events are correlated before updating paths.
- **Small-model context quality:** Qwen3-0.6B is sensitive to noisy context, so
  the prompt uses a small ranked evidence set and clearly labels source paths.

## Filesystem Watching and Incremental Indexing

The filesystem watcher is implemented directly on Linux inotify rather than
through a polling library. A small `ctypes` binding calls `inotify_init1`,
`inotify_add_watch`, and `inotify_rm_watch`. The descriptor is opened
nonblocking and close-on-exec, and the raw variable-length `inotify_event`
records are parsed with explicit header and name-length validation.

inotify watches directories, not an abstract recursive tree, so
`WatchCoordinator` maintains both `watch_descriptor -> path` and `path ->
watch_descriptor` maps. Startup walks the repository and registers each
directory. When a directory is created, its subtree is added; when it is
deleted, its mappings are removed. Symlink traversal is disabled by default to
avoid cycles and unexpectedly watching content outside the corpus.

Raw kernel flags are normalized into a small application vocabulary:
`CREATED`, `MODIFIED`, `METADATA_CHANGED`, `DELETED`, and `RENAMED`. Rename
handling is the interesting part. Linux emits `IN_MOVED_FROM` and
`IN_MOVED_TO` separately but gives both the same cookie. The coordinator holds
the first half temporarily, pairs it with the destination, and emits one rename
event. If no destination arrives within 100 ms, the move is treated as a
deletion from the watched tree. Renaming a directory also rewrites every stored
watch path below that subtree.

The async bridge registers the inotify file descriptor with
`loop.add_reader`. The event loop calls `_on_readable` only when kernel data is
available, so no thread blocks in `read()` and no timer repeatedly scans the
filesystem. Parsed events enter an `asyncio.Queue` and can be consumed through
`recv()`, timeout-aware receiving, or `async for`. Closing removes the reader,
cancels the pending rename timer, closes the descriptor, and wakes blocked
receivers.

The index event handler then maps filesystem changes to database operations:

- create/modify/metadata events read, hash, chunk, and transactionally upsert
  one supported file;
- deletion removes the source row and cascade-owned chunks/FTS entries;
- rename deletes the old stored path and indexes the new one;
- unsupported file types and directory-only events do not enter the chunker.

An inotify queue overflow is surfaced explicitly because some events may have
been lost; the safe recovery is a metadata rescan rather than pretending the
incremental state is still authoritative. A synchronous watcher frontend also
exists for uses that do not run an asyncio loop.

## Local API

Start the optional web interface and API:

```bash
make serve
```

It listens on `http://localhost:8000/` by default. `/health` reports service
health, `/api/repositories` manages local corpora, and `/ws` streams retrieval
status, sources, generated token fragments, completion, and errors. This is a
local development interface, not a hardened public service.

## Testing

```bash
make test
make lint
make lint-strict
```

The test suite covers discovery rules, document and Tree-sitter chunking, UTF-8
range conversion, SQLite transactions and FTS synchronization, BM25 ordering,
path/identifier ranking, unchanged-file skipping, watcher events, CLI edge
cases, Pydantic output, generation failures, and API/WebSocket behavior.

## Limitations

- FTS5 is lexical and can miss paraphrases with no shared useful terms.
- The filesystem watcher currently depends on Linux inotify.
- The small local model can produce incomplete phrasing even with good sources.
- Process workers have startup and memory costs and only help above a workload
  threshold.
- SQLite fits a local single-node tool, not a distributed multi-tenant service.
- FTS5 must be enabled in the Python environment's SQLite build.

## Resources

- [Retrieval-Augmented Generation paper](https://arxiv.org/abs/2005.11401)
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
- [SQLite FTS5 extension](https://sqlite.org/fts5.html)
- [SQLite query planner](https://sqlite.org/queryplanner.html)
- [Tree-sitter documentation](https://tree-sitter.github.io/tree-sitter/)
- [Python asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [Linux inotify manual](https://man7.org/linux/man-pages/man7/inotify.7.html)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Transformers documentation](https://huggingface.co/docs/transformers/)
- [Python Fire guide](https://google.github.io/python-fire/guide/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)

### Use of AI

AI tools were used to locate documentation, discuss edge cases, expand repetitive
test cases, improve docstrings and README explanations, translate already-known
backend concepts into Python/FastAPI idioms, and investigate unusual model,
tokenizer, typing, and async-lifecycle errors. And create the shitty html files 
without css for the server bonus.

Suggestions were checked against primary documentation and local tests before
being accepted. The author designed and remains responsible for the core
SQLite/FTS5 architecture, Tree-sitter chunking, ranking heuristics, exact range
handling, concurrency split, incremental watcher, error model, generation
integration, and API boundaries. All submitted code remains  hand written (except
 tests and the above mentioned html files) and explainable during peer evaluation.

## Repository Layout

```text
.
├── data/
│   ├── raw/                 # repositories to index
│   ├── processed/           # generated SQLite index
│   ├── datasets/            # question and ground-truth datasets
│   └── output/              # generated search and answer files
├── scripts/                 # subject and benchmark helpers
├── src/
│   └── rag_against_the_machine/
│       ├── fs/              # inotify watcher
│       ├── generation/      # context and local model
│       ├── indexing/        # discovery, readers, chunkers, pipeline
│       ├── models/          # validated shared models
│       ├── retrieval/       # query construction and ranking
│       ├── server/          # optional FastAPI/WebSocket interface
│       └── storage/         # SQLite schema and queries
├── tests/
├── web/
├── Makefile
├── pyproject.toml
└── README.md
```
