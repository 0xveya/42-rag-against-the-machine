# rag-against-the-machine

Bootstrap scaffold for a local RAG project over the supplied vLLM repository.

This repository currently contains project setup only: uv packaging, static-analysis
targets, a Python Fire dependency, the vendored 42 CLI diagnostic helpers, and a
minimal FastAPI application factory. No indexing, retrieval, generation, or evaluation
logic has been implemented.

## Local data

Keep downloaded subject materials and the vLLM archive under `data/`. Extract the
repository later into `data/raw/`; generated SQLite/BM25/vector artifacts belong in
`data/processed/`. These corpus and generated-data paths are ignored by Git.

## Checks

```bash
uv sync --all-extras
make lint
make typecheck
make test
```

`make typecheck` runs both mypy and ty; `make typecheck-ty` runs ty on its own.
