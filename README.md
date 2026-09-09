# vault-semantic-search

Semantic, keyword (BM25), and hybrid search for your Obsidian vault using local embeddings and SQLite FTS5.

```
$ uv run vsearch search "that discussion about NULL handling in pandas"

Results for: "that discussion about NULL handling in pandas" (hybrid)

1. [0.0325] jamalhansen.com/_series/sql-for-python-devs/posts/11-null-values/draft.md  [dense #1, bm25 #2]
   Section: NULL Values > Python Comparison
   "In pandas, missing values are represented as NaN or None. SQL uses NULL,
    which behaves differently in comparisons..."

2. [0.0164] Timeline/2026-02-15.md  [dense #2]
   Section: Morning Pages
   "Spent an hour debugging a join that returned fewer rows than expected.
    Turned out the join key had NULLs..."
```

## Why

Obsidian's built-in search is keyword-only. With 300+ notes, you can't remember the exact words you used. Semantic search finds notes by meaning — you can describe the concept loosely and still find the right file.

Hybrid search combines the best of both worlds:
- **Semantic (Dense Vector)**: Finds notes conceptually related to your query even when vocabulary differs.
- **BM25 (Sparse Keyword)**: Excels at exact terms, function names, error messages, and unique identifiers.
- **Reciprocal Rank Fusion (RRF)**: Merges ranked candidate lists without brittle score calibration.
- **Zero-Daemon Offline Mode**: BM25 search requires no Ollama or embedding model, enabling instant offline searches.

All processing runs locally. No API keys, no data sent anywhere.

## Installation

**Prerequisites:**
- [Ollama](https://ollama.ai) installed and running (for dense/hybrid search)
- Pull the default embedding model: `ollama pull nomic-embed-text`

**Install:**
```bash
cd vault-semantic-search
uv sync
```

## Usage

### Index your vault

Indexes your markdown files into ChromaDB (vector embeddings) and SQLite FTS5 (BM25 full-text index) simultaneously:

```bash
# First run: full index
uv run vsearch index

# Subsequent runs: incremental (only changed files)
uv run vsearch index

# Force full reindex
uv run vsearch index --full

# Use a different embedding model
uv run vsearch index --model mxbai-embed-large

# Explicit vault path
uv run vsearch index --vault ~/my-vault
```

### Search

```bash
# Hybrid search (default: combines semantic vectors + BM25 keyword rankings via RRF)
uv run vsearch search "baby milestones"

# BM25 keyword search only (instant, requires NO running Ollama daemon)
uv run vsearch search "window functions SQL" --bm25

# Pure semantic search only
uv run vsearch search "window functions SQL" --semantic

# Request more results
uv run vsearch search "window functions SQL" --top-k 10

# JSON output (for programmatic piping)
uv run vsearch search "sourdough hydration" --json

# Paths only (for piping to fzf or xargs)
uv run vsearch search "authentication flow" --paths-only
```

### Stats

```bash
uv run vsearch stats
```

## Configuration

| Method | Priority |
|--------|----------|
| `--vault` flag | Highest |
| `VSEARCH_VAULT` environment variable | Middle |
| Auto-detect (looks for `.obsidian/` directory) | Lowest |

Set your vault in `.envrc` for convenience:
```bash
export VSEARCH_VAULT=~/my-vault
```

## Excluding files

Create `.vsearchignore` in your vault root (same syntax as `.gitignore`):
```
private/
*.draft.md
_templates/
```

By default, these are always excluded:
- `.obsidian/` directory
- `_templates/` directory
- Hidden files/folders (starting with `.`)
- Files under 50 characters

## CLI Reference

```
uv run vsearch index   [--vault PATH] [--model MODEL] [--full] [--verbose]
uv run vsearch search  QUERY [--vault PATH] [--model MODEL] [--top-k N]
                             [--mode hybrid|semantic|bm25] [--bm25] [--semantic]
                             [--json] [--paths-only] [--verbose]
uv run vsearch stats   [--vault PATH] [--model MODEL] [--json]
```

## Architecture

```
Obsidian Vault ─┬─> Chunker ─┬─> Ollama /api/embed ─> ChromaDB (Dense) ─────┐
                │            │                                              ├─> Reciprocal Rank Fusion ─> Ranked Results
                │            └─> SQLite FTS5 (BM25 Sparse) ─────────────────┘
                │
                └─> Change Detection (hash + mtime)
```

- **Storage:**
  - ChromaDB: `~/.local/share/vsearch/chromadb/`
  - SQLite FTS5: `~/.local/share/vsearch/bm25.db`
- **Chunking:** Markdown-aware. Splits on H1/H2 headers first, then H3/H4, then paragraphs, then word boundaries. YAML frontmatter is stored as metadata.
- **Incremental indexing:** Files are re-indexed only when their content hash changes.
- **Embedding model:** Default is `nomic-embed-text` (768 dimensions, Ollama).

## Project Structure

This tool follows the [Local-First AI project blueprint](https://github.com/jamalhansen/local-first-common).

```
vault-semantic-search/
├── src/
│   └── vsearch/
│       ├── cli.py          # Typer CLI entry points (index, search, stats)
│       ├── core.py         # Domain orchestrators and pipeline
│       ├── bm25.py         # SQLite FTS5 BM25 persistence & query
│       ├── store.py        # ChromaDB wrapper
│       ├── chunker.py      # Markdown-aware chunking
│       ├── embeddings.py   # Ollama /api/embed client
│       ├── indexer.py      # Vault walker & dual-index synchronization
│       └── search.py       # Query execution, RRF fusion, formatting
├── pyproject.toml          # Managed by uv
└── tests/
    ├── conftest.py
    ├── test_bm25.py        # FTS5 indexing, query sanitization, and BM25 tests
    ├── test_search.py      # Semantic, BM25, and RRF hybrid search tests
    ├── test_indexer.py     # Dual-index synchronization tests
    └── ...
```

## Running tests

```bash
# Unit tests only (no Ollama required)
uv run pytest -m "not integration"

# All tests including integration (requires Ollama + nomic-embed-text)
uv run pytest
```

## Reuse by other tools

`store.py` and `bm25.py` are designed to be imported by other local-first tools:

```python
from vsearch.store import get_client, get_collection, query
from vsearch.bm25 import get_bm25_connection, query_bm25
from vsearch.search import search

# High-level hybrid search
results = search("null handling", collection=collection, bm25_conn=bm25_conn, mode="hybrid")
```

Tools using this pattern: `series-cross-link-suggester`.
