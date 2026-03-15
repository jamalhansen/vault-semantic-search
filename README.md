# vault-semantic-search

Semantic search for your Obsidian vault using local embeddings. Search by meaning instead of keywords.

```
$ uv run vsearch search "that discussion about NULL handling in pandas"

Results for: "that discussion about NULL handling in pandas"

1. [0.87] jamalhansen.com/_series/sql-for-python-devs/posts/11-null-values/draft.md
   Section: NULL Values > Python Comparison
   "In pandas, missing values are represented as NaN or None. SQL uses NULL,
    which behaves differently in comparisons..."

2. [0.82] Timeline/2026-02-15.md
   Section: Morning Pages
   "Spent an hour debugging a join that returned fewer rows than expected.
    Turned out the join key had NULLs..."
```

## Why

Obsidian's built-in search is keyword-only. With 300+ notes, you can't remember the exact words you used. Semantic search finds notes by meaning — you can describe the concept loosely and still find the right file.

All processing runs locally. No API keys, no data sent anywhere.

## Installation

**Prerequisites:**
- [Ollama](https://ollama.ai) installed and running
- Pull the default embedding model: `ollama pull nomic-embed-text`

**Install:**
```bash
cd vault-semantic-search
uv sync
```

## Usage

### Index your vault

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
# Natural language query
uv run vsearch search "baby milestones"

# More results
uv run vsearch search "window functions SQL" --top-k 10

# JSON output (for piping)
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
vsearch index   [--vault PATH] [--model MODEL] [--full] [--verbose]
vsearch search  QUERY [--vault PATH] [--model MODEL] [--top-k N]
                      [--json] [--paths-only] [--verbose]
vsearch stats   [--vault PATH] [--model MODEL]
```

## Architecture

```
Obsidian Vault → Chunker → Ollama /api/embed → ChromaDB
                                                     ↓
                            Query → Embed → Similarity Search → Results
```

**Storage:** ChromaDB persists to `~/.local/share/vsearch/chromadb/` (XDG-compliant, outside the vault).

**Chunking:** Markdown-aware. Splits on H1/H2 headers first, then H3/H4, then paragraphs, then word boundaries. YAML frontmatter is stored as metadata, not embedded. Minimum chunk size: 50 tokens.

**Incremental indexing:** Files are re-embedded only when their content hash changes. A full index of 300 files takes a few minutes; incremental runs take seconds.

**Embedding model:** Default is `nomic-embed-text` (768 dimensions, Ollama). Swap with `--model`.

## Project Structure

```
src/vsearch/
├── cli.py          # Typer CLI (index, search, stats)
├── config.py       # Defaults, paths, constants
├── chunker.py      # Markdown-aware chunking
├── embeddings.py   # Ollama /api/embed client
├── store.py        # ChromaDB wrapper (importable by other tools)
├── indexer.py      # Vault walker + orchestration
└── search.py       # Query + result formatting

tests/
├── conftest.py
├── test_chunker.py
├── test_embeddings.py   # unit + @integration tests
├── test_indexer.py
├── test_search.py
├── test_store.py
└── fixtures/sample_vault/   # Minimal fake vault
```

## Running tests

```bash
# Unit tests only (no Ollama required)
uv run pytest -m "not integration"

# All tests including integration (requires Ollama + nomic-embed-text)
uv run pytest
```

## Reuse by other tools

`store.py` is designed to be imported by other local-first tools that want to query the same index:

```python
from vsearch.store import get_client, get_collection, query

client = get_client()
collection = get_collection(client)
results = query(collection, my_embedding_vector, top_k=5)
```

Tools using this pattern: `series-cross-link-suggester`.
