# Dev Notes: vault-semantic-search

Notes for the future blog post. Written as we build.

---

## The Problem

The vault has 300+ files. Obsidian's search is keyword-only. You think "that thing about NULL handling in pandas" but the note says "missing values" and "NaN" — no match. Frustrating.

The fix is semantic search: embed everything with a vector model, store the vectors, and at query time embed your question and find similar vectors. You describe the concept in your own words and it finds notes that are semantically close.

## Why Local-First

No API keys. No data leaving the machine. No per-query costs. Ollama runs `nomic-embed-text` locally — 768-dimensional embeddings that are fast and good enough for a personal vault. ChromaDB stores the vectors persistently on disk.

The whole thing runs offline. This matters for a vault that may contain private notes, half-formed ideas, and things you wouldn't want sent to a cloud service.

## Build Order

We built it in this order (each step independently testable):

1. **chunker.py** — Pure Python, no dependencies. Get the markdown splitting right before anything else. Hardest conceptual part of the whole tool.
2. **embeddings.py** — Ollama HTTP client. Thin wrapper around `/api/embed`, with graceful error handling for "Ollama not running" and "model not pulled".
3. **store.py** — ChromaDB wrapper. Designed to be importable by other tools (series-cross-link-suggester wants to query the same index).
4. **config.py** — Defaults and paths. XDG-compliant storage at `~/.local/share/vsearch/chromadb/`.
5. **indexer.py** — Vault walker + orchestration. Ties chunker + embeddings + store together. Incremental indexing via mtime + content hash.
6. **search.py** — Query embedding + ChromaDB query + result formatting. Output modes: rich terminal, JSON, paths-only.
7. **cli.py** — Typer CLI. Thin glue layer over the above modules.

## Interesting Design Decisions

### Markdown-Aware Chunking

The spec said "do NOT use naive fixed-size character splitting." This was the right call.

The chunker splits on headers (H1/H2 first, then H3/H4 if sections are still too large), then falls back to paragraph boundaries, then word boundaries as a last resort. Each chunk carries a "breadcrumb" — the header path at its location, like `"NULL Values > Python Comparison"`.

Frontmatter (YAML between `---` fences) is parsed and stored as ChromaDB metadata on every chunk from that file, but NOT embedded as text. Embedding `status: idea` or `created: 2026-03-03` wastes vector space.

The minimum chunk size (50 tokens) prevents tiny header sections from becoming useless standalone chunks. They get merged into adjacent content instead.

### The Merge Bug

First attempt at the merge logic had a nasty bug: once a small piece went into "pending", subsequent pieces would keep getting merged into it as long as the total stayed under max_tokens (1000). For a file where every section was < 50 tokens, they'd all collapse into one giant chunk.

Fix: once a pending piece + new piece crosses the min_tokens threshold, commit it to the results and reset. Stop accumulating. This gives each chunk a fighting chance to stand alone.

### ChromaDB Isolation in Tests

`chromadb.Client()` (deprecated) and `chromadb.EphemeralClient()` both turned out to share state across test instances in ChromaDB 1.5.x. Tests were failing with leftover data from previous tests bleeding into subsequent ones.

Fix: use `chromadb.PersistentClient(path=str(tmp_path / "chromadb"))` in test fixtures. `tmp_path` is pytest's per-test temporary directory, so each test gets a genuinely isolated ChromaDB instance.

### mtime + Hash Change Detection

Incremental indexing checks mtime first (fast). If mtime changed, recomputes the content hash (MD5 is fine — we're not doing security here). If the hash is the same, the file was touched but not changed — skip it. Only if the hash changed do we delete old chunks and re-embed.

This handles: normal edits (mtime + hash change), `touch` or backup tools (mtime change, hash same → no reindex), and copied files that happen to have identical content.

### The Provider Abstraction Non-Issue

The local-first-common library has a provider abstraction for text generation (Anthropic, Groq, etc.). It was tempting to route embeddings through it.

Didn't. Embeddings go through Ollama's `/api/embed` endpoint directly. The provider abstraction is for text *generation*. Embedding is a different operation and mixing them would have made the architecture murkier. If we ever add a RAG "answer my question" mode, that generation step would use the provider abstraction — but the embedding step stays direct.

### CLI Command Naming

Typer registers commands by function name. We wanted `vsearch search` but can't have a function named `search` (conflicts with the import from `vsearch.search`). Solution: name the function `search_cmd` and then patch the registered command name after the fact:

```python
app.registered_commands[-2].name = "search"
```

Slightly hacky but works. The alternative would be to alias the import.

## Numbers

- **90 tests** passing (23 chunker, 11 embeddings unit, 22 store, 23 indexer, 21 search)
- **3 integration tests** (marked `@pytest.mark.integration`, skipped when Ollama unavailable)
- **7 source modules** in src/vsearch/
- Built in one session

### The Two-Character Bug

First real test against the vault crashed immediately:
```
AttributeError: 'OptionInfo' object has no attribute 'strip'
```

Root cause: `local_first_common.cli` exports `verbose_option` and `debug_option` as **factory functions**, not option values. The correct usage is `verbose: bool = verbose_option()`. I had written `verbose: bool = verbose_option` (without `()`), which passed the function object as Typer's default. Click then tried to type-cast the function to a bool by calling `.strip()` on it.

All tests passed because tests never exercise the CLI entry point directly. Two characters, caught only by actually running the tool.

### The 400 Bug: Three Bugs, One Error Code

After the first real vault run, 7 files produced `400 Bad Request` from Ollama's `/api/embed`. Useless error message. What followed was three rounds of debugging, each revealing a different bug.

**Attempt 1: num_ctx** — Added `"options": {"num_ctx": 8192}` to the embed request. nomic-embed-text's Modelfile already sets `num_ctx 8192`, and Ollama 0.17.7 ignores the `options` field in the embed endpoint anyway. Failed to fix any of the 7 files. Added a batch-splitting fallback (`if status == 400 and len(texts) > 1: retry singletons`) as a safety net regardless.

**Root cause 1: wrong tokenizer assumption.** `WORDS_PER_TOKEN = 0.75` was calibrated for BPE (GPT-style tokenizers), where 1 word ≈ 1.33 tokens. nomic-embed-text uses BERT **WordPiece**, which tokenizes ~2.5× more aggressively — especially on technical content, code blocks, and markdown syntax. A 750-word section was estimating to 1000 tokens but actually tokenizing to 2500+ real BERT tokens.

Fix: `WORDS_PER_TOKEN = 0.4` (1 word ≈ 2.5 BERT tokens). Now max_words = 400 per chunk. This fixed 5 of the 7 files.

**Root cause 2: oversized first paragraph.** In `_split_large_section`, the paragraph merge loop had `if _token_estimate(candidate) > max_tokens and current_parts:`. When the very first paragraph was already too large, `current_parts` was empty — the `and current_parts` check was falsy — so the oversized paragraph bypassed word-boundary splitting and was returned as-is. Fixed by separating the "flush current" and "split this paragraph" decisions.

**Root cause 3: word count is meaningless for URL-dense content.** The Gemini-generated archive file was a 81k-char table of URLs and favicons. URLs are single "words" in Python's `.split()`, but BERT tokenizes each URL segment (every `/`, `.`, `?=`) separately: ~2 chars/BERT token vs ~3-4 for prose. A 393-word, 5977-char paragraph had a token estimate of 982 (safely under 1000) but exceeded 2048 actual BERT tokens.

Fix: added `MAX_CHUNK_CHARS = 4000` as a hard cap alongside the token estimate. This cap was applied in four places: the splitting trigger in `chunk_file`, the termination condition in `_split_large_section`, the flush condition in the paragraph merge loop, and inside `_split_by_words` itself (now enforces both word count and char count).

Final result: 1019 files, 0 errors.

## First Real Results

After pulling `nomic-embed-text` and fixing the chunker, full reindex: 1019 files, 20 skipped, 0 errors.

```
$ vsearch search "building local ai tools"
1. [0.72] Lesson Plan/Hello LLM/Hello LLM - Week 4 - AI Tools.md
2. [0.69] Timeline/2026-03-02.md  (daily note mentioning PydanticAI integration)
3. [0.68] jamalhansen.com/_series/local-first-ai/Local-First AI - Prep Timeline.md
4. [0.67] PyTexas/2025 Conference/Demystifying AI Agents with Python Code.md
5. [0.66] jamalhansen.com/_series/local-first-ai/Local-First AI - Tool Ideas.md
```

That's exactly right. The daily note got surfaced because it mentioned wanting to integrate PydanticAI into local-first AI work — not keywords, but semantic meaning. That's the whole point.

## What's Next

- Consider storing chunk positions for better "jump to source" UX
- The `series-cross-link-suggester` can import from `vsearch.store` to reuse the same index
- **MCP server** — expose `search_vault(query, top_k)` and `get_vault_stats()` as MCP tools so Claude can query the vault directly in conversation. The store module is already importable; the wrapper would be thin. Natural next step once the index is stable.

## Rough Timeline

- Project scaffold + pyproject.toml: ~10 min
- chunker.py + tests + bug fixes: ~30 min
- embeddings.py + tests: ~10 min
- store.py + tests + ChromaDB isolation bug: ~20 min
- indexer.py + tests: ~20 min
- search.py + tests: ~15 min
- cli.py: ~10 min
- README + dev-notes: ~15 min
- Total: ~2 hours
