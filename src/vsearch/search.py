"""Query embedding + similarity search + result formatting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

import chromadb
from rich.console import Console
from rich.text import Text

from vsearch.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K, SNIPPET_LENGTH
from vsearch.embeddings import embed_texts
from vsearch.store import query as store_query

console = Console()


@dataclass
class SearchResult:
    rank: int
    score: float
    source_file: str
    breadcrumb: str
    snippet: str
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "source_file": self.source_file,
            "breadcrumb": self.breadcrumb,
            "snippet": self.snippet,
        }


def search(
    query_text: str,
    collection: chromadb.Collection,
    top_k: int = DEFAULT_TOP_K,
    model: str = DEFAULT_EMBEDDING_MODEL,
    embed_fn: Optional[Callable] = None,
) -> list[SearchResult]:
    """Embed a query and return the top_k most similar chunks.

    Args:
        query_text: Natural language query string.
        collection: ChromaDB collection to search.
        top_k: Number of results to return.
        model: Ollama embedding model.
        embed_fn: Override for embed_texts (used in tests).

    Returns:
        List of SearchResult objects, ranked by similarity.
    """
    _embed = embed_fn or embed_texts
    embeddings = _embed([query_text], model=model)
    query_vec = embeddings[0]

    hits = store_query(collection, query_vec, top_k=top_k)

    results = []
    for rank, hit in enumerate(hits, start=1):
        meta = hit["metadata"] or {}
        snippet = _make_snippet(hit["document"])
        results.append(
            SearchResult(
                rank=rank,
                score=hit["score"],
                source_file=meta.get("source_file", "unknown"),
                breadcrumb=meta.get("breadcrumb", ""),
                snippet=snippet,
                metadata=meta,
            )
        )
    return results


def _make_snippet(text: str, length: int = SNIPPET_LENGTH) -> str:
    """Return the first `length` characters of text, truncated cleanly."""
    text = text.strip()
    if len(text) <= length:
        return text
    truncated = text[:length]
    # Try to truncate at a word boundary
    last_space = truncated.rfind(" ")
    if last_space > length // 2:
        truncated = truncated[:last_space]
    return truncated + "…"


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def print_results(
    results: list[SearchResult],
    query_text: str,
    vault_root: Optional[str] = None,
) -> None:
    """Print results using Rich formatting."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f'\n[bold]Results for:[/bold] "{query_text}"\n')

    for r in results:
        score_color = "green" if r.score >= 0.7 else ("yellow" if r.score >= 0.5 else "red")
        header = Text()
        header.append(f"{r.rank}. ", style="bold")
        header.append(f"[{r.score:.2f}] ", style=score_color)
        header.append(r.source_file, style="cyan")
        if r.breadcrumb:
            header.append(f"\n   Section: {r.breadcrumb}", style="dim")

        console.print(header)
        console.print(f'   [italic]"{r.snippet}"[/italic]\n')


def print_results_json(results: list[SearchResult]) -> None:
    """Print results as JSON."""
    print(json.dumps([r.to_dict() for r in results], indent=2))


def print_results_paths(results: list[SearchResult]) -> None:
    """Print one file path per line (for piping to fzf/xargs)."""
    seen: set[str] = set()
    for r in results:
        if r.source_file not in seen:
            print(r.source_file)
            seen.add(r.source_file)
