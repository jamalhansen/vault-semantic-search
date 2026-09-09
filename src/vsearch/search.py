import json
import sqlite3
from dataclasses import dataclass
from typing import Callable, Optional

import chromadb
from rich.console import Console
from rich.text import Text

from vsearch.bm25 import get_bm25_connection, query_bm25
from vsearch.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SEARCH_MODE,
    DEFAULT_TOP_K,
    SNIPPET_LENGTH,
)
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


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    bm25_hits: list[dict],
    top_k: int = DEFAULT_TOP_K,
    k: int = 60,
) -> list[dict]:
    """Combine dense vector hits and BM25 sparse hits using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = sum(1 / (k + rank(d))) across sources.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    ranks: dict[str, dict[str, int]] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        items[cid] = hit
        ranks.setdefault(cid, {})["dense"] = rank

    for rank, hit in enumerate(bm25_hits, start=1):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in items:
            items[cid] = hit
        ranks.setdefault(cid, {})["bm25"] = rank

    sorted_ids = sorted(
        scores.keys(), key=lambda cid: scores[cid], reverse=True
    )[:top_k]
    fused: list[dict] = []
    for cid in sorted_ids:
        item = items[cid]
        meta = dict(item.get("metadata") or {})
        meta["rrf_score"] = scores[cid]
        if "dense" in ranks[cid]:
            meta["dense_rank"] = ranks[cid]["dense"]
        if "bm25" in ranks[cid]:
            meta["bm25_rank"] = ranks[cid]["bm25"]
        fused.append(
            {
                "id": cid,
                "document": item["document"],
                "metadata": meta,
                "score": scores[cid],
            }
        )
    return fused


def search(
    query_text: str,
    collection: Optional[chromadb.Collection] = None,
    top_k: int = DEFAULT_TOP_K,
    model: str = DEFAULT_EMBEDDING_MODEL,
    embed_fn: Optional[Callable] = None,
    mode: str = DEFAULT_SEARCH_MODE,
    bm25_conn: Optional[sqlite3.Connection] = None,
) -> list[SearchResult]:
    """Search the vault using hybrid, semantic, or BM25 search.

    Args:
        query_text: Natural language query string.
        collection: ChromaDB collection to search (required for semantic/hybrid).
        top_k: Number of results to return.
        model: Ollama embedding model.
        embed_fn: Override for embed_texts (used in tests).
        mode: Search mode: 'hybrid' (default), 'semantic', or 'bm25'.
        bm25_conn: Optional SQLite connection for BM25 search.

    Returns:
        List of SearchResult objects, ranked by relevance.
    """
    if mode == "bm25":
        if bm25_conn is None:
            bm25_conn = get_bm25_connection()
        hits = query_bm25(bm25_conn, query_text, top_k=top_k)

    elif mode == "semantic":
        if collection is None:
            raise ValueError("collection is required for semantic search")
        _embed = embed_fn or embed_texts
        embeddings = _embed([query_text], model=model)
        query_vec = embeddings[0]
        hits = store_query(collection, query_vec, top_k=top_k)

    elif mode == "hybrid":
        candidate_k = max(top_k * 2, 20)
        dense_hits: list[dict] = []
        if collection is not None and collection.count() > 0:
            _embed = embed_fn or embed_texts
            embeddings = _embed([query_text], model=model)
            query_vec = embeddings[0]
            dense_hits = store_query(collection, query_vec, top_k=candidate_k)

        bm25_hits: list[dict] = []
        conn = bm25_conn
        if conn is None:
            try:
                conn = get_bm25_connection()
            except Exception:
                conn = None
        if conn is not None:
            bm25_hits = query_bm25(conn, query_text, top_k=candidate_k)

        if dense_hits and bm25_hits:
            hits = reciprocal_rank_fusion(dense_hits, bm25_hits, top_k=top_k)
        elif dense_hits:
            hits = dense_hits[:top_k]
        elif bm25_hits:
            hits = bm25_hits[:top_k]
        else:
            hits = []

    else:
        raise ValueError(f"Unknown search mode: {mode}. Must be hybrid, semantic, or bm25.")

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
    mode: str = "hybrid",
) -> None:
    """Print results using Rich formatting."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f'\n[bold]Results for:[/bold] "{query_text}" [dim]({mode})[/dim]\n')

    for r in results:
        header = Text()
        header.append(f"{r.rank}. ", style="bold")
        if "dense_rank" in r.metadata or "bm25_rank" in r.metadata:
            sources = []
            if r.metadata.get("dense_rank"):
                sources.append(f"dense #{r.metadata['dense_rank']}")
            if r.metadata.get("bm25_rank"):
                sources.append(f"bm25 #{r.metadata['bm25_rank']}")
            header.append(f"[{r.score:.4f}] ", style="green")
            header.append(r.source_file, style="cyan")
            header.append(f"  [{', '.join(sources)}]", style="dim")
        else:
            score_color = (
                "green"
                if r.score >= 0.7
                else ("yellow" if r.score >= 0.5 else "red")
            )
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
