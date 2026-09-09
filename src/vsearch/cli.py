"""Typer CLI for vault-semantic-search."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer
import json
from rich.console import Console

from local_first_common.cli import debug_option, json_option, verbose_option
from local_first_common.obsidian import find_vault_root

from vsearch.bm25 import bm25_stats, get_bm25_connection
from vsearch.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K, VSEARCH_VAULT_ENV
from vsearch.embeddings import OllamaError
from vsearch.indexer import index_vault
from vsearch.search import (
    print_results,
    print_results_json,
    print_results_paths,
    search,
)
from vsearch.store import collection_stats, get_client, get_collection

app = typer.Typer(
    help="Semantic search for your Obsidian vault.",
    add_completion=False,
)
console = Console()


class VSearchError(Exception):
    """Base typed error for vault-semantic-search."""


class VaultDetectionError(VSearchError):
    """Raised when the vault root cannot be resolved."""


# ---------------------------------------------------------------------------
# Vault resolution helper
# ---------------------------------------------------------------------------


def _resolve_vault(vault_str: Optional[str]) -> Path:
    """Resolve vault root: CLI flag > VSEARCH_VAULT env > auto-detect."""
    if vault_str:
        p = Path(vault_str).expanduser().resolve()
        if not p.is_dir():
            console.print(f"[red]Error:[/red] Vault path does not exist: {p}")
            raise typer.Exit(code=1)
        return p

    env_val = os.environ.get(VSEARCH_VAULT_ENV)
    if env_val:
        p = Path(env_val).expanduser().resolve()
        if not p.is_dir():
            console.print(f"[red]Error:[/red] VSEARCH_VAULT path does not exist: {p}")
            raise typer.Exit(code=1)
        return p

    try:
        return find_vault_root()
    except VaultDetectionError:
        raise
    except Exception:
        console.print(
            "[red]Error:[/red] Could not auto-detect Obsidian vault. "
            "Set --vault, or the VSEARCH_VAULT environment variable."
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------


@app.command()
def index(
    vault: Annotated[
        Optional[str], typer.Option("--vault", "-V", help="Path to vault root")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Ollama embedding model")
    ] = DEFAULT_EMBEDDING_MODEL,
    full: Annotated[
        bool, typer.Option("--full", help="Reindex everything, ignore cache")
    ] = False,
    verbose: Annotated[bool, verbose_option()] = False,
    debug: Annotated[bool, debug_option()] = False,
) -> None:
    """Index vault files for semantic search."""
    vault_root = _resolve_vault(vault)

    if verbose:
        console.print(f"Vault: [cyan]{vault_root}[/cyan]")
        console.print(f"Model: [cyan]{model}[/cyan]")
        console.print(
            f"Mode:  [cyan]{'full reindex' if full else 'incremental'}[/cyan]\n"
        )

    client = get_client()
    collection = get_collection(client, model=model, vault_root=str(vault_root))
    bm25_conn = get_bm25_connection()

    try:
        result = index_vault(
            vault_root=vault_root,
            collection=collection,
            model=model,
            full=full,
            verbose=verbose,
            bm25_conn=bm25_conn,
        )
    except OllamaError as e:
        console.print(f"\n[red]Ollama error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(
        f"\nDone. "
        f"Indexed: [green]{result.indexed}[/green], "
        f"Skipped: [dim]{result.skipped}[/dim], "
        f"Errors: [{'red' if result.errors else 'dim'}]{result.errors}[/{'red' if result.errors else 'dim'}]"
    )

    if result.errors:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# search command
# ---------------------------------------------------------------------------


@app.command()
def search_cmd(
    query: Annotated[str, typer.Argument(help="Natural language or keyword search query")],
    top_k: Annotated[
        int, typer.Option("--top-k", "-k", help="Number of results")
    ] = DEFAULT_TOP_K,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Ollama embedding model")
    ] = DEFAULT_EMBEDDING_MODEL,
    mode: Annotated[
        str, typer.Option("--mode", "-M", help="Search mode: hybrid, semantic, or bm25")
    ] = "hybrid",
    bm25: Annotated[
        bool, typer.Option("--bm25", help="Shortcut for --mode bm25")
    ] = False,
    semantic: Annotated[
        bool, typer.Option("--semantic", help="Shortcut for --mode semantic")
    ] = False,
    json_output: Annotated[bool, json_option()] = False,
    paths_only: Annotated[
        bool, typer.Option("--paths-only", help="Output file paths only")
    ] = False,
    vault: Annotated[
        Optional[str], typer.Option("--vault", "-V", help="Path to vault root")
    ] = None,
    verbose: Annotated[bool, verbose_option()] = False,
    debug: Annotated[bool, debug_option()] = False,
) -> None:
    """Search the vault using hybrid, semantic, or BM25 keyword search."""
    if bm25:
        mode = "bm25"
    elif semantic:
        mode = "semantic"

    vault_root = _resolve_vault(vault)

    client = get_client()
    collection = get_collection(client, model=model, vault_root=str(vault_root))
    bm25_conn = get_bm25_connection()

    # Check that at least one index is available
    if mode == "bm25":
        b_stats = bm25_stats(bm25_conn)
        if b_stats["total_chunks"] == 0 and collection.count() == 0:
            console.print(
                "[yellow]Index is empty.[/yellow] Run [bold]vsearch index[/bold] first."
            )
            raise typer.Exit(code=1)
    elif collection.count() == 0:
        console.print(
            "[yellow]Index is empty.[/yellow] Run [bold]vsearch index[/bold] first."
        )
        raise typer.Exit(code=1)

    try:
        results = search(
            query_text=query,
            collection=collection if mode != "bm25" else None,
            top_k=top_k,
            model=model,
            mode=mode,
            bm25_conn=bm25_conn,
        )
    except OllamaError as e:
        console.print(f"\n[red]Ollama error:[/red] {e}")
        raise typer.Exit(code=1)

    if json_output:
        print_results_json(results)
    elif paths_only:
        print_results_paths(results)
    else:
        print_results(results, query_text=query, vault_root=str(vault_root), mode=mode)


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


@app.command()
def stats(
    vault: Annotated[
        Optional[str], typer.Option("--vault", "-V", help="Path to vault root")
    ] = None,
    model: Annotated[
        str, typer.Option("--model", "-m", help="Ollama embedding model")
    ] = DEFAULT_EMBEDDING_MODEL,
    json_output: Annotated[bool, json_option()] = False,
) -> None:
    """Show index statistics."""
    vault_root = _resolve_vault(vault)

    client = get_client()
    collection = get_collection(client, model=model, vault_root=str(vault_root))
    bm25_conn = get_bm25_connection()

    s = collection_stats(collection)
    b_stats = bm25_stats(bm25_conn)

    if json_output:
        s["bm25_chunks"] = b_stats["total_chunks"]
        s["bm25_files"] = b_stats["total_files"]
        print(json.dumps(s, indent=2))
        return

    console.print("\n[bold]Vault Index Stats[/bold]\n")
    console.print(f"  Total chunks : [cyan]{s['total_chunks']}[/cyan]")
    console.print(f"  Total files  : [cyan]{s['total_files']}[/cyan]")
    console.print(f"  BM25 chunks  : [cyan]{b_stats['total_chunks']}[/cyan]")
    console.print(f"  Model        : [cyan]{s['embedding_model']}[/cyan]")
    console.print(f"  Vault        : [cyan]{s['vault_root']}[/cyan]")
    console.print()



# Register 'search' as the public-facing name for search_cmd
app.registered_commands[-2].name = "search"


def main() -> None:
    app()


if __name__ == "__main__":
    main()
