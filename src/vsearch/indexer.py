"""Vault walker, chunker, and embedding orchestrator for vault-semantic-search."""

from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import chromadb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from vsearch.chunker import chunk_file
from vsearch.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SKIP_DIRS,
    DEFAULT_SKIP_PREFIXES,
    MIN_FILE_CHARS,
    VSEARCH_IGNORE_FILE,
)
from vsearch.embeddings import embed_texts
from vsearch.store import (
    delete_file_chunks,
    get_file_metadata,
    upsert_chunks,
)

console = Console()


# ---------------------------------------------------------------------------
# File filtering
# ---------------------------------------------------------------------------

def _load_vsearchignore(vault_root: Path) -> list[str]:
    ignore_file = vault_root / VSEARCH_IGNORE_FILE
    if not ignore_file.exists():
        return []
    lines = ignore_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _should_skip(path: Path, vault_root: Path, ignore_patterns: list[str]) -> bool:
    """Return True if the path should be excluded from indexing."""
    relative = path.relative_to(vault_root)
    parts = relative.parts

    # Skip hidden dirs/files
    for part in parts:
        if part.startswith(DEFAULT_SKIP_PREFIXES):
            return True

    # Skip default dirs
    if parts[0] in DEFAULT_SKIP_DIRS:
        return True

    # Skip non-markdown files
    if path.suffix.lower() != ".md":
        return True

    # Skip tiny files
    try:
        if path.stat().st_size < MIN_FILE_CHARS:
            return True
    except OSError:
        return True

    # Apply .vsearchignore patterns
    relative_str = str(relative)
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(relative_str, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True

    return False


def walk_vault(vault_root: Path) -> list[Path]:
    """Return all indexable markdown files in the vault."""
    ignore_patterns = _load_vsearchignore(vault_root)
    files = []
    for path in sorted(vault_root.rglob("*.md")):
        if not _should_skip(path, vault_root, ignore_patterns):
            files.append(path)
    return files


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def _mtime_str(path: Path) -> str:
    return str(path.stat().st_mtime)


def file_needs_reindex(
    path: Path, collection: chromadb.Collection, vault_root: Optional[Path] = None
) -> bool:
    """Return True if the file is new or has changed since last index.

    Uses the relative path (from vault_root) to look up stored metadata.
    """
    relative = str(path.relative_to(vault_root)) if vault_root else str(path)
    meta = get_file_metadata(collection, relative)
    if meta is None:
        return True
    stored_mtime = meta.get("mtime")
    current_mtime = _mtime_str(path)
    if stored_mtime == current_mtime:
        return False
    # mtime changed — check actual content
    stored_hash = meta.get("hash")
    current_hash = _file_hash(path)
    return stored_hash != current_hash


# ---------------------------------------------------------------------------
# Chunk → ChromaDB ID
# ---------------------------------------------------------------------------

def _chunk_id(source_file: str, chunk_index: int) -> str:
    return f"{source_file}::chunk::{chunk_index}"


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

@dataclass
class IndexResult:
    indexed: int = 0
    skipped: int = 0
    deleted: int = 0
    errors: int = 0


def index_vault(
    vault_root: Path,
    collection: chromadb.Collection,
    model: str = DEFAULT_EMBEDDING_MODEL,
    full: bool = False,
    verbose: bool = False,
    embed_fn: Optional[Callable] = None,
) -> IndexResult:
    """Index all files in the vault into ChromaDB.

    Args:
        vault_root: Root of the Obsidian vault.
        collection: ChromaDB collection to write to.
        model: Ollama embedding model name.
        full: If True, reindex everything. If False, skip unchanged files.
        verbose: Print extra info.
        embed_fn: Optional override for embed_texts (used in tests).
    """
    _embed = embed_fn or embed_texts
    result = IndexResult()

    files = walk_vault(vault_root)
    indexed_files: set[str] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing vault…", total=len(files))

        for file_path in files:
            relative = str(file_path.relative_to(vault_root))
            indexed_files.add(relative)
            progress.update(task, description=f"[cyan]{relative}[/cyan]")

            try:
                if not full and not file_needs_reindex(file_path, collection, vault_root):
                    result.skipped += 1
                    progress.advance(task)
                    continue

                # Remove stale chunks for this file
                delete_file_chunks(collection, relative)

                chunks = chunk_file(file_path, vault_root)
                if not chunks:
                    result.skipped += 1
                    progress.advance(task)
                    continue

                texts = [c.text for c in chunks]
                embeddings = _embed(texts, model=model)

                ids = [_chunk_id(relative, c.chunk_index) for c in chunks]
                documents = texts
                metadatas = []
                mtime = _mtime_str(file_path)
                content_hash = _file_hash(file_path)

                for c in chunks:
                    meta = {
                        "source_file": c.source_file,
                        "breadcrumb": c.breadcrumb,
                        "chunk_index": c.chunk_index,
                        "mtime": mtime,
                        "hash": content_hash,
                    }
                    # Flatten frontmatter metadata (strings only — ChromaDB limitation)
                    for k, v in c.frontmatter_meta.items():
                        if isinstance(v, (str, int, float, bool)):
                            meta[f"fm_{k}"] = str(v)
                        elif isinstance(v, list):
                            meta[f"fm_{k}"] = ", ".join(str(x) for x in v)
                    metadatas.append(meta)

                upsert_chunks(collection, ids, embeddings, documents, metadatas)
                result.indexed += 1

                if verbose:
                    console.print(f"  [green]✓[/green] {relative} ({len(chunks)} chunks)")

            except Exception as e:
                result.errors += 1
                console.print(f"  [red]✗[/red] {relative}: {e}")

            progress.advance(task)

    # Remove chunks for files that no longer exist in the vault
    if not full:
        result.deleted += _cleanup_deleted_files(collection, indexed_files, verbose)

    return result


def _cleanup_deleted_files(
    collection: chromadb.Collection,
    current_files: set[str],
    verbose: bool = False,
) -> int:
    """Delete chunks for files that were removed from the vault."""
    if collection.count() == 0:
        return 0

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    stored_files = {m["source_file"] for m in all_meta if m and "source_file" in m}
    removed = stored_files - current_files
    total_deleted = 0

    for source_file in removed:
        n = delete_file_chunks(collection, source_file)
        total_deleted += n
        if verbose:
            console.print(f"  [yellow]removed[/yellow] {source_file} ({n} chunks)")

    return total_deleted
