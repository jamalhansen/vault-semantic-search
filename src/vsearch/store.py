"""ChromaDB wrapper for vault-semantic-search.

Designed to be importable by other tools (series-cross-link-suggester, etc.)
without triggering CLI or Typer initialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api import ClientAPI

from vsearch.config import COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL, get_db_path


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_client(db_path: Optional[Path] = None) -> ClientAPI:
    """Return a persistent ChromaDB client at the given path (or default)."""
    path = db_path or get_db_path()
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_in_memory_client() -> ClientAPI:
    """Return an ephemeral in-memory ChromaDB client (for tests). Each call is isolated."""
    return chromadb.EphemeralClient()


# ---------------------------------------------------------------------------
# Collection factory
# ---------------------------------------------------------------------------

def get_collection(
    client: ClientAPI,
    model: str = DEFAULT_EMBEDDING_MODEL,
    vault_root: Optional[str] = None,
) -> chromadb.Collection:
    """Get or create the vault collection with the given embedding model."""
    meta: dict = {
        "embedding_model": model,
        "hnsw:space": "cosine",
    }
    if vault_root:
        meta["vault_root"] = vault_root
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata=meta)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def upsert_chunks(
    collection: chromadb.Collection,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Upsert a batch of chunks into the collection."""
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def delete_file_chunks(collection: chromadb.Collection, source_file: str) -> int:
    """Delete all chunks for a given source file. Returns the number deleted."""
    results = collection.get(where={"source_file": {"$eq": source_file}})
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def get_file_metadata(
    collection: chromadb.Collection, source_file: str
) -> Optional[dict]:
    """Return the stored mtime/hash metadata for a file, or None if not indexed."""
    results = collection.get(
        where={"source_file": {"$eq": source_file}},
        limit=1,
        include=["metadatas"],
    )
    metas = results.get("metadatas", [])
    if not metas:
        return None
    return metas[0]


def query(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """Query the collection and return top_k results as dicts.

    Each dict has: id, document, metadata, distance.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for id_, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append(
            {
                "id": id_,
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "score": 1.0 - dist,  # cosine similarity
            }
        )
    return hits


def collection_stats(collection: chromadb.Collection) -> dict:
    """Return basic statistics about the collection."""
    count = collection.count()
    meta = collection.metadata or {}
    # Count unique files
    if count > 0:
        all_metas = collection.get(include=["metadatas"])["metadatas"]
        files = {m.get("source_file") for m in all_metas if m}
    else:
        files = set()
    return {
        "total_chunks": count,
        "total_files": len(files),
        "embedding_model": meta.get("embedding_model", "unknown"),
        "vault_root": meta.get("vault_root", "unknown"),
    }
