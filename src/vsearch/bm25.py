"""SQLite FTS5 BM25 search index for vault-semantic-search.

Provides persistent full-text indexing with native BM25 ranking.
Works standalone without Ollama or external services.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional

from vsearch.config import get_bm25_db_path

SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    source_file,
    breadcrumb,
    chunk_index UNINDEXED,
    content,
    tokenize = 'porter unicode61'
);
"""


def get_bm25_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return an SQLite connection with the FTS5 chunks table created."""
    path = db_path or get_bm25_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    init_bm25_db(conn)
    return conn


def get_in_memory_bm25_connection() -> sqlite3.Connection:
    """Return an ephemeral in-memory SQLite connection (for tests)."""
    conn = sqlite3.connect(":memory:")
    init_bm25_db(conn)
    return conn


def init_bm25_db(conn: sqlite3.Connection) -> None:
    """Initialize FTS5 virtual table if it doesn't exist."""
    conn.execute(SCHEMA)
    conn.commit()


def sanitize_query(query: str) -> str:
    """Prepare a user query string for safe FTS5 MATCH evaluation.

    Extracts quoted phrases and individual alphanumeric words.
    Escapes them in quotes to prevent FTS5 syntax errors.
    """
    tokens = re.findall(r'\"([^\"]+)\"|(\w+)', query)
    terms: list[str] = []
    for phrase, word in tokens:
        if phrase:
            clean = " ".join(re.findall(r"\w+", phrase))
            if clean:
                terms.append(f'"{clean}"')
        elif word:
            terms.append(f'"{word}"')
    if not terms:
        return ""
    return " OR ".join(terms)


def upsert_bm25_chunks(
    conn: sqlite3.Connection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """Insert or replace chunks in the FTS5 index."""
    with conn:
        for chunk_id, doc, meta in zip(ids, documents, metadatas):
            conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
            conn.execute(
                """
                INSERT INTO chunks_fts(chunk_id, source_file, breadcrumb, chunk_index, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    meta.get("source_file", ""),
                    meta.get("breadcrumb", ""),
                    str(meta.get("chunk_index", 0)),
                    doc,
                ),
            )


def delete_bm25_file_chunks(conn: sqlite3.Connection, source_file: str) -> int:
    """Delete all chunks for a given source file. Returns deleted count."""
    with conn:
        cur = conn.execute(
            "DELETE FROM chunks_fts WHERE source_file = ?", (source_file,)
        )
        return cur.rowcount


def query_bm25(
    conn: sqlite3.Connection,
    query_text: str,
    top_k: int = 5,
) -> list[dict]:
    """Query the FTS5 index with BM25 ranking.

    Returns a list of dicts: id, document, metadata, score.
    """
    fts_query = sanitize_query(query_text)
    if not fts_query:
        return []

    sql = """
        SELECT chunk_id, source_file, breadcrumb, chunk_index, content, bm25(chunks_fts) as rank_score
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY rank_score ASC
        LIMIT ?
    """
    try:
        cursor = conn.execute(sql, (fts_query, top_k))
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        return []

    hits = []
    for row in rows:
        chunk_id, source_file, breadcrumb, chunk_index, content, rank_score = row
        # SQLite bm25() returns negative values where lower is better.
        # Invert so higher is better for ranking consistency.
        score = -float(rank_score)
        hits.append(
            {
                "id": chunk_id,
                "document": content,
                "metadata": {
                    "source_file": source_file,
                    "breadcrumb": breadcrumb,
                    "chunk_index": int(chunk_index) if str(chunk_index).isdigit() else 0,
                },
                "score": score,
            }
        )
    return hits


def bm25_stats(conn: sqlite3.Connection) -> dict:
    """Return statistics for the BM25 index."""
    try:
        cur = conn.execute("SELECT count(*), count(DISTINCT source_file) FROM chunks_fts")
        total_chunks, total_files = cur.fetchone()
    except sqlite3.OperationalError:
        return {"total_chunks": 0, "total_files": 0}
    return {
        "total_chunks": total_chunks or 0,
        "total_files": total_files or 0,
    }
