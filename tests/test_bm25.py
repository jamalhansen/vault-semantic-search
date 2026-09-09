"""Tests for vsearch.bm25 SQLite FTS5 search index."""


import pytest

from vsearch.bm25 import (
    bm25_stats,
    delete_bm25_file_chunks,
    get_bm25_connection,
    get_in_memory_bm25_connection,
    query_bm25,
    sanitize_query,
    upsert_bm25_chunks,
)


@pytest.fixture
def mem_conn():
    return get_in_memory_bm25_connection()


class TestSanitizeQuery:
    def test_empty_string(self):
        assert sanitize_query("") == ""
        assert sanitize_query("   ") == ""

    def test_simple_words(self):
        result = sanitize_query("pandas dataframe")
        assert result == '"pandas" OR "dataframe"'

    def test_quoted_phrase(self):
        result = sanitize_query('"null handling" in python')
        assert result == '"null handling" OR "in" OR "python"'

    def test_special_characters_sanitized(self):
        # Colon, asterisks, brackets should not trigger FTS5 syntax errors
        result = sanitize_query("error: [FATAL] *warning* (critical)")
        assert '"error"' in result
        assert '"FATAL"' in result
        assert '"warning"' in result
        assert '"critical"' in result
        assert ":" not in result
        assert "[" not in result

    def test_punctuation_only_returns_empty(self):
        assert sanitize_query("!@#$%^&*()") == ""


class TestBM25Store:
    def test_init_creates_table(self, mem_conn):
        cur = mem_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        )
        assert cur.fetchone() is not None

    def test_persistent_connection(self, tmp_path):
        db_file = tmp_path / "test_bm25.db"
        conn = get_bm25_connection(db_file)
        assert db_file.exists()
        conn.close()

    def test_upsert_and_query(self, mem_conn):
        ids = ["sql.md::chunk::0", "baby.md::chunk::0", "bread.md::chunk::0"]
        docs = [
            "NULL values in SQL require IS NULL instead of equal. Pandas uses NaN.",
            "Baby milestones at six months include sitting with support and babbling.",
            "Sourdough starter requires daily feeding with flour and water at 1:1:1 ratio.",
        ]
        metas = [
            {"source_file": "sql.md", "breadcrumb": "SQL > NULL", "chunk_index": 0},
            {"source_file": "baby.md", "breadcrumb": "Baby > Milestones", "chunk_index": 0},
            {"source_file": "bread.md", "breadcrumb": "Bread > Starter", "chunk_index": 0},
        ]
        upsert_bm25_chunks(mem_conn, ids, docs, metas)

        # Query for SQL
        hits = query_bm25(mem_conn, "SQL NULL handling", top_k=2)
        assert len(hits) >= 1
        assert hits[0]["id"] == "sql.md::chunk::0"
        assert hits[0]["metadata"]["source_file"] == "sql.md"
        assert hits[0]["metadata"]["breadcrumb"] == "SQL > NULL"
        assert hits[0]["score"] > 0

    def test_upsert_overwrites_existing(self, mem_conn):
        upsert_bm25_chunks(
            mem_conn,
            ids=["doc.md::chunk::0"],
            documents=["Original text about cats"],
            metadatas=[{"source_file": "doc.md", "chunk_index": 0}],
        )
        upsert_bm25_chunks(
            mem_conn,
            ids=["doc.md::chunk::0"],
            documents=["Updated text about dogs"],
            metadatas=[{"source_file": "doc.md", "chunk_index": 0}],
        )

        assert len(query_bm25(mem_conn, "cats")) == 0
        hits = query_bm25(mem_conn, "dogs")
        assert len(hits) == 1
        assert "dogs" in hits[0]["document"]

    def test_delete_file_chunks(self, mem_conn):
        upsert_bm25_chunks(
            mem_conn,
            ids=["file1.md::chunk::0", "file1.md::chunk::1", "file2.md::chunk::0"],
            documents=["First chunk", "Second chunk", "Other file chunk"],
            metadatas=[
                {"source_file": "file1.md", "chunk_index": 0},
                {"source_file": "file1.md", "chunk_index": 1},
                {"source_file": "file2.md", "chunk_index": 0},
            ],
        )

        deleted = delete_bm25_file_chunks(mem_conn, "file1.md")
        assert deleted == 2

        stats = bm25_stats(mem_conn)
        assert stats["total_chunks"] == 1
        assert stats["total_files"] == 1

    def test_query_empty_db(self, mem_conn):
        hits = query_bm25(mem_conn, "anything")
        assert hits == []

    def test_query_invalid_terms(self, mem_conn):
        hits = query_bm25(mem_conn, "??? ::: !!!")
        assert hits == []

    def test_stats(self, mem_conn):
        stats = bm25_stats(mem_conn)
        assert stats["total_chunks"] == 0
        assert stats["total_files"] == 0
