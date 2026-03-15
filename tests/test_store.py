"""Tests for vsearch.store."""

import chromadb
import pytest

from vsearch.store import (
    collection_stats,
    delete_file_chunks,
    get_collection,
    get_file_metadata,
    query,
    upsert_chunks,
)


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path / "chromadb"))


@pytest.fixture
def collection(client):
    return get_collection(client, model="nomic-embed-text", vault_root="/fake/vault")


# Minimal 3-dimensional fake embeddings for speed
FAKE_DIM = 3


def fake_vec(seed: float) -> list[float]:
    return [seed, seed * 0.5, seed * 0.25]


class TestGetCollection:
    def test_creates_collection(self, client):
        col = get_collection(client)
        assert col.name == "vault"

    def test_idempotent(self, client):
        col1 = get_collection(client)
        col2 = get_collection(client)
        assert col1.name == col2.name

    def test_stores_embedding_model_in_metadata(self, client):
        col = get_collection(client, model="mxbai-embed-large")
        assert col.metadata["embedding_model"] == "mxbai-embed-large"

    def test_stores_vault_root(self, client):
        col = get_collection(client, vault_root="/my/vault")
        assert col.metadata["vault_root"] == "/my/vault"

    def test_cosine_space(self, client):
        col = get_collection(client)
        assert col.metadata.get("hnsw:space") == "cosine"


class TestUpsertChunks:
    def test_upserts_and_counts(self, collection):
        upsert_chunks(
            collection,
            ids=["id1", "id2"],
            embeddings=[fake_vec(0.1), fake_vec(0.9)],
            documents=["doc1", "doc2"],
            metadatas=[{"source_file": "a.md"}, {"source_file": "b.md"}],
        )
        assert collection.count() == 2

    def test_upsert_is_idempotent(self, collection):
        upsert_chunks(
            collection,
            ids=["id1"],
            embeddings=[fake_vec(0.5)],
            documents=["doc1"],
            metadatas=[{"source_file": "a.md"}],
        )
        upsert_chunks(
            collection,
            ids=["id1"],
            embeddings=[fake_vec(0.5)],
            documents=["doc1 updated"],
            metadatas=[{"source_file": "a.md"}],
        )
        assert collection.count() == 1


class TestDeleteFileChunks:
    def test_deletes_all_chunks_for_file(self, collection):
        upsert_chunks(
            collection,
            ids=["f1c0", "f1c1", "f2c0"],
            embeddings=[fake_vec(0.1), fake_vec(0.2), fake_vec(0.3)],
            documents=["d1", "d2", "d3"],
            metadatas=[
                {"source_file": "file1.md"},
                {"source_file": "file1.md"},
                {"source_file": "file2.md"},
            ],
        )
        deleted = delete_file_chunks(collection, "file1.md")
        assert deleted == 2
        assert collection.count() == 1

    def test_delete_nonexistent_file_returns_zero(self, collection):
        deleted = delete_file_chunks(collection, "ghost.md")
        assert deleted == 0


class TestGetFileMetadata:
    def test_returns_metadata_for_indexed_file(self, collection):
        upsert_chunks(
            collection,
            ids=["x0"],
            embeddings=[fake_vec(0.3)],
            documents=["content"],
            metadatas=[{"source_file": "note.md", "mtime": "12345", "hash": "abc"}],
        )
        meta = get_file_metadata(collection, "note.md")
        assert meta is not None
        assert meta["mtime"] == "12345"
        assert meta["hash"] == "abc"

    def test_returns_none_for_unknown_file(self, collection):
        meta = get_file_metadata(collection, "missing.md")
        assert meta is None


class TestQuery:
    def test_returns_top_k_results(self, collection):
        upsert_chunks(
            collection,
            ids=[f"id{i}" for i in range(5)],
            embeddings=[fake_vec(i * 0.1) for i in range(5)],
            documents=[f"doc{i}" for i in range(5)],
            metadatas=[{"source_file": f"file{i}.md"} for i in range(5)],
        )
        results = query(collection, fake_vec(0.0), top_k=3)
        assert len(results) == 3

    def test_result_has_expected_keys(self, collection):
        upsert_chunks(
            collection,
            ids=["q1"],
            embeddings=[fake_vec(0.5)],
            documents=["content"],
            metadatas=[{"source_file": "a.md"}],
        )
        results = query(collection, fake_vec(0.5), top_k=1)
        assert len(results) == 1
        hit = results[0]
        assert "id" in hit
        assert "document" in hit
        assert "metadata" in hit
        assert "score" in hit
        assert "distance" in hit

    def test_score_is_one_minus_distance(self, collection):
        upsert_chunks(
            collection,
            ids=["s1"],
            embeddings=[fake_vec(0.5)],
            documents=["content"],
            metadatas=[{"source_file": "a.md"}],
        )
        results = query(collection, fake_vec(0.5), top_k=1)
        hit = results[0]
        assert abs(hit["score"] - (1.0 - hit["distance"])) < 1e-9


class TestCollectionStats:
    def test_empty_collection(self, collection):
        stats = collection_stats(collection)
        assert stats["total_chunks"] == 0
        assert stats["total_files"] == 0

    def test_counts_unique_files(self, collection):
        upsert_chunks(
            collection,
            ids=["a0", "a1", "b0"],
            embeddings=[fake_vec(0.1), fake_vec(0.2), fake_vec(0.3)],
            documents=["d1", "d2", "d3"],
            metadatas=[
                {"source_file": "a.md"},
                {"source_file": "a.md"},
                {"source_file": "b.md"},
            ],
        )
        stats = collection_stats(collection)
        assert stats["total_chunks"] == 3
        assert stats["total_files"] == 2

    def test_reports_embedding_model(self, collection):
        stats = collection_stats(collection)
        assert stats["embedding_model"] == "nomic-embed-text"
