"""Tests for vsearch.search."""

import json
from io import StringIO
from unittest.mock import patch

import chromadb
import pytest

from vsearch.search import (
    SearchResult,
    _make_snippet,
    print_results_json,
    print_results_paths,
    search,
)
from vsearch.store import get_collection, upsert_chunks


FAKE_DIM = 3


def fake_embed(texts, model="nomic-embed-text", **kwargs):
    return [[0.5, 0.5, 0.5]] * len(texts)


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path / "chromadb"))


@pytest.fixture
def populated_collection(client):
    col = get_collection(client)
    upsert_chunks(
        col,
        ids=["sql.md::chunk::0", "baby.md::chunk::0", "bread.md::chunk::0"],
        embeddings=[[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]],
        documents=[
            "NULL values in SQL require IS NULL, not = NULL. Pandas uses NaN.",
            "Baby milestones at six months include sitting with support.",
            "Sourdough starter requires daily feeding at 1:1:1 ratio.",
        ],
        metadatas=[
            {"source_file": "sql.md", "breadcrumb": "NULL Values > Python Comparison"},
            {"source_file": "baby.md", "breadcrumb": "Milestones > Six Months"},
            {"source_file": "bread.md", "breadcrumb": "Sourdough > Starter"},
        ],
    )
    return col


class TestMakeSnippet:
    def test_short_text_unchanged(self):
        text = "Short text."
        assert _make_snippet(text, length=200) == "Short text."

    def test_truncates_long_text(self):
        text = "word " * 100
        snippet = _make_snippet(text, length=50)
        assert len(snippet) <= 55  # allow for ellipsis
        assert snippet.endswith("…")

    def test_truncates_at_word_boundary(self):
        text = "The quick brown fox jumps over the lazy dog and more words"
        snippet = _make_snippet(text, length=20)
        assert "…" in snippet
        # Should not end mid-word
        before_ellipsis = snippet.rstrip("…").strip()
        assert before_ellipsis == before_ellipsis.rsplit(" ", 1)[0] or len(before_ellipsis) <= 20

    def test_strips_leading_trailing_whitespace(self):
        text = "   content   "
        snippet = _make_snippet(text, length=200)
        assert snippet == "content"


class TestSearch:
    def test_returns_search_results(self, populated_collection):
        results = search("SQL", populated_collection, top_k=3, embed_fn=fake_embed)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_rank_starts_at_one(self, populated_collection):
        results = search("query", populated_collection, top_k=2, embed_fn=fake_embed)
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_result_has_source_file(self, populated_collection):
        results = search("SQL NULL", populated_collection, top_k=1, embed_fn=fake_embed)
        assert results[0].source_file in {"sql.md", "baby.md", "bread.md"}

    def test_result_has_breadcrumb(self, populated_collection):
        results = search("query", populated_collection, top_k=1, embed_fn=fake_embed)
        assert results[0].breadcrumb != ""

    def test_result_has_snippet(self, populated_collection):
        results = search("query", populated_collection, top_k=1, embed_fn=fake_embed)
        assert len(results[0].snippet) > 0

    def test_top_k_respected(self, populated_collection):
        results = search("test", populated_collection, top_k=2, embed_fn=fake_embed)
        assert len(results) == 2

    def test_empty_collection_returns_empty(self, client):
        col = get_collection(client)
        results = search("anything", col, top_k=5, embed_fn=fake_embed)
        assert results == []

    def test_to_dict_has_expected_keys(self, populated_collection):
        results = search("query", populated_collection, top_k=1, embed_fn=fake_embed)
        d = results[0].to_dict()
        assert set(d.keys()) == {"rank", "score", "source_file", "breadcrumb", "snippet"}

    def test_score_between_zero_and_one(self, populated_collection):
        results = search("SQL NULL", populated_collection, top_k=3, embed_fn=fake_embed)
        for r in results:
            assert 0.0 <= r.score <= 1.0


class TestPrintResultsJson:
    def test_outputs_valid_json(self, populated_collection, capsys):
        results = search("query", populated_collection, top_k=2, embed_fn=fake_embed)
        print_results_json(results)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_json_contains_score(self, populated_collection, capsys):
        results = search("query", populated_collection, top_k=1, embed_fn=fake_embed)
        print_results_json(results)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "score" in parsed[0]


class TestPrintResultsPaths:
    def test_outputs_unique_paths(self, populated_collection, capsys):
        results = search("query", populated_collection, top_k=3, embed_fn=fake_embed)
        print_results_paths(results)
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().splitlines() if l]
        assert len(lines) == len(set(lines))  # unique

    def test_one_path_per_line(self, populated_collection, capsys):
        results = search("query", populated_collection, top_k=2, embed_fn=fake_embed)
        print_results_paths(results)
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().splitlines() if l]
        assert all("/" not in l or l.count("\n") == 0 for l in lines)
