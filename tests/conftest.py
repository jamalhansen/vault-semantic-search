"""Shared fixtures for vault-semantic-search tests."""

from pathlib import Path

import pytest
from local_first_common.testing import isolate_tracking_db  # noqa: F401

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_VAULT = FIXTURES_DIR / "sample_vault"


@pytest.fixture
def sample_vault() -> Path:
    """Return the path to the sample vault fixture directory."""
    return SAMPLE_VAULT


@pytest.fixture
def tmp_vault(tmp_path: Path) -> Path:
    """Return a temporary vault with a .obsidian directory."""
    obsidian = tmp_path / ".obsidian"
    obsidian.mkdir()
    return tmp_path


@pytest.fixture(autouse=True)
def isolated_bm25_index(tmp_path: Path, monkeypatch):
    """Keep every test off the real BM25 index.

    search() in hybrid mode (the default) opens the on-disk index when no bm25_conn is
    passed, so a test with an empty Chroma collection still got hits from the actual
    vault -- test_empty_collection_returns_empty failed that way for weeks (fixed
    2026-10-07). bm25.py imports the path helper by name, so patch it there.
    """
    from vsearch import bm25

    monkeypatch.setattr(bm25, "get_bm25_db_path", lambda: tmp_path / "bm25-test.db")
