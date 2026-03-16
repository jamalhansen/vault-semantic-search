"""Tests for vsearch.indexer."""


import chromadb
import pytest

from vsearch.indexer import (
    _chunk_id,
    _file_hash,
    _should_skip,
    file_needs_reindex,
    index_vault,
    walk_vault,
)
from vsearch.store import get_collection, upsert_chunks


FAKE_DIM = 3


def fake_embed(texts, model="nomic-embed-text", **kwargs):
    return [[0.1, 0.2, 0.3]] * len(texts)


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path / "chromadb"))


@pytest.fixture
def collection(client):
    return get_collection(client, model="nomic-embed-text")


class TestShouldSkip:
    def test_skips_obsidian_dir(self, sample_vault):
        path = sample_vault / ".obsidian" / "config.json"
        # Create temporarily
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}")
        assert _should_skip(path, sample_vault, []) is True

    def test_skips_hidden_files(self, tmp_vault):
        hidden = tmp_vault / ".hidden.md"
        hidden.write_text("# Hidden\n\nContent here for hidden file test.")
        assert _should_skip(hidden, tmp_vault, []) is True

    def test_skips_non_markdown(self, tmp_vault):
        f = tmp_vault / "image.png"
        f.write_bytes(b"\x89PNG")
        assert _should_skip(f, tmp_vault, []) is True

    def test_skips_tiny_files(self, tmp_vault):
        tiny = tmp_vault / "tiny.md"
        tiny.write_text("hi")
        assert _should_skip(tiny, tmp_vault, []) is True

    def test_does_not_skip_normal_markdown(self, tmp_vault):
        normal = tmp_vault / "note.md"
        normal.write_text("# Note\n\nThis is a normal note with enough content to be indexed properly.")
        assert _should_skip(normal, tmp_vault, []) is False

    def test_skips_templates_dir(self, tmp_vault):
        tmpl_dir = tmp_vault / "_templates"
        tmpl_dir.mkdir()
        tmpl = tmpl_dir / "daily.md"
        tmpl.write_text("# {{date}}\n\nTemplate content here that is long enough to pass the size check.")
        assert _should_skip(tmpl, tmp_vault, []) is True

    def test_vsearchignore_pattern(self, tmp_vault):
        f = tmp_vault / "private.md"
        f.write_text("# Private\n\nSome private content that should not be indexed in the vault search.")
        assert _should_skip(f, tmp_vault, ["private*.md"]) is True

    def test_vsearchignore_does_not_skip_non_matching(self, tmp_vault):
        f = tmp_vault / "public.md"
        f.write_text("# Public\n\nSome public content that is fine to be indexed in the vault search.")
        assert _should_skip(f, tmp_vault, ["private*.md"]) is False


class TestWalkVault:
    def test_returns_markdown_files(self, sample_vault):
        files = walk_vault(sample_vault)
        assert all(f.suffix == ".md" for f in files)

    def test_skips_obsidian_dir(self, sample_vault):
        files = walk_vault(sample_vault)
        assert not any(".obsidian" in str(f) for f in files)

    def test_skips_stub_file(self, sample_vault):
        # stub.md has < 50 chars
        files = walk_vault(sample_vault)
        names = [f.name for f in files]
        assert "stub.md" not in names

    def test_finds_nested_daily_notes(self, sample_vault):
        files = walk_vault(sample_vault)
        names = [f.name for f in files]
        assert "2026-03-15.md" in names

    def test_vsearchignore_respected(self, tmp_vault):
        (tmp_vault / "private.md").write_text(
            "# Private\n\nContent that should be excluded from semantic search indexing."
        )
        (tmp_vault / "public.md").write_text(
            "# Public\n\nContent that should be included in semantic search indexing."
        )
        (tmp_vault / ".vsearchignore").write_text("private.md\n")
        files = walk_vault(tmp_vault)
        names = [f.name for f in files]
        assert "private.md" not in names
        assert "public.md" in names


class TestFileNeedsReindex:
    def test_new_file_needs_reindex(self, tmp_vault, collection):
        f = tmp_vault / "new.md"
        f.write_text("# New\n\nContent.")
        assert file_needs_reindex(f, collection, vault_root=tmp_vault) is True

    def test_unchanged_file_does_not_need_reindex(self, tmp_vault, collection):
        f = tmp_vault / "stable.md"
        f.write_text("# Stable\n\nContent.")
        mtime = str(f.stat().st_mtime)
        h = _file_hash(f)
        # Pre-populate collection with matching mtime/hash (using relative path)
        upsert_chunks(
            collection,
            ids=["stable.md::chunk::0"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["Content."],
            metadatas=[{"source_file": "stable.md", "mtime": mtime, "hash": h}],
        )
        assert file_needs_reindex(f, collection, vault_root=tmp_vault) is False

    def test_changed_content_needs_reindex(self, tmp_vault, collection):
        f = tmp_vault / "changed.md"
        f.write_text("# Original\n\nOriginal content.")
        _file_hash(f)
        # Store stale mtime (different from current) so the hash path is exercised
        upsert_chunks(
            collection,
            ids=["changed.md::chunk::0"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["Original."],
            metadatas=[{"source_file": "changed.md", "mtime": "0.0", "hash": "stale_hash"}],
        )
        assert file_needs_reindex(f, collection, vault_root=tmp_vault) is True


class TestIndexVault:
    def test_indexes_all_files(self, sample_vault, collection):
        result = index_vault(
            sample_vault, collection, model="nomic-embed-text", embed_fn=fake_embed
        )
        assert result.indexed > 0
        assert result.errors == 0

    def test_skips_unchanged_files_on_second_run(self, sample_vault, collection):
        # First run: index everything
        index_vault(sample_vault, collection, model="nomic-embed-text", embed_fn=fake_embed)
        initial_count = collection.count()

        # Second run: nothing changed → all skipped
        result = index_vault(
            sample_vault, collection, model="nomic-embed-text", embed_fn=fake_embed
        )
        assert result.indexed == 0
        assert collection.count() == initial_count

    def test_full_flag_reindexes_everything(self, sample_vault, collection):
        index_vault(sample_vault, collection, model="nomic-embed-text", embed_fn=fake_embed)
        result = index_vault(
            sample_vault, collection, model="nomic-embed-text", full=True, embed_fn=fake_embed
        )
        assert result.indexed > 0

    def test_deleted_file_chunks_are_removed(self, tmp_vault, collection):
        f = tmp_vault / "ephemeral.md"
        f.write_text("# Ephemeral\n\nThis file will be deleted after indexing to test cleanup.")
        index_vault(tmp_vault, collection, model="nomic-embed-text", embed_fn=fake_embed)
        assert collection.count() > 0

        # Delete the file and reindex
        f.unlink()
        index_vault(tmp_vault, collection, model="nomic-embed-text", embed_fn=fake_embed)
        assert collection.count() == 0

    def test_chunk_id_format(self):
        cid = _chunk_id("daily/2026-03-15.md", 2)
        assert cid == "daily/2026-03-15.md::chunk::2"

    def test_metadata_stored_on_chunks(self, tmp_vault, collection):
        f = tmp_vault / "meta_test.md"
        f.write_text(
            "---\ntitle: Test Note\ntags: [a, b]\n---\n\n# Test\n\nContent for metadata test."
        )
        index_vault(tmp_vault, collection, model="nomic-embed-text", embed_fn=fake_embed)
        results = collection.get(include=["metadatas"])
        metas = results["metadatas"]
        assert len(metas) > 0
        m = metas[0]
        assert m["source_file"] == "meta_test.md"
        assert "mtime" in m
        assert "hash" in m
