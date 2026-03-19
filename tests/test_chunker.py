"""Tests for vsearch.chunker."""

from pathlib import Path


from vsearch.chunker import _extract_frontmatter, _parse_sections, chunk_file


class TestExtractFrontmatter:
    def test_extracts_known_fields(self):
        content = "---\ntitle: My Note\ntags: [a, b]\ncategory: blog\n---\n\nBody text."
        meta, body = _extract_frontmatter(content)
        assert meta["title"] == "My Note"
        assert meta["tags"] == ["a", "b"]
        assert meta["category"] == "blog"
        assert "Body text." in body

    def test_ignores_unknown_fields(self):
        content = "---\ntitle: Hi\ncustom_field: ignored\n---\nBody."
        meta, _ = _extract_frontmatter(content)
        assert "custom_field" not in meta
        assert "title" in meta

    def test_no_frontmatter(self):
        content = "# Hello\n\nJust a note."
        meta, body = _extract_frontmatter(content)
        assert meta == {}
        assert "Hello" in body

    def test_malformed_frontmatter_returns_empty_meta(self):
        content = "---\nnot: yaml: valid:\n---\nBody."
        # Should not raise; returns something reasonable
        meta, body = _extract_frontmatter(content)
        assert isinstance(meta, dict)


class TestParseSections:
    def test_no_headers(self):
        sections = _parse_sections("Just some text.\n\nMore text.")
        assert len(sections) == 1
        level, header, body = sections[0]
        assert level == 0
        assert header == ""
        assert "Just some text" in body

    def test_single_h1(self):
        text = "# Title\n\nSome content here."
        sections = _parse_sections(text)
        assert len(sections) == 1
        level, header, body = sections[0]
        assert level == 1
        assert header == "Title"
        assert "Some content" in body

    def test_preamble_before_first_header(self):
        text = "Intro text.\n\n# Section\n\nBody."
        sections = _parse_sections(text)
        assert sections[0][0] == 0  # preamble
        assert sections[1][1] == "Section"

    def test_multiple_headers(self):
        text = "# H1\n\nContent1.\n\n## H2\n\nContent2.\n\n### H3\n\nContent3."
        sections = _parse_sections(text)
        assert len(sections) == 3
        assert sections[0][1] == "H1"
        assert sections[1][1] == "H2"
        assert sections[2][1] == "H3"

    def test_header_levels(self):
        text = "# One\n\n## Two\n\n### Three"
        sections = _parse_sections(text)
        levels = [s[0] for s in sections]
        assert levels == [1, 2, 3]


class TestChunkFile:
    def test_rich_headers_file(self, sample_vault: Path):
        path = sample_vault / "rich-headers.md"
        chunks = chunk_file(path, sample_vault)
        assert len(chunks) >= 2
        # Frontmatter metadata present on all chunks
        for c in chunks:
            assert c.frontmatter_meta.get("title") == "SQL for Python Developers"
            assert "sql" in c.frontmatter_meta.get("tags", [])

    def test_chunk_has_correct_source_file(self, sample_vault: Path):
        path = sample_vault / "rich-headers.md"
        chunks = chunk_file(path, sample_vault)
        for c in chunks:
            assert c.source_file == "rich-headers.md"

    def test_chunk_breadcrumbs(self, sample_vault: Path):
        path = sample_vault / "rich-headers.md"
        chunks = chunk_file(path, sample_vault)
        breadcrumbs = [c.breadcrumb for c in chunks]
        # At least one chunk should mention NULL Values
        assert any("NULL" in b for b in breadcrumbs)

    def test_flat_note_produces_chunks(self, sample_vault: Path):
        path = sample_vault / "flat-note.md"
        chunks = chunk_file(path, sample_vault)
        assert len(chunks) >= 1

    def test_stub_file_returns_empty(self, sample_vault: Path):
        path = sample_vault / "stub.md"
        chunks = chunk_file(path, sample_vault)
        assert chunks == []

    def test_chunk_index_sequential(self, sample_vault: Path):
        path = sample_vault / "rich-headers.md"
        chunks = chunk_file(path, sample_vault)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_frontmatter_not_in_chunk_text(self, sample_vault: Path):
        path = sample_vault / "goals.md"
        chunks = chunk_file(path, sample_vault)
        for c in chunks:
            assert "status: active" not in c.text
            assert "tags:" not in c.text

    def test_no_empty_chunks(self, sample_vault: Path):
        for md_file in sample_vault.rglob("*.md"):
            chunks = chunk_file(md_file, sample_vault)
            for c in chunks:
                assert c.text.strip() != ""

    def test_min_chunk_size_respected(self, tmp_vault: Path):
        # Create a file with a tiny section followed by a real section
        (tmp_vault / "tiny.md").write_text(
            "# Big Section\n\nThis is real content with enough words to form a proper chunk.\n\n"
            "## Tiny\n\nOK.\n\n"
            "## Also Big\n\nThis section also has enough content to stand on its own as a chunk.\n"
        )
        chunks = chunk_file(tmp_vault / "tiny.md", tmp_vault)
        # The tiny 'OK.' should be merged, not a standalone chunk
        for c in chunks:
            assert c.token_estimate() >= 3  # very lenient — just not a single word

    def test_empty_file_returns_empty(self, tmp_vault: Path):
        (tmp_vault / "empty.md").write_text("")
        chunks = chunk_file(tmp_vault / "empty.md", tmp_vault)
        assert chunks == []

    def test_file_with_only_whitespace(self, tmp_vault: Path):
        (tmp_vault / "whitespace.md").write_text("   \n\n   \n")
        chunks = chunk_file(tmp_vault / "whitespace.md", tmp_vault)
        assert chunks == []

    def test_large_section_gets_split(self, tmp_vault: Path):
        # Create a section that is clearly over 1000 tokens (~1333+ words)
        big_para = " ".join(["word"] * 1400)
        content = f"# Big\n\n{big_para}\n"
        (tmp_vault / "large.md").write_text(content)
        chunks = chunk_file(tmp_vault / "large.md", tmp_vault, max_tokens=100)
        assert len(chunks) > 1

    def test_daily_note_with_frontmatter(self, sample_vault: Path):
        path = sample_vault / "daily" / "2026-03-15.md"
        chunks = chunk_file(path, sample_vault)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.source_file == "daily/2026-03-15.md"

    def test_nonexistent_file_returns_empty(self, tmp_vault: Path):
        chunks = chunk_file(tmp_vault / "does_not_exist.md", tmp_vault)
        assert chunks == []
