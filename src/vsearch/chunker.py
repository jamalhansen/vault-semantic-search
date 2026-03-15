"""Markdown-aware chunking for Obsidian vault files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import frontmatter

from vsearch.config import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS, WORDS_PER_TOKEN


@dataclass
class Chunk:
    text: str
    source_file: str          # relative path from vault root
    breadcrumb: str           # e.g. "Design Principles > CLI Conventions"
    chunk_index: int          # position within the file
    frontmatter_meta: dict = field(default_factory=dict)

    def token_estimate(self) -> int:
        return max(1, int(len(self.text.split()) / WORDS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_FIELDS = {"title", "tags", "category", "status", "date", "created", "modified"}


def _extract_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter and return (meta_dict, body_without_frontmatter)."""
    try:
        post = frontmatter.loads(content)
        meta = {k: v for k, v in post.metadata.items() if k in _FRONTMATTER_FIELDS}
        return meta, post.content
    except Exception:
        return {}, content


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _parse_sections(text: str) -> list[tuple[int, str, str]]:
    """Split text into sections by headers.

    Returns list of (level, header_text, section_body) tuples.
    The section_body does NOT include the header line itself.
    A leading 'preamble' before the first header gets level=0, header=''.
    """
    sections: list[tuple[int, str, str]] = []
    pos = 0
    matches = list(_HEADER_RE.finditer(text))

    if not matches:
        return [(0, "", text.strip())]

    # Preamble before first header
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append((0, "", preamble))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        header = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((level, header, body))

    return sections


# ---------------------------------------------------------------------------
# Breadcrumb tracking
# ---------------------------------------------------------------------------

def _update_breadcrumb(stack: list[tuple[int, str]], level: int, header: str) -> str:
    """Maintain a header stack and return the current breadcrumb string."""
    if level == 0:
        return header
    # Pop any headers at same or deeper level
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, header))
    return " > ".join(h for _, h in stack)


# ---------------------------------------------------------------------------
# Core chunking
# ---------------------------------------------------------------------------

def _words(text: str) -> int:
    return len(text.split())


def _token_estimate(text: str) -> int:
    return max(1, int(_words(text) / WORDS_PER_TOKEN))


def _split_on_paragraphs(text: str, breadcrumb: str) -> list[tuple[str, str]]:
    """Split text on double-newlines, returning (piece, breadcrumb) pairs."""
    paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    return [(p, breadcrumb) for p in paras]


def _split_by_words(text: str, breadcrumb: str, max_tokens: int) -> list[tuple[str, str]]:
    """Split text on word boundaries to enforce max_tokens. Last-resort fallback."""
    max_words = max(1, int(max_tokens * WORDS_PER_TOKEN))
    words = text.split()
    if len(words) <= max_words:
        return [(text, breadcrumb)]
    chunks = []
    for i in range(0, len(words), max_words):
        piece = " ".join(words[i : i + max_words])
        chunks.append((piece, breadcrumb))
    return chunks


def _split_large_section(
    body: str, breadcrumb: str, max_tokens: int
) -> list[tuple[str, str]]:
    """Recursively split a large section body by H3/H4, then paragraphs."""
    if _token_estimate(body) <= max_tokens:
        return [(body, breadcrumb)]

    # Try splitting on H3/H4 first
    sub_sections = _parse_sections(body)
    if len(sub_sections) > 1:
        results = []
        for level, header, sub_body in sub_sections:
            sub_crumb = f"{breadcrumb} > {header}" if header else breadcrumb
            results.extend(_split_large_section(sub_body, sub_crumb, max_tokens))
        return results

    # Fall back to paragraph splitting
    paras = _split_on_paragraphs(body, breadcrumb)
    if len(paras) <= 1:
        # Can't split on paragraphs — split on word boundaries as last resort
        return _split_by_words(body, breadcrumb, max_tokens)

    # Merge small paragraphs into chunks under max_tokens
    chunks: list[tuple[str, str]] = []
    current_parts: list[str] = []
    for para_text, crumb in paras:
        candidate = "\n\n".join(current_parts + [para_text])
        if _token_estimate(candidate) > max_tokens and current_parts:
            chunks.append(("\n\n".join(current_parts), crumb))
            current_parts = [para_text]
        else:
            current_parts.append(para_text)
    if current_parts:
        chunks.append(("\n\n".join(current_parts), breadcrumb))
    return chunks


def chunk_file(
    file_path: Path,
    vault_root: Path,
    max_tokens: int = MAX_CHUNK_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> list[Chunk]:
    """Parse a markdown file and return a list of Chunk objects.

    Frontmatter is stripped from the text and stored as metadata on each chunk.
    """
    try:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if len(raw.strip()) < 50:
        return []

    meta, body = _extract_frontmatter(raw)
    relative_path = str(file_path.relative_to(vault_root))

    sections = _parse_sections(body)

    # Build raw (text, breadcrumb) pieces with splitting
    raw_pieces: list[tuple[str, str]] = []
    breadcrumb_stack: list[tuple[int, str]] = []

    for level, header, section_body in sections:
        if not section_body:
            # Update breadcrumb state even for empty sections
            if level > 0:
                _update_breadcrumb(breadcrumb_stack, level, header)
            continue
        crumb = _update_breadcrumb(breadcrumb_stack, level, header)
        if _token_estimate(section_body) > max_tokens:
            raw_pieces.extend(_split_large_section(section_body, crumb, max_tokens))
        else:
            raw_pieces.append((section_body, crumb))

    # Merge tiny pieces with adjacent pieces.
    # A tiny piece (< min_tokens) is attached to the next piece.
    # Once a piece crosses min_tokens after merging, commit it.
    merged: list[tuple[str, str]] = []
    pending_text: Optional[str] = None
    pending_crumb: Optional[str] = None

    for text, crumb in raw_pieces:
        if pending_text is not None:
            candidate = pending_text + "\n\n" + text
            if _token_estimate(candidate) > max_tokens:
                # Merged would be too large — flush pending as-is, start fresh
                merged.append((pending_text, pending_crumb))
                pending_text = None
                pending_crumb = None
                if _token_estimate(text) < min_tokens:
                    pending_text = text
                    pending_crumb = crumb
                else:
                    merged.append((text, crumb))
            else:
                # Merge is within max_tokens
                if _token_estimate(candidate) >= min_tokens:
                    # Big enough now — commit and reset
                    merged.append((candidate, pending_crumb))
                    pending_text = None
                    pending_crumb = None
                else:
                    # Still too small — keep accumulating
                    pending_text = candidate
        else:
            if _token_estimate(text) < min_tokens:
                pending_text = text
                pending_crumb = crumb
            else:
                merged.append((text, crumb))

    if pending_text:
        if merged:
            # Attach leftover tiny piece to last chunk if it fits
            last_text, last_crumb = merged[-1]
            candidate = last_text + "\n\n" + pending_text
            if _token_estimate(candidate) <= max_tokens:
                merged[-1] = (candidate, last_crumb)
            else:
                merged.append((pending_text, pending_crumb))
        else:
            merged.append((pending_text, pending_crumb))

    # Build Chunk objects
    chunks = []
    for idx, (text, crumb) in enumerate(merged):
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                text=text.strip(),
                source_file=relative_path,
                breadcrumb=crumb or relative_path,
                chunk_index=idx,
                frontmatter_meta=meta,
            )
        )

    return chunks
