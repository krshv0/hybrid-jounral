"""
Semantic Markdown Chunker

Splits a Markdown document into semantically coherent chunks:
    • Each heading starts a new chunk
    • Paragraph blocks are grouped
    • Bullet / numbered lists are kept together
    • Blockquotes (e.g. journal entries) are treated as one unit
    • Timestamped entries are detected and separated

Target size: RAGConfig.CHUNK_MIN_TOKENS – CHUNK_MAX_TOKENS per chunk.
Large sections are sub-split on double newlines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import RAGConfig


@dataclass
class Chunk:
    """One semantic unit extracted from a document."""
    chunk_id: str           # "<file_stem>__<index>"
    file_path: str          # absolute path to source file
    text: str               # raw text of chunk
    heading: str            # nearest parent heading (or "preamble")
    chunk_type: str         # heading | paragraph | bullets | blockquote | preamble
    index: int              # position in document
    token_estimate: int = field(init=False)

    def __post_init__(self):
        self.token_estimate = _rough_tokens(self.text)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rough_tokens(text: str) -> int:
    """Rough token count: words ÷ 0.75 ≈ GPT tokens."""
    return max(1, int(len(text.split()) / 0.75))


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TIMESTAMP_RE = re.compile(r"(?:Date[_:]?\s*\d|\d{1,2}[\/\-]\d{1,2})", re.IGNORECASE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_INLINE_TAG_RE = re.compile(r"^#\w+(\s+#\w+)*$", re.MULTILINE)


def _strip_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_block, body).  frontmatter may be ''."""
    m = _FRONTMATTER_RE.match(text)
    if m:
        return m.group(0), text[m.end():]
    return "", text


# ── Main chunker ─────────────────────────────────────────────────────────────

class MarkdownChunker:
    """
    Parses a Markdown file and returns a list of Chunk objects.
    Does NOT modify the file.
    """

    def chunk_file(self, file_path: str | Path) -> List[Chunk]:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        return self.chunk_text(text, str(path))

    def chunk_text(self, text: str, file_path: str) -> List[Chunk]:
        file_stem = Path(file_path).stem
        frontmatter, body = _strip_frontmatter(text)
        raw_chunks = self._split_body(body, file_stem, file_path)

        # Sub-split oversized chunks
        final: List[Chunk] = []
        for ch in raw_chunks:
            if ch.token_estimate > RAGConfig.CHUNK_MAX_TOKENS:
                final.extend(self._subsplit(ch, file_stem, file_path))
            elif ch.token_estimate >= RAGConfig.CHUNK_MIN_TOKENS:
                final.append(ch)
            else:
                # Merge tiny chunk into previous if possible
                if final:
                    prev = final[-1]
                    merged_text = prev.text + "\n\n" + ch.text
                    final[-1] = Chunk(
                        chunk_id=prev.chunk_id,
                        file_path=prev.file_path,
                        text=merged_text,
                        heading=prev.heading,
                        chunk_type=prev.chunk_type,
                        index=prev.index,
                    )
                else:
                    final.append(ch)

        # Re-index
        for i, ch in enumerate(final):
            object.__setattr__(ch, "chunk_id", f"{file_stem}__{i}")
            object.__setattr__(ch, "index", i)

        return final

    # ── Body parsing ─────────────────────────────────────────────────────

    def _split_body(self, body: str, file_stem: str, file_path: str) -> List[Chunk]:
        lines = body.split("\n")
        chunks: List[Chunk] = []
        current_lines: List[str] = []
        current_heading = "preamble"
        current_type = "preamble"
        idx = 0

        def flush():
            nonlocal idx
            text = "\n".join(current_lines).strip()
            if text:
                chunks.append(Chunk(
                    chunk_id=f"{file_stem}__{idx}",
                    file_path=file_path,
                    text=text,
                    heading=current_heading,
                    chunk_type=current_type,
                    index=idx,
                ))
                idx += 1
            current_lines.clear()

        i = 0
        while i < len(lines):
            line = lines[i]

            # ── Heading
            hm = _HEADING_RE.match(line)
            if hm:
                flush()
                current_heading = hm.group(2).strip()
                current_type = "heading"
                current_lines.append(line)
                i += 1
                continue

            # ── Horizontal rule  →  just flush, don't include "---"
            if re.match(r"^-{3,}$", line.strip()):
                flush()
                i += 1
                continue

            # ── Inline tags line (e.g. "#thoughts #mindfulness")
            if _INLINE_TAG_RE.match(line.strip()) and line.strip():
                i += 1
                continue

            # ── Blockquote block
            if line.startswith(">"):
                if current_type not in ("blockquote",):
                    flush()
                    current_type = "blockquote"
                block_lines = []
                while i < len(lines) and (lines[i].startswith(">") or lines[i].strip() == ""):
                    if lines[i].startswith(">"):
                        block_lines.append(lines[i].lstrip("> ").rstrip())
                    i += 1
                # Check for embedded timestamp → multiple sub-entries
                self._split_blockquote_by_timestamp(
                    block_lines, file_stem, file_path,
                    current_heading, chunks, idx
                )
                idx += len(self._count_timestamp_splits(block_lines))
                flush()
                current_type = "paragraph"
                continue

            # ── Bullet / numbered list block
            if re.match(r"^(\s*[-*+]|\s*\d+\.)\s", line):
                if current_type not in ("bullets",):
                    flush()
                    current_type = "bullets"
                while i < len(lines) and (
                    re.match(r"^(\s*[-*+]|\s*\d+\.)\s", lines[i]) or
                    (lines[i].startswith("  ") and current_lines)
                ):
                    current_lines.append(lines[i])
                    i += 1
                continue

            # ── Image embeds — skip
            if re.match(r"^!\[\[", line.strip()):
                i += 1
                continue

            # ── Blank line → potential paragraph break
            if line.strip() == "":
                if current_lines:
                    current_lines.append(line)
                i += 1
                continue

            # ── Default paragraph
            if current_type not in ("paragraph", "heading"):
                flush()
                current_type = "paragraph"
            current_lines.append(line)
            i += 1

        flush()
        return chunks

    def _split_blockquote_by_timestamp(
        self,
        lines: List[str],
        file_stem: str,
        file_path: str,
        heading: str,
        chunks: List[Chunk],
        start_idx: int,
    ) -> None:
        """Split a blockquote block on embedded Date_ markers."""
        segments: List[List[str]] = []
        current: List[str] = []
        for line in lines:
            if _TIMESTAMP_RE.search(line) and current:
                segments.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            segments.append(current)

        for j, seg in enumerate(segments):
            text = "\n".join(seg).strip()
            if text and _rough_tokens(text) >= RAGConfig.CHUNK_MIN_TOKENS:
                chunks.append(Chunk(
                    chunk_id=f"{file_stem}__{start_idx + j}",
                    file_path=file_path,
                    text=text,
                    heading=heading,
                    chunk_type="blockquote",
                    index=start_idx + j,
                ))

    def _count_timestamp_splits(self, lines: List[str]) -> List[None]:
        count = 1
        for line in lines:
            if _TIMESTAMP_RE.search(line):
                count += 1
        return [None] * count

    def _subsplit(self, chunk: Chunk, file_stem: str, file_path: str) -> List[Chunk]:
        """Sub-split an oversized chunk on double newlines."""
        paragraphs = re.split(r"\n{2,}", chunk.text)
        result: List[Chunk] = []
        buffer = ""

        for para in paragraphs:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para.strip()
            if _rough_tokens(candidate) <= RAGConfig.CHUNK_MAX_TOKENS:
                buffer = candidate
            else:
                if buffer and _rough_tokens(buffer) >= RAGConfig.CHUNK_MIN_TOKENS:
                    result.append(Chunk(
                        chunk_id=f"{file_stem}__{chunk.index}_{len(result)}",
                        file_path=file_path,
                        text=buffer,
                        heading=chunk.heading,
                        chunk_type=chunk.chunk_type,
                        index=chunk.index,
                    ))
                buffer = para.strip()

        if buffer and _rough_tokens(buffer) >= RAGConfig.CHUNK_MIN_TOKENS:
            result.append(Chunk(
                chunk_id=f"{file_stem}__{chunk.index}_{len(result)}",
                file_path=file_path,
                text=buffer,
                heading=chunk.heading,
                chunk_type=chunk.chunk_type,
                index=chunk.index,
            ))

        return result or [chunk]
