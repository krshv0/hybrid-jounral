"""
Markdown Mutation Engine

Safely writes back to Markdown files:
    • Injects/updates frontmatter (tags, entities)
    • Injects/updates the ## Related Notes section
    • Never overwrites user content
    • All operations are idempotent (safe to run multiple times)
    • Uses structured string operations, not regex on full file

All mutations are logged via the MutationLogger.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from .config import RAGConfig
from .backlink_engine import BacklinkDecision

logger = logging.getLogger(__name__)

# Pattern that matches our managed backlinks section
_BACKLINKS_SECTION_RE = re.compile(
    r"(## Related Notes\s*\n)(.*?)(?=\n##|\Z)",
    re.DOTALL,
)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class MarkdownEditor:
    """
    Provides surgical, idempotent mutations for a single Markdown file.
    Instantiate per-file.
    """

    def __init__(self, file_path: str) -> None:
        self.path = Path(file_path)
        self._text: str = self.path.read_text(encoding="utf-8")

    # ── Public mutations ─────────────────────────────────────────────────

    def update_frontmatter(
        self,
        tags: List[str],
        entities: List[str],
    ) -> bool:
        """
        Update or add `tags` and `entities` keys in YAML frontmatter.
        Returns True if the file was changed.
        """
        if not tags and not entities:
            return False

        if _FRONTMATTER_RE.match(self._text):
            changed = self._update_existing_frontmatter(tags, entities)
        else:
            changed = self._prepend_frontmatter(tags, entities)

        return changed

    def inject_backlinks(self, decisions: List[BacklinkDecision]) -> bool:
        """
        Write (or update) the ## Related Notes section.

        Format:
            ## Related Notes

            - [[Note Title]] — reason for link
            - [[Another Note]] — reason for link

        De-duplicates against already-present links.
        Returns True if the file was changed.
        """
        if not decisions:
            return False

        new_lines = self._build_backlink_lines(decisions)
        if not new_lines:
            return False

        existing_section_match = _BACKLINKS_SECTION_RE.search(self._text)

        if existing_section_match:
            existing_body  = existing_section_match.group(2)
            existing_links = set(re.findall(r"\[\[([^\]]+)\]\]", existing_body))
            new_lines = [
                ln for ln in new_lines
                if not any(f"[[{d.target_title}]]" in ln for d in decisions
                           if d.target_title in existing_links)
            ]
            if not new_lines:
                return False
            updated_body = existing_body.rstrip() + "\n" + "\n".join(new_lines) + "\n"
            self._text = (
                self._text[: existing_section_match.start(2)]
                + updated_body
                + self._text[existing_section_match.end(2):]
            )
        else:
            section = (
                f"\n\n{RAGConfig.BACKLINKS_HEADING}\n\n"
                + "\n".join(new_lines)
                + "\n"
            )
            self._text = self._text.rstrip() + section

        return True

    def save(self) -> None:
        """Write mutated content back to disk."""
        self.path.write_text(self._text, encoding="utf-8")
        logger.debug("Saved: %s", self.path)

    def read_frontmatter_field(self, key: str) -> Optional[str]:
        """Return raw value of a frontmatter key, or None."""
        m = _FRONTMATTER_RE.match(self._text)
        if not m:
            return None
        body = m.group(1)
        for line in body.split("\n"):
            if line.startswith(f"{key}:"):
                return line[len(key) + 1:].strip()
        return None

    def read_frontmatter_list(self, key: str) -> List[str]:
        """
        Return the value of a YAML list frontmatter key as a Python list.
        Handles both inline `key: [a, b]` and block `key:\n  - a` forms.
        Returns an empty list if the key is absent or unparseable.
        """
        raw = self.read_frontmatter_field(key)
        if not raw:
            return []
        # Inline list: [a, b, c]
        if raw.startswith("["):
            inner = raw.strip("[]").strip()
            if not inner:
                return []
            return [item.strip().strip('"\"') for item in inner.split(",") if item.strip()]
        # Scalar (single value)
        return [raw.strip().strip('"\"')] if raw.strip() else []

    def read_existing_backlink_titles(self) -> List[str]:
        """
        Return the list of note titles already linked in the ## Related Notes section.
        Extracts all [[Title]] patterns from that section only.
        """
        m = _BACKLINKS_SECTION_RE.search(self._text)
        if not m:
            return []
        body = m.group(2)
        return re.findall(r"\[\[([^\]]+)\]\]", body)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _build_backlink_lines(self, decisions: List[BacklinkDecision]) -> List[str]:
        lines = []
        for d in decisions:
            reason = d.reason.strip() if d.reason else ""
            if reason:
                lines.append(f"- [[{d.target_title}]] — {reason}")
            else:
                lines.append(f"- [[{d.target_title}]]")
        return lines

    def _update_existing_frontmatter(
        self, tags: List[str], entities: List[str]
    ) -> bool:
        """Modify in-place the tags and entities lines inside existing frontmatter."""
        m = _FRONTMATTER_RE.match(self._text)
        fm_body = m.group(1)
        original_fm = fm_body

        fm_body = self._set_yaml_list(fm_body, "tags", tags)
        fm_body = self._set_yaml_list(fm_body, "entities", entities)

        if fm_body == original_fm:
            return False

        self._text = f"---\n{fm_body}\n---\n" + self._text[m.end():]
        return True

    def _prepend_frontmatter(self, tags: List[str], entities: List[str]) -> bool:
        tags_str = self._yaml_list(tags)
        entities_str = self._yaml_list(entities)
        fm = f"---\ntags: {tags_str}\nentities: {entities_str}\n---\n\n"
        self._text = fm + self._text
        return True

    @staticmethod
    def _yaml_list(items: List[str]) -> str:
        if not items:
            return "[]"
        return "[" + ", ".join(items) + "]"

    @staticmethod
    def _set_yaml_list(fm_body: str, key: str, values: List[str]) -> str:
        """Set or replace a YAML list key in a frontmatter body string."""
        if not values:
            return fm_body

        new_line = f"{key}: [{', '.join(values)}]"
        # Try to replace existing key (handles both scalar and list forms)
        pattern = re.compile(
            rf"^{re.escape(key)}\s*:.*$", re.MULTILINE
        )
        if pattern.search(fm_body):
            return pattern.sub(new_line, fm_body, count=1)
        else:
            return fm_body + f"\n{new_line}"
